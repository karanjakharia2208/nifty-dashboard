"""Research pass for the 9:35 action call: what else could be changed, and
does the evidence support it? Runs on the same locked inputs as the main
backtest (so nothing here peeks), and every candidate is held to the same
gate. Nothing in this module changes the live call; it only reports.

Candidates
  * leg weights   — technical-only, regression-only, structure-only, and mixes
  * lean filters  — trade only when |score| clears a threshold
  * agreement     — trade only when the 9:35 lean agrees with the morning call
  * exits         — stop to breakeven after +1R, time exits (13:30, 14:30), both
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import actioncall as A

WEIGHT_SETS = {
    "current 50/30/20": dict(A.WEIGHTS),
    "technical only": {"technical": 1.0},
    "regression only": {"regression": 1.0},
    "structure only": {"structure": 1.0},
    "tech-heavy 20/60/20": {"regression": 0.2, "technical": 0.6, "structure": 0.2},
    "reg-heavy 70/20/10": {"regression": 0.7, "technical": 0.2, "structure": 0.1},
    "equal thirds": {"regression": 1 / 3, "technical": 1 / 3, "structure": 1 / 3},
    "regression + structure 60/40": {"regression": 0.6, "structure": 0.4},
    "technical + structure 60/40": {"technical": 0.6, "structure": 0.4},
}
LEAN_FILTERS = {"|score| > 0.10 (current)": 0.10, "|score| > 0.20": 0.20, "|score| > 0.30": 0.30, "|score| > 0.40": 0.40}
EXITS = {
    "as tested (stop or 15:25)": {},
    "stop to breakeven after +1R": {"breakeven_r": 1.0},
    "stop to breakeven after +0.5R": {"breakeven_r": 0.5},
    "exit at 14:30": {"exit_time": dt.time(14, 30)},
    "exit at 13:30": {"exit_time": dt.time(13, 30)},
    "breakeven after +1R and exit 14:30": {"breakeven_r": 1.0, "exit_time": dt.time(14, 30)},
}


def score_from_legs(legs: dict, w: dict) -> float:
    use = {k: w[k] for k in w if k in legs}
    if not use:
        return 0.0
    return float(np.clip(sum(legs[k] * use[k] for k in use) / sum(use.values()), -1, 1))


def _skipped(call: dict, after: pd.DataFrame, why: str) -> dict:
    ev = A.evaluate_action(call, after)
    if ev.get("void"):
        return ev
    ev["result"]["taken"] = {"triggered": False, "result": why, "points": 0.0, "side": call["primary"]["side"]}
    return ev


def managed_outcome(plan: dict, bars: pd.DataFrame, breakeven_r: float | None = None, exit_time: dt.time | None = None) -> dict:
    """Like daycall._plan_outcome, with two optional management rules: move the
    stop to the entry once price has gone `breakeven_r` x risk in favour (from
    the next bar on), and/or flatten at `exit_time`."""
    if exit_time is not None:
        bars = bars[bars.index.time <= exit_time]
    if not len(bars):
        return {"triggered": False, "result": "no trade", "points": 0.0}
    o = float(bars["open"].iloc[0])
    side, trig, stop, tgt = plan["side"], plan["trigger"], plan["stop"], plan["targets"][0]
    sgn = 1 if side == "buy" else -1
    risk_d = abs(trig - stop)
    if not plan.get("touch") and ((side == "buy" and o > trig + 0.5 * risk_d) or (side == "sell" and o < trig - 0.5 * risk_d)):
        return {"triggered": False, "result": "no trade (price ran away before entry)", "points": 0.0}
    entry, entry_t = None, None
    if plan.get("market"):
        entry, entry_t = o, int(bars.index[0].timestamp())
    moved = False
    for t, b in bars.iterrows():
        h, l, c = float(b["high"]), float(b["low"]), float(b["close"])
        if entry is None:
            if side == "buy" and (h >= trig if not plan.get("touch") else l <= trig):
                entry = max(trig, o) if (t == bars.index[0] and not plan.get("touch")) else trig
                entry_t = int(t.timestamp())
            elif side == "sell" and (l <= trig if not plan.get("touch") else h >= trig):
                entry = min(trig, o) if (t == bars.index[0] and not plan.get("touch")) else trig
                entry_t = int(t.timestamp())
            if entry is None:
                continue
        hit_stop = (l <= stop) if side == "buy" else (h >= stop)
        hit_tgt = (h >= tgt) if side == "buy" else (l <= tgt)
        if hit_stop:
            return {"triggered": True, "entry": round(entry, 1), "entry_t": entry_t, "result": "breakeven" if moved and abs(stop - entry) < 1e-6 else "stop",
                    "exit": round(stop, 1), "points": round(sgn * (stop - entry), 1), "exit_t": int(t.timestamp())}
        if hit_tgt:
            return {"triggered": True, "entry": round(entry, 1), "entry_t": entry_t, "result": "target",
                    "exit": tgt, "points": round(sgn * (tgt - entry), 1), "exit_t": int(t.timestamp())}
        if breakeven_r is not None and not moved:
            fav = (h - entry) if side == "buy" else (entry - l)
            if fav >= breakeven_r * risk_d:
                stop, moved = entry, True   # applies from the next bar
    if entry is None:
        return {"triggered": False, "result": "no trade", "points": 0.0}
    c = float(bars["close"].iloc[-1])
    return {"triggered": True, "entry": round(entry, 1), "entry_t": entry_t, "result": "open at close" if exit_time is None else f"exit {exit_time:%H:%M}",
            "exit": round(c, 1), "points": round(sgn * (c - entry), 1), "exit_t": int(bars.index[-1].timestamp())}


def _eval_managed(call: dict, after: pd.DataFrame, **rules) -> dict:
    ev = A.evaluate_action(call, after)
    if ev.get("void") or call["primary"].get("trigger") is None:
        return ev
    p = managed_outcome(call["primary"], after, **rules)
    a = managed_outcome(call["alternate"], after, **rules) if call.get("either_way") else None
    taken, side = p, call["primary"]["side"]
    if a is not None:   # either-way day: whichever breakout triggered first counts, as in the main test
        cand = [(x, s) for x, s in ((p, call["primary"]["side"]), (a, call["alternate"]["side"])) if x["triggered"]]
        if cand:
            taken, side = min(cand, key=lambda xs: xs[0]["entry_t"])
    ev["result"]["primary"] = p
    ev["result"]["taken"] = dict(taken, side=side)
    return ev


def _row(summary: dict, baseline: dict | None = None) -> dict:
    keys = ("days", "trades", "win_rate", "payoff_ratio", "net_points", "net_r", "avg_r_per_trade", "t_stat",
            "max_drawdown_points", "direction_hit_rate", "no_trade_days")
    r = {k: summary.get(k) for k in keys}
    if baseline is not None:
        r["baseline_net_r"] = baseline.get("net_r")
        r["passes"] = A.gate(summary, baseline)["ok"]
    return r


def run(inputs: list, variant: str, day_scores: dict | None = None) -> dict:
    """inputs: list of (inp, bars_after) exactly as the main backtest built them."""
    rng = np.random.default_rng(1)
    out = {"variant": variant, "sessions": len(inputs)}

    # 1. leg weights: rescore every session from its stored legs, same plan
    weights = {}
    for name, w in WEIGHT_SETS.items():
        scores = np.array([score_from_legs(inp["legs"], w) for inp, _ in inputs])
        shuffled = rng.permutation(scores)
        real = [A.evaluate_action(A.build_call(inp, sc, variant=variant), after) for (inp, after), sc in zip(inputs, scores)]
        base = [A.evaluate_action(A.build_call(inp, sc, variant=variant), after) for (inp, after), sc in zip(inputs, shuffled)]
        weights[name] = _row(A.summarise(real), A.summarise(base))
    out["weights"] = weights

    # 2. lean-strength filters on the current weights
    filters = {}
    base_calls = [(A.build_call(inp, variant=variant), after) for inp, after in inputs]
    for name, th in LEAN_FILTERS.items():
        ev = [A.evaluate_action(c, after) if abs(c["score"]) > th else _skipped(c, after, "skipped (weak lean)") for c, after in base_calls]
        filters[name] = _row(A.summarise(ev))
    # 3. agreement with the morning (pre-open) lean, when it is available
    if day_scores:
        def lean(x):
            return 1 if x > 0.15 else -1 if x < -0.15 else 0
        for name, ok in (("agree with morning lean", lambda a, m: m != 0 and a == m),
                         ("agree, or morning neutral", lambda a, m: m == 0 or a == m),
                         ("disagree with morning lean", lambda a, m: m != 0 and a == -m)):
            ev = []
            n_known = 0
            for c, after in base_calls:
                m = day_scores.get(c["date"])
                if m is None:
                    ev.append(_skipped(c, after, "skipped (no morning call)"))
                    continue
                n_known += 1
                a = 1 if c["score"] > 0.1 else -1 if c["score"] < -0.1 else 0
                ev.append(A.evaluate_action(c, after) if (a != 0 and ok(a, lean(m))) else _skipped(c, after, "skipped (filter)"))
            r = _row(A.summarise(ev))
            r["sessions_with_morning_call"] = n_known
            filters[name] = r
    out["filters"] = filters

    # 4. exits / trade management on the current weights
    exits = {}
    for name, rules in EXITS.items():
        ev = [_eval_managed(c, after, **rules) for c, after in base_calls]
        exits[name] = _row(A.summarise(ev))
    out["exits"] = exits

    out["note"] = ("Every row uses the same 59-session walk-forward inputs as the main test and the same plan variant; "
                   "only the named thing changes. 'passes' applies the full gate against a shuffled-score baseline "
                   "(weights rows only). Rows without a baseline are compared with the 'as tested' row.")
    return out


def dataset(inputs: list, day_scores: dict | None = None) -> dict:
    """Compact export of the locked inputs and the bars that followed, so weight
    and exit ideas can be re-tested without re-downloading market data."""
    rows = []
    for inp, after in inputs:
        rows.append({"date": inp["date"], "legs": inp["legs"], "score": inp["score"], "spot": inp["spot"], "open": inp["open"],
                     "prev_close": inp["prev_close"], "orb_high": inp["orb_high"], "orb_low": inp["orb_low"],
                     "atr_daily": inp["atr_daily"], "atr_5m": inp["atr_5m"], "vix_ratio": inp["vix_ratio"], "gap_pct": inp["gap_pct"],
                     "regression_detail": inp.get("regression_detail"),
                     "morning_score": (day_scores or {}).get(inp["date"]),
                     "bars_after": [[int(t.timestamp()), round(float(b["open"]), 2), round(float(b["high"]), 2),
                                     round(float(b["low"]), 2), round(float(b["close"]), 2)] for t, b in after.iterrows()]})
    return {"generated_at": int(pd.Timestamp.now(tz="Asia/Kolkata").timestamp()), "lock_bar": "09:30", "sessions": rows,
            "columns": "bars_after = [unix time, open, high, low, close] 5-minute bars from 09:35 to the close"}
