"""The 9:35 action call: a second, tradeable call per day built from the
opening range and the intraday factor model — the only component that has
shown predictive content — and scored at the close on 5-minute bars.

Inputs known at lock time (first run at or after 09:35 IST):
  * the intraday predictive regression's 1h/2h view (fitted on earlier sessions)
  * the intraday indicator votes (VWAP, EMAs, RSI... on the bars so far)
  * opening structure: where price sits relative to the open and the first
    15-minute range (ORB), scaled by the 5-minute ATR
Score = 0.5 * regression + 0.3 * intraday technicals + 0.2 * structure.

Plan: break of the opening range in the scored direction, risk capped at
0.35 daily ATR (never less than the ORB height), targets at 1R and 2R; on a
neutral score both breakout plans are listed and the first to trigger counts.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import indicators as ti
from daycall import _plan_outcome, KEEP_DAYS

LOCK_TIME = dt.time(9, 32)      # earliest lock: needs the 9:15, 9:20, 9:25, 9:30 bars
ORB_BARS = 3                    # 09:15–09:30 opening range
WEIGHTS = {"regression": 0.5, "technical": 0.3, "structure": 0.2}


def _label(score):
    if score > 0.1:
        return "Bullish", "BUY BREAKOUT"
    if score < -0.1:
        return "Bearish", "SELL BREAKDOWN"
    return "Neutral", "BREAKOUT EITHER WAY"


def action_inputs(now: pd.Timestamp, tb: pd.DataFrame, nd_hist: pd.DataFrame, tech: dict, im: dict,
                  vix_ratio: float, prev_close: float) -> dict | None:
    """Everything the call needs, computed from data available at `now`."""
    if tb is None or len(tb) < 1:
        return None
    orb = tb.iloc[:ORB_BARS] if len(tb) > ORB_BARS else tb
    orb_h, orb_l = float(orb["high"].max()), float(orb["low"].min())
    o, spot = float(tb["open"].iloc[0]), float(tb["close"].iloc[-1])
    atr_d = float(ti.atr(nd_hist, 14).dropna().iloc[-1])
    atr5 = float(tech["_ind5"]["atr"]) if "_ind5" in tech else atr_d / 11
    legs = {}
    if im.get("ok") and im.get("signals"):
        zs = [s["z"] for s in im["signals"].values()]
        legs["regression"] = float(np.tanh(1.5 * np.mean(zs)))
    legs["technical"] = float(tech["intraday"]["score"])
    legs["structure"] = float(np.tanh((spot - o) / (1.5 * atr5)))
    w = {k: WEIGHTS[k] for k in legs}
    score = float(np.clip(sum(legs[k] * w[k] for k in legs) / sum(w.values()), -1, 1))
    return {"date": now.strftime("%Y-%m-%d"), "locked_at": int(tb.index[-1].timestamp()) + 300,
            "locked_at_ist": now.strftime("%H:%M"), "bars_at_lock": int(len(tb)),
            "open": round(o, 2), "spot": round(spot, 2), "prev_close": round(prev_close, 2),
            "orb_high": round(orb_h, 1), "orb_low": round(orb_l, 1), "atr_daily": round(atr_d, 1), "atr_5m": round(atr5, 1),
            "vix_ratio": round(float(np.clip(vix_ratio, 0.7, 1.6)), 2), "gap_pct": round((o / prev_close - 1) * 100, 2),
            "legs": {k: round(v, 3) for k, v in legs.items()}, "score": round(score, 3),
            "regression_detail": {h: {"pred_points": round(s["pred_points"], 1), "z": round(s["z"], 2)} for h, s in (im.get("signals") or {}).items()}}


VARIANTS = {
    # how the scored direction is turned into a trade; the backtest picks the one that works.
    # Only variants whose first target is at least 2x the risk (or that let winners run to the
    # close) may be shown as trade levels; the 1R / 1.5R rows stay in the test as reference.
    "breakout_1r": {"text": "break of the opening range in the scored direction; stop ≤ 0.35 ATR; target 1R then 2R", "eligible": False},
    "breakout_2r": {"text": "break of the opening range in the scored direction; stop ≤ 0.35 ATR; target 2R then 3R", "eligible": True},
    "direction_hold": {"text": "enter at the lock price in the scored direction; stop 0.35 ATR; no target — hold to the close", "eligible": True},
    "direction_1_5r": {"text": "enter at the lock price in the scored direction; stop 0.35 ATR; target 1.5R then 3R", "eligible": False},
    "direction_2r": {"text": "enter at the lock price in the scored direction; stop 0.35 ATR; target 2R then 3R", "eligible": True},
}
DEFAULT_VARIANT = "breakout_2r"      # tracked (levels hidden) when nothing clears the gate


def _plan(side, trig, risk, tmult, market, spot, hold=False):
    sgn = 1 if side == "buy" else -1
    stop = trig - sgn * risk
    t1 = trig + sgn * risk * (tmult if not hold else 10.0)
    t2 = trig + sgn * risk * (tmult * 1.5 if tmult < 2 else tmult + 1) if not hold else t1
    verb = ("Buy" if side == "buy" else "Sell")
    where = f"{verb} now near {spot:,.0f}" if market else (f"{verb} above {trig:,.0f}" if side == "buy" else f"{verb} below {trig:,.0f}")
    tgt_txt = "hold to the close" if hold else f"targets {t1:,.0f} then {t2:,.0f}"
    return {"side": side, "trigger": round(trig, 1), "stop": round(stop, 1), "targets": [round(t1, 1), round(t2, 1)],
            "risk": round(risk, 1), "market": bool(market), "hold": bool(hold), "text": f"{where}; stop {stop:,.0f}; {tgt_txt}"}


def build_call(inp: dict, score: float | None = None, variant: str = DEFAULT_VARIANT) -> dict:
    s = float(inp["score"] if score is None else score)
    bias, stance = _label(s)
    spot, orb_h, orb_l, atr_d, atr5, vr = inp["spot"], inp["orb_high"], inp["orb_low"], inp["atr_daily"], inp["atr_5m"], inp["vix_ratio"]
    direction_variant = variant.startswith("direction")
    if direction_variant:
        risk = 0.35 * atr_d
        hold = variant == "direction_hold"
        tmult = {"direction_1_5r": 1.5, "direction_2r": 2.0}.get(variant, 1.0)
        buy = _plan("buy", spot, risk, tmult, True, spot, hold)
        sell = _plan("sell", spot, risk, tmult, True, spot, hold)
    else:
        risk = min(0.35 * atr_d, max(orb_h - orb_l, 2 * atr5))
        tmult = 2.0 if variant == "breakout_2r" else 1.0
        bt = max(orb_h + 0.1 * atr5, spot) if spot > orb_h + 0.1 * atr5 else orb_h + 0.1 * atr5
        st = min(orb_l - 0.1 * atr5, spot) if spot < orb_l - 0.1 * atr5 else orb_l - 0.1 * atr5
        buy = _plan("buy", bt, risk, tmult, spot >= bt, spot)
        sell = _plan("sell", st, risk, tmult, spot <= st, spot)
    # rest-of-day expected range
    rem = max(0.1, 1 - inp["bars_at_lock"] / 75)
    half = 0.6 * atr_d * vr * np.sqrt(rem)
    center = spot + s * 0.3 * atr_d * rem
    either_way = abs(s) <= 0.1
    if s > 0.1:
        primary, alternate = buy, sell
        advice = (f"Lean long. {buy['text']}. Opening range {orb_l:,.0f}–{orb_h:,.0f}. "
                  + ("" if direction_variant else f"Take the sell plan only if {sell['trigger']:,.0f} gives way first."))
    elif s < -0.1:
        primary, alternate = sell, buy
        advice = (f"Lean short. {sell['text']}. Opening range {orb_l:,.0f}–{orb_h:,.0f}. "
                  + ("" if direction_variant else f"Take the buy plan only if {buy['trigger']:,.0f} gives way first."))
    else:
        primary, alternate = buy, sell
        if direction_variant:
            either_way = False
            primary = dict(buy, trigger=None, text="No trade: the model has no lean at the lock")
            advice = "No lean at the lock: no trade today under this plan."
        else:
            advice = (f"No lean yet: trade the break of the opening range {orb_l:,.0f}–{orb_h:,.0f} in whichever direction comes first — "
                      f"{buy['text']}, or {sell['text']}. Skip the day if neither triggers by 11:30.")
    return dict(inp, **{"bias": bias, "stance": stance, "score": round(s, 3), "variant": variant,
                        "variant_text": VARIANTS.get(variant, {}).get("text", ""),
                        "confidence": round(float(np.clip(0.35 + 0.5 * abs(s), 0, 1)), 2),
                        "range_low": round(center - half, 1), "range_high": round(center + half, 1), "range_center": round(center, 1),
                        "primary": primary, "alternate": alternate, "reversal_plan": not direction_variant,
                        "either_way": bool(either_way), "advice": advice, "evaluated": False})


def evaluate_action(call: dict, bars_after: pd.DataFrame) -> dict:
    if not len(bars_after):
        return dict(call, void=True, void_reason="no bars after the lock")
    c = float(bars_after["close"].iloc[-1])
    h, l = float(bars_after["high"].max()), float(bars_after["low"].min())
    direction = 1 if c > call["spot"] else -1 if c < call["spot"] else 0
    pred = 1 if call["score"] > 0.1 else -1 if call["score"] < -0.1 else 0
    if call["primary"].get("trigger") is None:
        p = {"triggered": False, "result": "no trade", "points": 0.0}
    else:
        p = _plan_outcome(call["primary"], bars_after)
    a = _plan_outcome(call["alternate"], bars_after)
    if call.get("either_way"):
        # whichever triggered first counts
        cand = [x for x in (p, a) if x["triggered"]]
        taken = min(cand, key=lambda x: x["entry_t"]) if cand else p
        taken_side = call["primary"]["side"] if taken is p else call["alternate"]["side"]
    else:
        taken, taken_side = p, call["primary"]["side"]
    res = {"direction_actual": "up" if direction > 0 else "down" if direction < 0 else "flat",
           "direction_hit": (direction == pred) if pred else None,
           "move_from_lock": round(c - call["spot"], 1),
           "range_contained": bool(l >= call["range_low"] and h <= call["range_high"]),
           "close_in_range": bool(call["range_low"] <= c <= call["range_high"]),
           "primary": p, "alternate": a, "taken": dict(taken, side=taken_side)}
    return dict(call, evaluated=True, result=res, actual={"close": round(c, 2), "high_after": round(h, 2), "low_after": round(l, 2)},
                evaluated_at=int(bars_after.index[-1].timestamp()))


def summarise(calls: list[dict]) -> dict:
    ev = [c for c in calls if c.get("evaluated") and not c.get("void")]
    if not ev:
        return {"days": 0}
    d = [c for c in ev if c["result"]["direction_hit"] is not None]
    tr = [c for c in ev if c["result"]["taken"]["triggered"]]
    # a trade still open at the close counts by its mark-to-close points, so hold-to-close plans are scored too
    wins = [c for c in tr if c["result"]["taken"]["points"] > 0]
    losses = [c for c in tr if c["result"]["taken"]["points"] <= 0]
    pts = [c["result"]["taken"]["points"] for c in tr]
    r_pts = [c["result"]["taken"]["points"] / max(float(c["primary"]["risk"]), 1.0) for c in tr]
    win_pts = [p for p in pts if p > 0]
    loss_pts = [p for p in pts if p <= 0]
    # t-statistic of the per-trade R: how many standard errors the average trade sits above zero
    t_stat = round(float(np.mean(r_pts) / (np.std(r_pts, ddof=1) / np.sqrt(len(r_pts)))), 2) if len(r_pts) >= 3 and np.std(r_pts, ddof=1) > 0 else None
    eq = np.cumsum(pts) if pts else np.array([0.0])
    return {"days": len(ev), "window": [ev[0]["date"], ev[-1]["date"]],
            "avg_win_points": round(float(np.mean(win_pts)), 1) if win_pts else None,
            "avg_loss_points": round(float(np.mean(loss_pts)), 1) if loss_pts else None,
            "payoff_ratio": round(float(np.mean(win_pts) / abs(np.mean(loss_pts))), 2) if win_pts and loss_pts and np.mean(loss_pts) != 0 else None,
            "t_stat": t_stat, "max_drawdown_points": round(float((np.maximum.accumulate(eq) - eq).max()), 1),
            "direction_calls": len(d), "direction_hit_rate": round(sum(1 for c in d if c["result"]["direction_hit"]) / len(d), 3) if d else None,
            "range_contained_rate": round(sum(1 for c in ev if c["result"]["range_contained"]) / len(ev), 3),
            "close_in_range_rate": round(sum(1 for c in ev if c["result"]["close_in_range"]) / len(ev), 3),
            "trades": len(tr), "wins": len(wins), "losses": len(losses),
            "open_at_close": sum(1 for c in tr if c["result"]["taken"]["result"] == "open at close"),
            "no_trade_days": len(ev) - len(tr), "win_rate": round(len(wins) / (len(wins) + len(losses)), 3) if (wins or losses) else None,
            "net_points": round(float(sum(pts)), 1), "avg_points_per_trade": round(float(np.mean(pts)), 1) if pts else None,
            "net_r": round(float(sum(r_pts)), 2), "avg_r_per_trade": round(float(np.mean(r_pts)), 3) if r_pts else None}


def update_log(actions: list[dict], now: pd.Timestamp, s5: pd.DataFrame, market_status: str, make_inputs,
               variant: str = DEFAULT_VARIANT) -> list[dict]:
    """Lock today's action call once (from 09:32), evaluate finished sessions."""
    actions = list(actions)
    by_date = {c["date"]: c for c in actions}
    today = now.strftime("%Y-%m-%d")
    days = s5.index.normalize()
    for c in actions:
        if c.get("evaluated") or c.get("void"):
            continue
        d = pd.Timestamp(c["date"]).tz_localize(s5.index.tz)
        bars = s5[days == d]
        session_over = (c["date"] < today) or (market_status in ("closed", "closed-weekend") and now.time() >= dt.time(15, 30))
        if not session_over:
            continue
        after = bars[bars.index >= pd.Timestamp(c["locked_at"], unit="s", tz=s5.index.tz)]
        if len(bars) >= 60:
            c.update(evaluate_action(c, after))
        elif c["date"] < today or now.time() >= dt.time(16, 0):
            c["void"], c["void_reason"] = True, "no session data (holiday or feed gap)"
    if today not in by_date and now.weekday() < 5 and market_status == "open" and now.time() >= LOCK_TIME:
        inp = make_inputs()
        if inp:
            actions.append(build_call(inp, variant=variant))
    return sorted(actions, key=lambda c: c["date"])[-KEEP_DAYS:]


def gate(s: dict, b: dict) -> dict:
    """A plan may show trade levels only if its walk-forward test clears all of these."""
    if not s or not s.get("days"):
        return {"ok": False, "checks": [[False, "no test yet"]]}
    tr = s.get("trades") or 0
    checks = [
        [s["days"] >= 30, f"{s['days']} sessions tested (need 30)"],
        [tr >= 20, f"{tr} trades (need 20)"],
        [(s.get("net_r") or 0) > (b.get("net_r") or 0), f"net R {s.get('net_r')} vs random {b.get('net_r')} (must beat random)"],
        [(s.get("t_stat") or 0) >= 2.0, f"t-stat {s.get('t_stat')} (need 2.0: average trade at least two standard errors above zero)"],
        [(s.get("avg_r_per_trade") or 0) >= 0.1, f"avg {s.get('avg_r_per_trade')} R per trade (need 0.10)"],
        [(s.get("direction_hit_rate") or 0) >= 0.55, f"direction right {round((s.get('direction_hit_rate') or 0) * 100)}% (need 55%)"],
        [(s.get("payoff_ratio") or 0) >= 1.5, f"average win {s.get('avg_win_points')} vs average loss {s.get('avg_loss_points')} pts (need wins ≥ 1.5x losses)"],
    ]
    return {"ok": all(c[0] for c in checks), "checks": checks}


# --------------------------------------------------------------- backtest --
LOCK_SWEEP = {"09:20": dt.time(9, 15), "09:25": dt.time(9, 20), "09:30": dt.time(9, 25), "09:35": dt.time(9, 30),
              "09:45": dt.time(9, 40), "10:00": dt.time(9, 55)}


def backtest(s5: pd.DataFrame, nd: pd.DataFrame, daily: dict, intraday: dict, technical_score, intraday_model,
             max_sessions: int = 60, sweep: bool = True) -> dict:
    """Walk-forward over the available 5-minute history: lock at 09:35 with
    only data up to then, score at the close. A shuffled-score baseline uses
    the same opening-range plans with the day's scores permuted. Also sweeps
    other lock times (09:20 … 10:00) so the lock time is chosen by evidence."""
    main = _backtest_at(s5, nd, daily, intraday, technical_score, intraday_model, dt.time(9, 30), max_sessions)
    if not sweep:
        return main
    main["sweep_generated_at"] = int(pd.Timestamp.now(tz="Asia/Kolkata").timestamp())
    sweep = {}
    for label, last_bar in LOCK_SWEEP.items():
        if last_bar == dt.time(9, 30):
            r = main
        else:
            try:
                r = _backtest_at(s5, nd, daily, intraday, technical_score, intraday_model, last_bar, max_sessions)
            except Exception as e:
                r = {"error": f"{type(e).__name__}: {e}"}
        if r.get("summary"):
            s, b = r["summary"], r["baseline"]
            sweep[label] = {"days": s["days"], "direction_hit_rate": s["direction_hit_rate"], "win_rate": s["win_rate"],
                            "trades": s["trades"], "net_points": s["net_points"], "net_r": s["net_r"],
                            "baseline_net_r": b.get("net_r"), "baseline_direction_hit_rate": b.get("direction_hit_rate"),
                            "edge_r": round((s["net_r"] or 0) - (b.get("net_r") or 0), 2)}
        else:
            sweep[label] = {"error": r.get("error", "no result")}
    main["lock_time_sweep"] = sweep
    return main


def _backtest_at(s5, nd, daily, intraday, technical_score, intraday_model, last_bar_time: dt.time, max_sessions: int) -> dict:
    days = sorted(set(s5.index.normalize()))
    test_days = days[-(max_sessions + 1):-1]
    inputs = []
    for d in test_days:
        sess = s5[s5.index.normalize() == d]
        if len(sess) < 60:
            continue
        cut = sess[sess.index.time <= last_bar_time]
        if len(cut) < 1:
            continue
        t_now = cut.index[-1] + pd.Timedelta(minutes=5)
        hist5 = s5[s5.index < t_now]
        histd = nd[nd.index.normalize() < d]
        if len(histd) < 220:
            continue
        prev = s5[s5.index.normalize() < d]
        prev_close = float(prev["close"].iloc[-1]) if len(prev) else float(sess["open"].iloc[0])
        vr = 1.0
        if "INDIAVIX" in daily:
            v = daily["INDIAVIX"]["close"]
            v = v[v.index.normalize() < d]
            if len(v) > 25:
                vr = float(v.iloc[-1] / v.tail(21).mean())
        try:
            tech = technical_score(hist5, histd, None, None)
            im = intraday_model(hist5, {k: v[v.index < t_now] for k, v in intraday.items()}, asof=t_now)
            inp = action_inputs(t_now, cut, histd, tech, im, vr, prev_close)
        except Exception:
            inp = None
        if inp:
            inputs.append((inp, sess[sess.index >= t_now]))
    if len(inputs) < 10:
        return {"error": f"only {len(inputs)} usable sessions"}
    scores = np.array([inp["score"] for inp, _ in inputs])
    shuffled = np.random.default_rng(0).permutation(scores)
    variants = {}
    for v, meta in VARIANTS.items():
        real_v = [evaluate_action(build_call(inp, variant=v), after) for inp, after in inputs]
        base_v = [evaluate_action(build_call(inp, sc, variant=v), after) for (inp, after), sc in zip(inputs, shuffled)]
        sv, bv = summarise(real_v), summarise(base_v)
        variants[v] = {"description": meta["text"], "eligible": meta["eligible"], "summary": sv, "baseline": bv,
                       "edge_r": round((sv.get("net_r") or 0) - (bv.get("net_r") or 0), 2),
                       "passes": gate(sv, bv)["ok"]}
    # among eligible variants that clear the gate, prefer the most statistically solid (t-stat), then net R
    chosen = max((v for v in variants if variants[v]["passes"] and variants[v]["eligible"]),
                 key=lambda v: ((variants[v]["summary"].get("t_stat") or -1e9), (variants[v]["summary"].get("net_r") or -1e9)), default=None)
    show = chosen or DEFAULT_VARIANT
    real = [evaluate_action(build_call(inp, variant=show), after) for inp, after in inputs]
    return {"summary": variants[show]["summary"], "baseline": variants[show]["baseline"], "variants": variants,
            "chosen_variant": chosen, "shown_variant": show, "gate": gate(variants[show]["summary"], variants[show]["baseline"]),
            "records": [{k: c[k] for k in ("date", "score", "stance", "spot", "orb_high", "orb_low")} | {
                "taken_side": c["result"]["taken"]["side"], "result": c["result"]["taken"]["result"],
                "points": c["result"]["taken"]["points"], "direction_hit": c["result"]["direction_hit"]} for c in real],
            "generated_at": int(pd.Timestamp.now(tz="Asia/Kolkata").timestamp()),
            "method": "walk-forward over the available 5-minute history; locked at 09:35 with data up to then; the same "
                      "direction score is traded five ways (opening-range break or entry at the lock; 1R/1.5R/2R targets or "
                      "hold to the close), risk capped at 0.35 daily ATR; scored on 5-minute bars (same-bar stop/target "
                      "conflicts count as a stop); baseline = same plans with the daily scores shuffled; only a variant whose "
                      "first target is at least 2R (or that holds to the close) may be shown, and only if it clears every gate check"}
