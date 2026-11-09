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
    if tb is None or len(tb) < ORB_BARS + 1:
        return None
    orb = tb.iloc[:ORB_BARS]
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


def build_call(inp: dict, score: float | None = None) -> dict:
    s = float(inp["score"] if score is None else score)
    bias, stance = _label(s)
    spot, orb_h, orb_l, atr_d, atr5, vr = inp["spot"], inp["orb_high"], inp["orb_low"], inp["atr_daily"], inp["atr_5m"], inp["vix_ratio"]
    risk = min(0.35 * atr_d, max(orb_h - orb_l, 2 * atr5))
    bt = max(orb_h + 0.1 * atr5, spot) if spot > orb_h + 0.1 * atr5 else orb_h + 0.1 * atr5
    st = min(orb_l - 0.1 * atr5, spot) if spot < orb_l - 0.1 * atr5 else orb_l - 0.1 * atr5
    buy = {"side": "buy", "trigger": round(bt, 1), "stop": round(bt - risk, 1), "targets": [round(bt + risk, 1), round(bt + 2 * risk, 1)],
           "risk": round(risk, 1), "market": bool(spot >= bt),
           "text": (f"Buy now near {spot:,.0f}" if spot >= bt else f"Buy above {bt:,.0f}") + f"; stop {bt - risk:,.0f}; targets {bt + risk:,.0f} then {bt + 2 * risk:,.0f}"}
    sell = {"side": "sell", "trigger": round(st, 1), "stop": round(st + risk, 1), "targets": [round(st - risk, 1), round(st - 2 * risk, 1)],
            "risk": round(risk, 1), "market": bool(spot <= st),
            "text": (f"Sell now near {spot:,.0f}" if spot <= st else f"Sell below {st:,.0f}") + f"; stop {st + risk:,.0f}; targets {st - risk:,.0f} then {st - 2 * risk:,.0f}"}
    # rest-of-day expected range
    rem = max(0.1, 1 - inp["bars_at_lock"] / 75)
    half = 0.6 * atr_d * vr * np.sqrt(rem)
    center = spot + s * 0.3 * atr_d * rem
    if s > 0.1:
        primary, alternate = buy, sell
        advice = (f"Lean long. {buy['text']}. Opening range {orb_l:,.0f}–{orb_h:,.0f}; a close back inside it after entry is the warning sign. "
                  f"Take the sell plan only if {sell['trigger']:,.0f} gives way first.")
    elif s < -0.1:
        primary, alternate = sell, buy
        advice = (f"Lean short. {sell['text']}. Opening range {orb_l:,.0f}–{orb_h:,.0f}; a close back inside it after entry is the warning sign. "
                  f"Take the buy plan only if {buy['trigger']:,.0f} gives way first.")
    else:
        primary, alternate = buy, sell
        advice = (f"No lean yet: trade the break of the opening range {orb_l:,.0f}–{orb_h:,.0f} in whichever direction comes first — "
                  f"{buy['text']}, or {sell['text']}. Skip the day if neither triggers by 11:30.")
    return dict(inp, **{"bias": bias, "stance": stance, "score": round(s, 3),
                        "confidence": round(float(np.clip(0.35 + 0.5 * abs(s), 0, 1)), 2),
                        "range_low": round(center - half, 1), "range_high": round(center + half, 1), "range_center": round(center, 1),
                        "primary": primary, "alternate": alternate, "either_way": bool(abs(s) <= 0.1), "advice": advice, "evaluated": False})


def evaluate_action(call: dict, bars_after: pd.DataFrame) -> dict:
    if not len(bars_after):
        return dict(call, void=True, void_reason="no bars after the lock")
    c = float(bars_after["close"].iloc[-1])
    h, l = float(bars_after["high"].max()), float(bars_after["low"].min())
    direction = 1 if c > call["spot"] else -1 if c < call["spot"] else 0
    pred = 1 if call["score"] > 0.1 else -1 if call["score"] < -0.1 else 0
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
    wins = [c for c in tr if c["result"]["taken"]["result"] == "target"]
    losses = [c for c in tr if c["result"]["taken"]["result"] == "stop"]
    pts = [c["result"]["taken"]["points"] for c in tr]
    r_pts = [c["result"]["taken"]["points"] / max(float(c["primary"]["risk"]), 1.0) for c in tr]
    return {"days": len(ev), "window": [ev[0]["date"], ev[-1]["date"]],
            "direction_calls": len(d), "direction_hit_rate": round(sum(1 for c in d if c["result"]["direction_hit"]) / len(d), 3) if d else None,
            "range_contained_rate": round(sum(1 for c in ev if c["result"]["range_contained"]) / len(ev), 3),
            "close_in_range_rate": round(sum(1 for c in ev if c["result"]["close_in_range"]) / len(ev), 3),
            "trades": len(tr), "wins": len(wins), "losses": len(losses), "open_at_close": len(tr) - len(wins) - len(losses),
            "no_trade_days": len(ev) - len(tr), "win_rate": round(len(wins) / (len(wins) + len(losses)), 3) if (wins or losses) else None,
            "net_points": round(float(sum(pts)), 1), "avg_points_per_trade": round(float(np.mean(pts)), 1) if pts else None,
            "net_r": round(float(sum(r_pts)), 2), "avg_r_per_trade": round(float(np.mean(r_pts)), 3) if r_pts else None}


def update_log(actions: list[dict], now: pd.Timestamp, s5: pd.DataFrame, market_status: str, make_inputs) -> list[dict]:
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
            actions.append(build_call(inp))
    return sorted(actions, key=lambda c: c["date"])[-KEEP_DAYS:]


# --------------------------------------------------------------- backtest --
def backtest(s5: pd.DataFrame, nd: pd.DataFrame, daily: dict, intraday: dict, technical_score, intraday_model,
             max_sessions: int = 60) -> dict:
    """Walk-forward over the available 5-minute history: lock at 09:35 with
    only data up to then, score at the close. A shuffled-score baseline uses
    the same opening-range plans with the day's scores permuted."""
    days = sorted(set(s5.index.normalize()))
    test_days = days[-(max_sessions + 1):-1]
    inputs = []
    for d in test_days:
        sess = s5[s5.index.normalize() == d]
        if len(sess) < 60:
            continue
        cut = sess[sess.index.time <= dt.time(9, 30)]
        if len(cut) < ORB_BARS + 1:
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
    real = [evaluate_action(build_call(inp), after) for inp, after in inputs]
    scores = np.array([inp["score"] for inp, _ in inputs])
    shuffled = np.random.default_rng(0).permutation(scores)
    base = [evaluate_action(build_call(inp, sc), after) for (inp, after), sc in zip(inputs, shuffled)]
    return {"summary": summarise(real), "baseline": summarise(base),
            "records": [{k: c[k] for k in ("date", "score", "stance", "spot", "orb_high", "orb_low")} | {
                "taken_side": c["result"]["taken"]["side"], "result": c["result"]["taken"]["result"],
                "points": c["result"]["taken"]["points"], "direction_hit": c["result"]["direction_hit"]} for c in real],
            "generated_at": int(pd.Timestamp.now(tz="Asia/Kolkata").timestamp()),
            "method": "walk-forward over the available 5-minute history; locked at 09:35 with data up to then; "
                      "opening-range breakout plans, risk capped at 0.35 daily ATR, first target 1R; scored on 5-minute bars "
                      "(same-bar stop/target conflicts count as a stop); baseline = same plans with the daily scores shuffled"}
