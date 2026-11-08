"""The day call: one prediction of record per trading day, locked on the first
engine run of the day, evaluated after the close, kept in a rolling log so the
dashboard can show how often it was right.

A call carries:
  * expected range for the session (low / high) and the expected open
  * bias (bullish / bearish / neutral) with its score and confidence
  * a primary trade plan (buy above X or sell below Y, with stop and targets)
    plus the opposite plan in case the day goes the other way, or a range
    plan on neutral days
Evaluation uses the session's 5-minute bars: direction (close vs open),
range containment and coverage, and whether the plan triggered and then hit
its first target or its stop (same-bar conflicts count as a stop).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import indicators as ti

KEEP_DAYS = 120


def _label(score: float) -> str:
    bias = "Bullish" if score > 0.15 else "Bearish" if score < -0.15 else "Neutral / range"
    return ("Strongly " + bias.lower()) if abs(score) > 0.45 else bias


def make_day_call(now: pd.Timestamp, tb_today: pd.DataFrame, nd_hist: pd.DataFrame, tech: dict, dm: dict,
                  gm: dict, im: dict, zones: dict, vix_ratio: float, spot: float, prev_close: float,
                  fund: float | None = None) -> dict:
    date = now.strftime("%Y-%m-%d")
    atr_d = float(ti.atr(nd_hist, 14).dropna().iloc[-1])
    atr5 = float(tech["_ind5"]["atr"])
    has_bars = tb_today is not None and len(tb_today) > 0
    if has_bars:
        exp_open = float(tb_today["open"].iloc[0])
        ref = float(tb_today["close"].iloc[-1])
    else:
        exp_open = float(gm["estimated_open"]) if gm.get("ok") else prev_close
        ref = exp_open

    # --- bias -------------------------------------------------------------
    tech_score = float(tech["score"])
    legs = []
    if dm.get("ok"):
        implied = prev_close * float(np.exp(dm["pred_today_log_ret"]))
        z = float(np.clip(np.log(implied / ref) / max(dm["resid_std"], 1e-6), -3, 3))
        legs.append(("daily factor model", float(np.tanh(0.5 * z)), 0.5))
    if gm.get("ok") and not has_bars:
        legs.append(("overnight gap model", float(np.tanh(gm["estimated_gap_pct"] / 0.5)), 0.5))
    if has_bars and im.get("ok") and im.get("signals"):
        zs = [s["z"] for s in im["signals"].values()]
        legs.append(("intraday predictive model", float(np.tanh(1.5 * np.mean(zs))), 0.6))
    reg = sum(v * w for _, v, w in legs) / sum(w for _, _, w in legs) if legs else 0.0
    f = 0.0 if fund is None else float(fund)
    score = float(np.clip(0.5 * tech_score + 0.45 * reg + 0.05 * f, -1, 1))
    agree = 1 - abs(np.sign(tech_score) - np.sign(reg)) / 2 if tech_score and reg else 0.5
    conf = float(np.clip(0.35 + 0.4 * abs(score) + 0.25 * agree, 0, 1))
    bias = _label(score)

    # --- expected range ---------------------------------------------------
    vr = float(np.clip(vix_ratio, 0.7, 1.6))
    half = 0.55 * atr_d * vr
    center = exp_open + score * 0.35 * atr_d
    lo, hi = center - half, center + half
    if has_bars:  # never predict a range the day has already left
        lo = min(lo, float(tb_today["low"].min()) - 0.1 * atr5)
        hi = max(hi, float(tb_today["high"].max()) + 0.1 * atr5)

    # --- levels around the expected open ----------------------------------
    allz = sorted(zones.get("resistance", []) + zones.get("support", []) + zones.get("at_price", []),
                  key=lambda z: z["center"])
    res = [z for z in allz if z["center"] > ref + 0.25 * atr5]
    sup = [z for z in allz if z["center"] < ref - 0.25 * atr5][::-1]
    r1 = res[0]["center"] if res else ref + 1.5 * atr5
    r2 = res[1]["center"] if len(res) > 1 else r1 + max(2 * atr5, (r1 - ref))
    s1 = sup[0]["center"] if sup else ref - 1.5 * atr5
    s2 = sup[1]["center"] if len(sup) > 1 else s1 - max(2 * atr5, (ref - s1))
    buy = {"side": "buy", "trigger": round(r1 + 0.15 * atr5, 1), "stop": round(s1 - 0.25 * atr5, 1),
           "targets": [round(r2, 1), round(r2 + (r2 - r1), 1)],
           "text": f"Buy above {r1 + 0.15 * atr5:,.0f}; stop {s1 - 0.25 * atr5:,.0f}; targets {r2:,.0f} then {r2 + (r2 - r1):,.0f}"}
    sell = {"side": "sell", "trigger": round(s1 - 0.15 * atr5, 1), "stop": round(r1 + 0.25 * atr5, 1),
            "targets": [round(s2, 1), round(s2 - (s1 - s2), 1)],
            "text": f"Sell below {s1 - 0.15 * atr5:,.0f}; stop {r1 + 0.25 * atr5:,.0f}; targets {s2:,.0f} then {s2 - (s1 - s2):,.0f}"}
    range_buy = {"side": "buy", "trigger": round(lo + 0.15 * atr_d, 1), "stop": round(lo - 0.35 * atr5 * 2, 1),
                 "targets": [round(center, 1), round(hi - 0.15 * atr_d, 1)], "touch": True,
                 "text": f"Buy near the range low {lo + 0.15 * atr_d:,.0f}; stop {lo - 0.7 * atr5:,.0f}; targets {center:,.0f} then {hi - 0.15 * atr_d:,.0f}"}
    range_sell = {"side": "sell", "trigger": round(hi - 0.15 * atr_d, 1), "stop": round(hi + 0.35 * atr5 * 2, 1),
                  "targets": [round(center, 1), round(lo + 0.15 * atr_d, 1)], "touch": True,
                  "text": f"Sell near the range high {hi - 0.15 * atr_d:,.0f}; stop {hi + 0.7 * atr5:,.0f}; targets {center:,.0f} then {lo + 0.15 * atr_d:,.0f}"}
    if score > 0.15:
        primary, alternate, stance = buy, sell, "BUY"
        advice = (f"Lean long. Buy on a sustained move above {buy['trigger']:,.0f} (or on a dip that holds {s1:,.0f}), "
                  f"stop {buy['stop']:,.0f}, book at {buy['targets'][0]:,.0f} and {buy['targets'][1]:,.0f}. "
                  f"Flip to the sell plan only if {sell['trigger']:,.0f} breaks.")
    elif score < -0.15:
        primary, alternate, stance = sell, buy, "SELL"
        advice = (f"Lean short. Sell on a sustained break below {sell['trigger']:,.0f} (or on a bounce that fails at {r1:,.0f}), "
                  f"stop {sell['stop']:,.0f}, book at {sell['targets'][0]:,.0f} and {sell['targets'][1]:,.0f}. "
                  f"Flip to the buy plan only if {buy['trigger']:,.0f} breaks.")
    else:
        primary, alternate, stance = range_sell if score <= 0 else range_buy, range_buy if score <= 0 else range_sell, "RANGE"
        advice = (f"No directional edge: trade the range. Sell near {hi - 0.15 * atr_d:,.0f}, buy near {lo + 0.15 * atr_d:,.0f}, "
                  f"stops just outside the range, target the middle {center:,.0f}. Stand aside on a clean break of either edge.")
    return {
        "date": date, "locked_at": int(now.timestamp()), "locked_at_ist": now.strftime("%H:%M"),
        "locked_with_bars": int(len(tb_today)) if has_bars else 0,
        "prev_close": round(prev_close, 2), "expected_open": round(exp_open, 1),
        "range_low": round(lo, 1), "range_high": round(hi, 1), "range_center": round(center, 1),
        "atr_daily": round(atr_d, 1), "atr_5m": round(atr5, 1), "vix_ratio": round(vr, 2),
        "score": round(score, 3), "bias": bias, "stance": stance, "confidence": round(conf, 2),
        "inputs": {"technical": round(tech_score, 3), "regression": round(reg, 3), "fundamental": round(f, 3),
                   "regression_legs": [{"name": n, "score": round(v, 3), "weight": w} for n, v, w in legs]},
        "levels": {"r1": round(r1, 1), "r2": round(r2, 1), "s1": round(s1, 1), "s2": round(s2, 1),
                   "r1_name": res[0]["members"][0] if res else "ATR projection", "s1_name": sup[0]["members"][0] if sup else "ATR projection"},
        "primary": primary, "alternate": alternate, "advice": advice,
        "evaluated": False,
    }


def _plan_outcome(plan: dict, bars: pd.DataFrame) -> dict:
    """Walk the session bars: trigger, then first target or stop."""
    o = float(bars["open"].iloc[0])
    side, trig, stop, tgt = plan["side"], plan["trigger"], plan["stop"], plan["targets"][0]
    entry, entry_t = None, None
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
            # same bar can already resolve
        hit_stop = (l <= stop) if side == "buy" else (h >= stop)
        hit_tgt = (h >= tgt) if side == "buy" else (l <= tgt)
        if hit_stop:  # conservative: stop wins ties
            return {"triggered": True, "entry": round(entry, 1), "entry_t": entry_t, "result": "stop",
                    "exit": stop, "points": round((stop - entry) if side == "buy" else (entry - stop), 1), "exit_t": int(t.timestamp())}
        if hit_tgt:
            return {"triggered": True, "entry": round(entry, 1), "entry_t": entry_t, "result": "target",
                    "exit": tgt, "points": round((tgt - entry) if side == "buy" else (entry - tgt), 1), "exit_t": int(t.timestamp())}
    if entry is None:
        return {"triggered": False, "result": "no trade", "points": 0.0}
    c = float(bars["close"].iloc[-1])
    return {"triggered": True, "entry": round(entry, 1), "entry_t": entry_t, "result": "open at close", "exit": round(c, 1),
            "points": round((c - entry) if side == "buy" else (entry - c), 1), "exit_t": int(bars.index[-1].timestamp())}


def evaluate_call(call: dict, bars: pd.DataFrame) -> dict:
    o, h, l, c = float(bars["open"].iloc[0]), float(bars["high"].max()), float(bars["low"].min()), float(bars["close"].iloc[-1])
    lo, hi = call["range_low"], call["range_high"]
    direction = 1 if c > o else -1 if c < o else 0
    pred_dir = 1 if call["score"] > 0.15 else -1 if call["score"] < -0.15 else 0
    overlap = max(0.0, min(h, hi) - max(l, lo))
    actual = {"open": round(o, 2), "high": round(h, 2), "low": round(l, 2), "close": round(c, 2),
              "change_pct": round((c / call["prev_close"] - 1) * 100, 2), "session_pct": round((c / o - 1) * 100, 2),
              "range_points": round(h - l, 1)}
    res = {
        "direction_actual": "up" if direction > 0 else "down" if direction < 0 else "flat",
        "direction_hit": (direction == pred_dir) if pred_dir != 0 else None,
        "close_vs_prev_hit": (np.sign(c - call["prev_close"]) == pred_dir) if pred_dir != 0 else None,
        "range_contained": bool(l >= lo and h <= hi),
        "range_coverage": round(overlap / (h - l), 3) if h > l else None,
        "close_in_range": bool(lo <= c <= hi),
        "high_vs_range_high": round(h - hi, 1), "low_vs_range_low": round(l - lo, 1),
        "primary": _plan_outcome(call["primary"], bars),
        "alternate": _plan_outcome(call["alternate"], bars),
    }
    out = dict(call)
    out.update({"evaluated": True, "actual": actual, "result": res, "evaluated_at": int(bars.index[-1].timestamp())})
    return out


def summarise(calls: list[dict]) -> dict:
    ev = [c for c in calls if c.get("evaluated") and not c.get("void")]
    if not ev:
        return {"days": 0}
    d = [c for c in ev if c["result"]["direction_hit"] is not None]
    tr = [c for c in ev if c["result"]["primary"]["triggered"]]
    wins = [c for c in tr if c["result"]["primary"]["result"] == "target"]
    losses = [c for c in tr if c["result"]["primary"]["result"] == "stop"]
    pts = [c["result"]["primary"]["points"] for c in tr]
    last20 = ev[-20:]
    return {
        "days": len(ev), "window": [ev[0]["date"], ev[-1]["date"]],
        "direction_calls": len(d), "direction_hit_rate": round(sum(1 for c in d if c["result"]["direction_hit"]) / len(d), 3) if d else None,
        "range_contained_rate": round(sum(1 for c in ev if c["result"]["range_contained"]) / len(ev), 3),
        "close_in_range_rate": round(sum(1 for c in ev if c["result"]["close_in_range"]) / len(ev), 3),
        "avg_range_coverage": round(float(np.mean([c["result"]["range_coverage"] for c in ev if c["result"]["range_coverage"] is not None])), 3),
        "trades": len(tr), "wins": len(wins), "losses": len(losses), "open_at_close": len(tr) - len(wins) - len(losses),
        "no_trade_days": len(ev) - len(tr),
        "win_rate": round(len(wins) / (len(wins) + len(losses)), 3) if (wins or losses) else None,
        "net_points": round(float(sum(pts)), 1), "avg_points_per_trade": round(float(np.mean(pts)), 1) if pts else None,
        "last20_net_points": round(float(sum(c["result"]["primary"]["points"] for c in last20 if c["result"]["primary"]["triggered"])), 1),
        "last20_direction_hit_rate": round(sum(1 for c in last20 if c["result"]["direction_hit"]) / max(1, sum(1 for c in last20 if c["result"]["direction_hit"] is not None)), 3),
    }


def update_log(log: dict, now: pd.Timestamp, s5: pd.DataFrame, market_status: str, make_call) -> dict:
    """Lock today's call if missing, evaluate finished sessions, prune."""
    calls = list(log.get("calls", []))
    by_date = {c["date"]: c for c in calls}
    today = now.strftime("%Y-%m-%d")
    days = s5.index.normalize()
    # evaluate every unevaluated call whose session is over
    for c in calls:
        if c.get("evaluated") or c.get("void"):
            continue
        d = pd.Timestamp(c["date"]).tz_localize(s5.index.tz)
        bars = s5[days == d]
        session_over = (c["date"] < today) or (market_status in ("closed", "closed-weekend") and now.time() >= dt.time(15, 30))
        if not session_over:
            continue
        if len(bars) >= 60:
            c.update(evaluate_call(c, bars))
        elif c["date"] < today or now.time() >= dt.time(16, 0):
            c["void"] = True
            c["void_reason"] = "no session data (holiday or feed gap)"
    # lock today's call once, on weekdays, from the pre-open run onwards
    if today not in by_date and now.weekday() < 5 and market_status in ("pre-market", "pre-open", "open"):
        calls.append(make_call())
    calls = sorted(calls, key=lambda c: c["date"])[-KEEP_DAYS:]
    return {"calls": calls, "summary": summarise(calls), "updated_at": int(now.timestamp())}
