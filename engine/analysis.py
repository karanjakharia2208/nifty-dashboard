"""Levels, scoring, forecast cone and the walk-forward backtest."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import indicators as ti
import patterns as pt
from data import session_filter
from regression import daily_model, intraday_model, HORIZONS

W_TECH, W_REG, W_FUND = 0.50, 0.45, 0.05


# ----------------------------------------------------------------- levels ---
def pivots(h, l, c):
    p = (h + l + c) / 3
    out = {"P": p, "R1": 2 * p - l, "S1": 2 * p - h, "R2": p + (h - l), "S2": p - (h - l),
           "R3": h + 2 * (p - l), "S3": l - 2 * (h - p)}
    # Central pivot range
    bc = (h + l) / 2
    tc = 2 * p - bc
    out["TC"], out["BC"] = max(tc, bc), min(tc, bc)
    # Camarilla
    r = h - l
    out["CamR3"], out["CamR4"] = c + r * 1.1 / 4, c + r * 1.1 / 2
    out["CamS3"], out["CamS4"] = c - r * 1.1 / 4, c - r * 1.1 / 2
    return out


def build_levels(n5: pd.DataFrame, daily: pd.DataFrame, ind5: dict, indd: dict, oc: dict | None) -> list[dict]:
    px = float(n5["close"].iloc[-1])
    days = sorted(set(n5.index.normalize()))
    today = days[-1]
    today_bars = n5[n5.index.normalize() == today]
    prev = n5[n5.index.normalize() == days[-2]] if len(days) > 1 else today_bars
    lv = []

    def add(name, value, kind, source, strength=1.0):
        if value is None or not np.isfinite(value):
            return
        lv.append({"name": name, "value": round(float(value), 1), "kind": kind, "source": source,
                   "strength": round(float(strength), 2)})

    # previous day
    ph, pl, pc = float(prev["high"].max()), float(prev["low"].min()), float(prev["close"].iloc[-1])
    add("Prev day high", ph, "resistance", "structure", 1.2)
    add("Prev day low", pl, "support", "structure", 1.2)
    add("Prev close", pc, "pivot", "structure", 0.8)
    pv = pivots(ph, pl, pc)
    for k in ("P", "R1", "R2", "R3", "S1", "S2", "S3"):
        add(f"Pivot {k}", pv[k], "resistance" if k.startswith("R") else "support" if k.startswith("S") else "pivot",
            "classic pivots", 1.0 if k in ("P", "R1", "S1") else 0.7)
    add("CPR top", pv["TC"], "pivot", "CPR", 0.7)
    add("CPR bottom", pv["BC"], "pivot", "CPR", 0.7)
    for k in ("CamR3", "CamR4", "CamS3", "CamS4"):
        add(f"Camarilla {k[3:]}", pv[k], "resistance" if "R" in k else "support", "Camarilla", 0.6)
    # today's structure
    if len(today_bars):
        add("Day high", float(today_bars["high"].max()), "resistance", "structure", 1.0)
        add("Day low", float(today_bars["low"].min()), "support", "structure", 1.0)
        add("Day open", float(today_bars["open"].iloc[0]), "pivot", "structure", 0.6)
        orb = today_bars.iloc[:3]  # first 15 minutes
        if len(orb) >= 3:
            add("Opening range high", float(orb["high"].max()), "resistance", "ORB 15m", 0.9)
            add("Opening range low", float(orb["low"].min()), "support", "ORB 15m", 0.9)
    # indicator-derived
    if ind5.get("vwap"):
        add("VWAP", ind5["vwap"], "pivot", "VWAP", 1.1)
    for k, nm in (("bb_upper", "Bollinger upper (5m)"), ("bb_lower", "Bollinger lower (5m)")):
        if ind5.get(k):
            add(nm, ind5[k], "resistance" if "upper" in k else "support", "Bollinger", 0.6)
    for k in ("ema21", "ema50", "sma200"):
        if ind5.get(k):
            add(f"{k.upper()} (5m)", ind5[k], "resistance" if ind5[k] > px else "support", "moving average", 0.6)
    for k in ("sma20", "sma50", "sma200"):
        if indd.get(k):
            add(f"{k.upper()} (daily)", indd[k], "resistance" if indd[k] > px else "support", "moving average", 0.9)
    if ind5.get("supertrend"):
        add("Supertrend (5m)", ind5["supertrend"], "support" if ind5["supertrend_dir"] > 0 else "resistance", "Supertrend", 0.7)
    if ind5.get("psar"):
        add("Parabolic SAR (5m)", ind5["psar"], "support" if ind5["psar_trend"] > 0 else "resistance", "PSAR", 0.4)
    for k, nm in (("ichi_span_a", "Ichimoku span A (daily)"), ("ichi_span_b", "Ichimoku span B (daily)")):
        if indd.get(k):
            add(nm, indd[k], "resistance" if indd[k] > px else "support", "Ichimoku", 0.6)
    # fibonacci (daily swing, then intraday swing)
    for src, ind in (("daily swing", indd), ("intraday swing", ind5)):
        fib = ind.get("fib")
        if fib:
            for r in ("0.236", "0.382", "0.500", "0.618", "0.786"):
                v = fib.get(r)
                if v:
                    add(f"Fib {float(r) * 100:.1f}% ({src})", v, "resistance" if v > px else "support", "Fibonacci", 0.8 if src == "daily swing" else 0.6)
    # recent swing highs/lows
    for i, p, k in (indd.get("swings") or [])[-4:]:
        add(f"Swing {'high' if k.startswith('H') else 'low'} (daily)", p, "resistance" if p > px else "support", "swings", 0.8)
    for i, p, k in (ind5.get("swings") or [])[-4:]:
        add(f"Swing {'high' if k.startswith('H') else 'low'} (5m)", p, "resistance" if p > px else "support", "swings", 0.6)
    # option chain walls
    if oc and oc.get("expiries"):
        e = oc["expiries"][0]
        if e.get("call_wall"):
            add(f"Call OI wall {e['expiry']}", e["call_wall"], "resistance", "option chain", 1.1)
        if e.get("put_wall"):
            add(f"Put OI wall {e['expiry']}", e["put_wall"], "support", "option chain", 1.1)
        if e.get("max_pain"):
            add(f"Max pain {e['expiry']}", e["max_pain"], "pivot", "option chain", 0.8)
    # round numbers
    base = round(px / 100) * 100
    for k in (-200, -100, 0, 100, 200):
        v = base + k
        add(f"Round {v:.0f}", v, "resistance" if v > px else "support" if v < px else "pivot", "round number", 0.4)
    lv.sort(key=lambda x: x["value"])
    # cluster levels within 0.1% into zones for the summary
    return lv


def cluster_zones(levels: list[dict], px: float, tol_pct=0.12, atr5: float | None = None) -> dict:
    """Group nearby levels into zones; return the strongest few above and below.
    Levels more than 3% away are ignored and anything within ~0.35 ATR of
    spot is 'at price', not a usable support/resistance."""
    near = atr5 * 0.35 if atr5 else px * 0.0006
    zones = []
    for l in levels:
        if abs(l["value"] - px) / px > 0.03:
            continue
        if zones and abs(l["value"] - zones[-1]["center"]) / px * 100 <= tol_pct:
            z = zones[-1]
            z["members"].append(l["name"])
            z["strength"] += l["strength"]
            z["center"] = (z["center"] * (len(z["members"]) - 1) + l["value"]) / len(z["members"])
        else:
            zones.append({"center": l["value"], "members": [l["name"]], "strength": l["strength"]})
    for z in zones:
        z["center"] = round(z["center"], 1)
        z["strength"] = round(z["strength"], 2)
        z["kind"] = "resistance" if z["center"] > px else "support"
    at_price = [z for z in zones if abs(z["center"] - px) <= near]
    for z in at_price:
        z["kind"] = "at price"
    above = sorted([z for z in zones if z["center"] > px + near], key=lambda z: z["center"])[:5]
    below = sorted([z for z in zones if z["center"] < px - near], key=lambda z: -z["center"])[:5]
    return {"resistance": above, "support": below, "at_price": at_price}


# ---------------------------------------------------------------- scoring ---
def technical_score(n5: pd.DataFrame, nd: pd.DataFrame, oc: dict | None, manual_oc: dict | None) -> dict:
    s5 = session_filter(n5)
    ind5 = ti.compute_all(s5, "intraday")
    indd = ti.compute_all(nd, "daily")
    v5 = ti.votes(ind5, "intraday")
    vd = ti.votes(indd, "daily")
    p5 = pt.detect(s5, "intraday")
    pd_ = pt.detect(nd, "daily")
    pv5, pn5 = pt.pattern_vote(p5)
    pvd, pnd = pt.pattern_vote(pd_)
    a5, ad = ti.aggregate(v5), ti.aggregate(vd)
    # positioning from option chain: high PCR (>1.3) = puts heavy = supportive; low (<0.7) = calls heavy
    pos_vote, pos_note = None, "no option-chain data"
    src = None
    if oc and oc.get("expiries"):
        e = oc["expiries"][0]
        pcr = e.get("pcr_oi")
        src = oc.get("source")
    elif manual_oc and manual_oc.get("pcr"):
        pcr = float(manual_oc["pcr"])
        src = "manual"
    else:
        pcr = None
    if pcr:
        pos_vote = float(np.clip((pcr - 1.0) * 1.2, -1, 1))
        pos_note = f"PCR {pcr:.2f} ({src}) → {'put-heavy, supportive' if pcr > 1.1 else 'call-heavy, capped' if pcr < 0.9 else 'balanced'}"
    comps = [("intraday indicators", a5, 0.45), ("daily indicators", ad, 0.25),
             ("intraday patterns", pv5, 0.12), ("daily patterns", pvd, 0.08)]
    if pos_vote is not None:
        comps.append(("positioning (PCR/OI)", pos_vote, 0.10))
    wsum = sum(w for _, _, w in comps)
    score = sum(v * w for _, v, w in comps) / wsum
    return {
        "score": float(np.clip(score, -1, 1)),
        "components": [{"name": n, "score": round(float(v), 3), "weight": round(w / wsum, 3)} for n, v, w in comps],
        "intraday": {"indicators": _json_ind(ind5), "votes": v5, "score": a5, "patterns": p5, "pattern_note": pn5},
        "daily": {"indicators": _json_ind(indd), "votes": vd, "score": ad, "patterns": pd_, "pattern_note": pnd},
        "positioning_note": pos_note,
        "_ind5": ind5, "_indd": indd,
    }


def _json_ind(ind: dict) -> dict:
    out = {}
    for k, v in ind.items():
        if k in ("swings",):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {kk: (None if vv is None or not np.isfinite(vv) else round(float(vv), 2)) for kk, vv in v.items()}
        elif isinstance(v, (bool, str)):
            out[k] = v
        elif v is None or (isinstance(v, float) and not np.isfinite(v)):
            out[k] = None
        else:
            out[k] = round(float(v), 4)
    return out


def regression_score(dm: dict, im: dict, spot: float, prev_close: float) -> dict:
    parts = []
    # daily: spot vs factor-implied close
    daily_sig = None
    if dm.get("ok"):
        implied = prev_close * float(np.exp(dm["pred_today_log_ret"]))
        gap = np.log(implied / spot)
        z = float(np.clip(gap / max(dm["resid_std"], 1e-6), -3, 3))
        # softened: a big unexplained move is a lean, not a certainty
        daily_sig = float(np.tanh(0.5 * z))
        dm["z_spot_vs_implied"] = z
        dm["implied_close"] = implied
        dm["spot_vs_implied_pct"] = -gap * 100
        parts.append(("daily factor model", daily_sig, 0.4))
    intr = None
    if im.get("ok") and im.get("signals"):
        zs = [s["z"] for s in im["signals"].values()]
        intr = float(np.tanh(1.5 * np.mean(zs)))
        parts.append(("intraday predictive model", intr, 0.6))
    if not parts:
        return {"score": 0.0, "components": [], "note": "regression unavailable"}
    w = sum(p[2] for p in parts)
    score = sum(p[1] * p[2] for p in parts) / w
    return {"score": float(np.clip(score, -1, 1)),
            "components": [{"name": n, "score": round(v, 3), "weight": round(ww / w, 3)} for n, v, ww in parts]}


def combine(tech: float, reg: float, fund: float | None) -> dict:
    f = 0.0 if fund is None else float(fund)
    total = W_TECH * tech + W_REG * reg + W_FUND * f
    # confidence: agreement between the two main legs plus magnitude
    agree = 1 - abs(np.sign(tech) - np.sign(reg)) / 2 if tech and reg else 0.5
    conf = float(np.clip(0.35 + 0.4 * abs(total) + 0.25 * agree, 0, 1))
    bias = "Bullish" if total > 0.15 else "Bearish" if total < -0.15 else "Neutral / range"
    if abs(total) > 0.45:
        bias = "Strongly " + bias.lower()
    return {"score": float(total), "bias": bias, "confidence": conf,
            "weights": {"technical": W_TECH, "regression": W_REG, "fundamental": W_FUND},
            "inputs": {"technical": tech, "regression": reg, "fundamental": f, "fundamental_source": "db" if fund is not None else "none yet"}}


def forecast_cone(n5: pd.DataFrame, score: float, sigma_5m: float, vix_ratio: float, bars: int = 24) -> dict:
    s5 = session_filter(n5)
    spot = float(s5["close"].iloc[-1])
    last_t = s5.index[-1]
    a = float(ti.atr(s5, 14).dropna().iloc[-1])
    sig = max(sigma_5m, 1e-5) * float(np.clip(vix_ratio, 0.6, 1.8))
    # expected drift over `bars`: score * 0.9 * ATR * sqrt(bars/ (1/ (5min))) -> use ATR-scaled move
    drift_total = score * 0.9 * a * np.sqrt(bars) / np.sqrt(3)
    pts = []
    t = last_t
    for i in range(1, bars + 1):
        t = t + pd.Timedelta(minutes=5)
        if t.time() >= dt.time(15, 30):
            # roll to next session open
            nxt = (t.normalize() + pd.Timedelta(days=1)).replace(hour=9, minute=15)
            while nxt.weekday() >= 5:
                nxt += pd.Timedelta(days=1)
            t = nxt
        center = spot + drift_total * i / bars
        band = spot * sig * np.sqrt(i)
        pts.append({"t": int(t.timestamp()), "center": round(center, 1),
                    "lo68": round(center - band, 1), "hi68": round(center + band, 1),
                    "lo90": round(center - 1.645 * band, 1), "hi90": round(center + 1.645 * band, 1)})
    return {"spot": spot, "from": int(last_t.timestamp()), "points": pts, "atr_5m": a, "sigma_5m_pct": sig * 100,
            "expected_move_points": {h: round(drift_total * nb / bars, 1) for h, nb in HORIZONS.items()},
            "range_68_points": {h: round(spot * sig * np.sqrt(nb), 1) for h, nb in HORIZONS.items()}}


def trade_plan(spot: float, zones: dict, combined: dict, cone: dict, a5: float) -> dict:
    res = zones["resistance"][0]["center"] if zones["resistance"] else spot + 2 * a5
    sup = zones["support"][0]["center"] if zones["support"] else spot - 2 * a5
    res2 = zones["resistance"][1]["center"] if len(zones["resistance"]) > 1 else res + 2 * a5
    sup2 = zones["support"][1]["center"] if len(zones["support"]) > 1 else sup - 2 * a5
    s = combined["score"]
    plan = {"bias": combined["bias"], "nearest_resistance": res, "nearest_support": sup,
            "next_resistance": res2, "next_support": sup2}
    if s > 0.15:
        plan["long_trigger"] = round(res + 0.15 * a5, 1)
        plan["long_invalidation"] = round(sup - 0.25 * a5, 1)
        plan["long_targets"] = [round(res2, 1), round(res2 + (res2 - res), 1)]
        plan["note"] = ("Bullish lean: odds favour longs on a sustained move above the nearest resistance, "
                        "with the nearest support as the line in the sand.")
    elif s < -0.15:
        plan["short_trigger"] = round(sup - 0.15 * a5, 1)
        plan["short_invalidation"] = round(res + 0.25 * a5, 1)
        plan["short_targets"] = [round(sup2, 1), round(sup2 - (sup - sup2), 1)]
        plan["note"] = ("Bearish lean: odds favour shorts on a sustained break below the nearest support, "
                        "with the nearest resistance as the invalidation.")
    else:
        plan["range_high"], plan["range_low"] = round(res, 1), round(sup, 1)
        plan["note"] = "No edge either way: treat it as a range between the nearest support and resistance."
    plan["expected_move_1h"] = cone["expected_move_points"]["1h"]
    plan["expected_move_2h"] = cone["expected_move_points"]["2h"]
    plan["typical_range_1h"] = cone["range_68_points"]["1h"]
    return plan


# --------------------------------------------------------------- backtest ---
def backtest(n5: pd.DataFrame, nd: pd.DataFrame, daily: dict, intraday: dict, sessions: int = 25) -> dict:
    """Walk-forward: at fixed times on each of the last `sessions` days, compute
    the technical + regression score using only data up to that time and
    compare with the realised 1h / 2h move. Fundamental leg is excluded
    (no news history), so the test covers 95% of the final score."""
    s5 = session_filter(n5)
    days = sorted(set(s5.index.normalize()))
    test_days = days[-(sessions + 1):-1]  # completed sessions only
    times = [dt.time(9, 45), dt.time(10, 30), dt.time(11, 30), dt.time(12, 30), dt.time(13, 30), dt.time(14, 15)]
    recs = []
    dm_cache = {}
    for d in test_days:
        sess = s5[s5.index.normalize() == d]
        if len(sess) < 40:
            continue
        # daily model fitted with data before this session
        try:
            dm = daily_model(daily, asof=d - pd.Timedelta(days=1), target_date=d)
        except Exception:
            dm = {"ok": False}
        prev_close = float(s5[s5.index.normalize() < d]["close"].iloc[-1]) if len(s5[s5.index.normalize() < d]) else float(sess["open"].iloc[0])
        for tm in times:
            cut = sess[sess.index.time <= tm]
            if len(cut) < 6:
                continue
            t_now = cut.index[-1]
            hist5 = s5[s5.index <= t_now]
            histd = nd[nd.index.normalize() < d]
            if len(histd) < 220:
                continue
            spot = float(cut["close"].iloc[-1])
            try:
                ts = technical_score(hist5, histd, None, None)
                im = intraday_model(hist5, {k: v[v.index <= t_now] for k, v in intraday.items()}, asof=t_now)
                rs = regression_score(dict(dm), im, spot, prev_close)
                comb = combine(ts["score"], rs["score"], None)
            except Exception as e:  # keep going; record failure count
                recs.append({"date": str(d.date()), "time": tm.strftime("%H:%M"), "error": type(e).__name__})
                continue
            rec = {"date": str(d.date()), "time": tm.strftime("%H:%M"), "spot": spot, "score": round(comb["score"], 3),
                   "tech": round(ts["score"], 3), "reg": round(rs["score"], 3)}
            for h, nb in HORIZONS.items():
                fut = sess[sess.index > t_now]
                if not len(fut):
                    continue
                target = fut["close"].iloc[min(nb - 1, len(fut) - 1)]
                rec[f"ret_{h}"] = round((float(target) / spot - 1) * 100, 3)
            recs.append(rec)
    good = [r for r in recs if "error" not in r]
    summary = {"n": len(good), "errors": len(recs) - len(good), "sessions": len(test_days),
               "window": [str(test_days[0].date()), str(test_days[-1].date())] if test_days else None}
    for h in HORIZONS:
        rr = [r for r in good if f"ret_{h}" in r and abs(r["score"]) > 0.15]
        if not rr:
            continue
        hits = sum(1 for r in rr if np.sign(r[f"ret_{h}"]) == np.sign(r["score"]))
        strong = [r for r in rr if abs(r["score"]) > 0.4]
        s_hits = sum(1 for r in strong if np.sign(r[f"ret_{h}"]) == np.sign(r["score"]))
        avg_signed = float(np.mean([np.sign(r["score"]) * r[f"ret_{h}"] for r in rr]))
        summary[h] = {"signals": len(rr), "hit_rate": round(hits / len(rr), 3),
                      "strong_signals": len(strong), "strong_hit_rate": round(s_hits / len(strong), 3) if strong else None,
                      "avg_signed_return_pct": round(avg_signed, 3),
                      "avg_abs_move_pct": round(float(np.mean([abs(r[f"ret_{h}"]) for r in rr])), 3)}
        # by component
        for leg in ("tech", "reg"):
            lr = [r for r in good if f"ret_{h}" in r and abs(r[leg]) > 0.15]
            if lr:
                summary[h][f"{leg}_hit_rate"] = round(sum(1 for r in lr if np.sign(r[f"ret_{h}"]) == np.sign(r[leg])) / len(lr), 3)
    return {"summary": summary, "records": good[-300:], "generated_at": int(pd.Timestamp.now(tz="Asia/Kolkata").timestamp())}
