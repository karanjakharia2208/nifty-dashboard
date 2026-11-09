"""Long-horizon walk-forward backtest of the morning call on DAILY bars.

The live call uses 5-minute bars, the option chain and a news scan; none of
those exist as free history, so this is the same recipe rebuilt from what
does exist since Nifty's data begins (Yahoo: Sept 2007):

  * technical leg  — the same daily indicator votes + daily pattern vote
  * regression leg — the daily factor model (lagged US session, dollar,
                      Brent, gold, yields, VIX, India ETF…) and a gap model,
                      refitted on a trailing 250 sessions every 20 sessions,
                      using only data available before that day's open
  * score          — 0.5 * technical + 0.45 * regression (+ 0 news)
  * range          — daily ATR(14) × India-VIX ratio, shifted by the score
  * levels         — classic pivots and previous-day high/low around the
                      expected open; 5-minute ATR approximated as ATR/11
  * trade          — primary plan evaluated on the day's O/H/L/C.

Daily bars cannot say whether the stop or the target was touched first, nor
whether the day's low came before the entry, so trade results carry three
readings: headline (the close decides unknown orderings), conservative (any
stop touch counts) and optimistic (a stop counts only if the day closes
through it). The same geometry is also run with the day's scores shuffled
("random baseline"), so the edge of the signal is the difference between the
real result and that baseline, not the raw number.

Output: a dict written under the key "twenty_year" in backtest.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D  # noqa: E402
import indicators as ti  # noqa: E402
import patterns as pt  # noqa: E402
from regression import _daily_series, _lagged_us_return, ols, predict  # noqa: E402

LAG_SPEC = {  # factors known before the Indian open (previous session) — the live model minus same-day Asia
    "SPX": "ret", "NDX": "ret", "INDA": "ret", "EEM": "ret", "VIX": "diff", "US10Y": "diff", "DXY": "ret",
    "BRENT": "ret", "GOLD": "ret", "COPPER": "ret", "STOXX50": "ret", "INDIAVIX": "diff",
}
REFIT_EVERY = 20
LOOKBACK = 250
MIN_HISTORY = 260


def _factor_matrix(daily: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    cols = {}
    for nm, kind in LAG_SPEC.items():
        if nm not in daily:
            continue
        cl = daily[nm]["close"].astype(float)
        if kind == "diff":
            cl = np.exp(cl / 10.0)
        cols[nm] = _lagged_us_return(cl, dates).values
    return pd.DataFrame(cols, index=dates)


def _label(score):
    bias = "Bullish" if score > 0.15 else "Bearish" if score < -0.15 else "Neutral"
    return ("Strongly " + bias.lower()) if abs(score) > 0.45 else bias


def signals(daily: dict) -> list[dict]:
    """First pass: everything known before each day's open."""
    nd = daily["NIFTY"].copy()
    nd = nd[~nd.index.duplicated(keep="last")].sort_index()
    y = _daily_series(nd["close"])
    dates = y.index
    X = _factor_matrix(daily, dates)
    gap_feats = [c for c in ("SPX", "INDA") if c in X.columns]
    opens = nd["open"].astype(float).values
    closes = nd["close"].astype(float).values
    highs = nd["high"].astype(float).values
    lows = nd["low"].astype(float).values
    nd_dates = pd.DatetimeIndex(nd.index.tz_convert(D.IST).tz_localize(None).normalize())
    pos_of = {d: i for i, d in enumerate(nd_dates)}
    gap_series = pd.Series(np.log(opens[1:] / closes[:-1]), index=nd_dates[1:])
    atr = ti.atr(nd, 14)
    atr.index = nd_dates
    vix = None
    if "INDIAVIX" in daily:
        v = daily["INDIAVIX"]["close"].astype(float)
        v.index = pd.DatetimeIndex(v.index.tz_convert(D.IST).tz_localize(None).normalize())
        vix = (v / v.rolling(21).mean()).reindex(nd_dates).ffill()

    out = []
    model = gapm = None
    cols_used = []
    for k, d in enumerate(dates):
        i = pos_of.get(d)
        if i is None or i < MIN_HISTORY:
            continue
        if model is None or k % REFIT_EVERY == 0:
            win = X.iloc[k - LOOKBACK:k]
            yy = y.iloc[k - LOOKBACK:k].values
            keep = [c for c in win.columns if win[c].notna().mean() > 0.7]
            model = ols(win[keep].fillna(0.0).values, yy, ridge=1e-3, names=keep) if keep else None
            cols_used = keep
            gk = [c for c in gap_feats if c in keep]
            gy = gap_series.reindex(dates[k - LOOKBACK:k]).fillna(0.0).values
            gapm = ols(X.iloc[k - LOOKBACK:k][gk].fillna(0.0).values, gy, ridge=1e-3, names=gk) if gk else None
        prev_close = closes[i - 1]
        hist = nd.iloc[max(0, i - 420): i]  # completed sessions only; enough for SMA200 / Ichimoku
        try:
            ind = ti.compute_all(hist, "daily")
            tech_ind = ti.aggregate(ti.votes(ind, "daily"))
            pv, _ = pt.pattern_vote(pt.detect(hist, "daily"))
        except Exception:
            tech_ind, pv = 0.0, 0.0
        tech = float(np.clip(0.75 * tech_ind + 0.25 * pv, -1, 1))
        legs = []
        if model is not None and cols_used:
            pred = predict(model, X.iloc[k][cols_used].fillna(0.0).values)
            z = float(np.clip(pred / max(model["resid_std"], 1e-6), -3, 3))
            legs.append(float(np.tanh(0.5 * z)))
        gap_est = 0.0
        if gapm is not None:
            gap_est = float(predict(gapm, X.iloc[k][[c for c in gap_feats if c in cols_used]].fillna(0.0).values))
            legs.append(float(np.tanh(gap_est * 100 / 0.5)))
        reg = float(np.mean(legs)) if legs else 0.0
        a = float(atr.iloc[i - 1]) if np.isfinite(atr.iloc[i - 1]) else prev_close * 0.012
        vr = float(np.clip(vix.iloc[i - 1], 0.7, 1.6)) if vix is not None and np.isfinite(vix.iloc[i - 1]) else 1.0
        out.append({"date": str(d.date()), "i": i, "tech": tech, "reg": reg, "gap_est": gap_est, "atr": a, "vr": vr,
                    "prev_close": prev_close, "ph": highs[i - 1], "pl": lows[i - 1], "pc": closes[i - 1],
                    "o": opens[i], "h": highs[i], "l": lows[i], "c": closes[i]})
    return out


def plan_and_score(s: dict, score: float) -> dict:
    """Second pass for one day: build the plan from `score` and score it on O/H/L/C."""
    a, vr, prev_close = s["atr"], s["vr"], s["prev_close"]
    exp_open = prev_close * float(np.exp(s["gap_est"]))
    half = 0.65 * a * vr
    center = exp_open + score * 0.35 * a
    lo, hi = center - half, center + half
    atr5 = a / 11.0
    ph, pl, pc = s["ph"], s["pl"], s["pc"]
    p = (ph + pl + pc) / 3
    base = round(exp_open / 100) * 100
    lv = sorted({ph, pl, 2 * p - pl, 2 * p - ph, p + (ph - pl), p - (ph - pl), base, base + 100, base - 100})
    res = [v for v in lv if v > exp_open + 0.25 * atr5]
    sup = [v for v in lv if v < exp_open - 0.25 * atr5][::-1]
    r1 = res[0] if res else exp_open + 1.5 * atr5
    r2 = res[1] if len(res) > 1 else r1 + max(2 * atr5, r1 - exp_open)
    s1 = sup[0] if sup else exp_open - 1.5 * atr5
    s2 = sup[1] if len(sup) > 1 else s1 - max(2 * atr5, exp_open - s1)
    # risk capped at 0.35 ATR, first target at least 2R away (same as the live plan)
    if score > 0.15:
        side, trig = "buy", r1 + 0.15 * atr5
        risk = min(trig - (s1 - 0.25 * atr5), 0.35 * a)
        stop, tgt = trig - risk, max(r2, trig + 2 * risk)
    elif score < -0.15:
        side, trig = "sell", s1 - 0.15 * atr5
        risk = min((r1 + 0.25 * atr5) - trig, 0.35 * a)
        stop, tgt = trig + risk, min(s2, trig - 2 * risk)
    elif score <= 0:
        side, trig, stop, tgt = "sell", hi - 0.12 * a, hi + 0.12 * a, center
    else:
        side, trig, stop, tgt = "buy", lo + 0.12 * a, lo - 0.12 * a, center
    o, h, l, c = s["o"], s["h"], s["l"], s["c"]
    direction = 1 if c > o else -1 if c < o else 0
    pred_dir = 1 if score > 0.15 else -1 if score < -0.15 else 0
    touch = pred_dir == 0
    risk_d = abs(trig - stop)
    if side == "buy":
        trig_hit = (l <= trig) if touch else (h >= trig)
        entry = trig if touch else max(trig, o)
        stop_hit, tgt_hit = l <= stop, h >= tgt
        if not touch and o > trig + 0.5 * risk_d:
            trig_hit = False  # gap open through the trigger: breakout happened before you could act
    else:
        trig_hit = (h >= trig) if touch else (l <= trig)
        entry = trig if touch else min(trig, o)
        stop_hit, tgt_hit = h >= stop, l <= tgt
        if not touch and o < trig - 0.5 * risk_d:
            trig_hit = False

    def pnl(exit_px):
        return (exit_px - entry) if side == "buy" else (entry - exit_px)

    closed_through_stop = (c <= stop) if side == "buy" else (c >= stop)
    if not trig_hit:
        cons = opt = mid = 0.0
        res_c = res_o = res_m = "no trade"
    else:
        if stop_hit:
            cons, res_c = pnl(stop), "stop"
        elif tgt_hit:
            cons, res_c = pnl(tgt), "target"
        else:
            cons, res_c = pnl(c), "close"
        if tgt_hit:
            opt, res_o = pnl(tgt), "target"
        elif closed_through_stop:
            opt, res_o = pnl(stop), "stop"
        else:
            opt, res_o = pnl(c), "close"
        if stop_hit and tgt_hit:
            mid, res_m = (pnl(tgt), "target") if pnl(c) >= 0 else (pnl(stop), "stop")
        elif stop_hit:
            mid, res_m = (pnl(stop), "stop") if pnl(c) < 0 else (pnl(c), "close")
        elif tgt_hit:
            mid, res_m = pnl(tgt), "target"
        else:
            mid, res_m = pnl(c), "close"
    return {
        "date": s["date"], "score": round(score, 3), "bias": _label(score), "side": side,
        "exp_open": round(exp_open, 1), "lo": round(lo, 1), "hi": round(hi, 1),
        "o": round(o, 1), "h": round(h, 1), "l": round(l, 1), "c": round(c, 1),
        "dir_hit": (direction == pred_dir) if pred_dir else None,
        "range_held": bool(l >= lo and h <= hi), "close_in": bool(lo <= c <= hi),
        "triggered": bool(trig_hit), "res_c": res_c, "res_o": res_o, "res_m": res_m,
        "pts_c": round(float(cons), 1), "pts_o": round(float(opt), 1), "pts_m": round(float(mid), 1),
        "pct_c": round(float(cons) / prev_close * 100, 3), "pct_o": round(float(opt) / prev_close * 100, 3),
        "pct_m": round(float(mid) / prev_close * 100, 3),
    }


def _block(g: pd.DataFrame) -> dict:
    d = g[g["dir_hit"].notna()]
    tr = g[g["triggered"]]

    def wr(col):
        if not len(tr):
            return None
        dec = tr[col].isin(["target", "stop"])
        return round(float((tr.loc[dec, col] == "target").mean()), 3) if dec.any() else None

    return {
        "days": int(len(g)), "directional_days": int(len(d)),
        "direction_hit_rate": round(float(d["dir_hit"].mean()), 3) if len(d) else None,
        "range_held_rate": round(float(g["range_held"].mean()), 3),
        "close_in_range_rate": round(float(g["close_in"].mean()), 3),
        "trades": int(len(tr)),
        "win_rate": wr("res_m"), "net_pct": round(float(tr["pct_m"].sum()), 2),
        "avg_pct_per_trade": round(float(tr["pct_m"].mean()), 3) if len(tr) else None,
        "win_rate_conservative": wr("res_c"), "net_pct_conservative": round(float(tr["pct_c"].sum()), 2),
        "win_rate_optimistic": wr("res_o"), "net_pct_optimistic": round(float(tr["pct_o"].sum()), 2),
        "both_touched_days": int(((tr["res_c"] == "stop") & (tr["res_o"] == "target")).sum()),
    }


def summarise(real: list[dict], base: list[dict], secs: float) -> dict:
    df = pd.DataFrame(real)
    bf = pd.DataFrame(base)
    df["year"] = df["date"].str[:4]
    bf["year"] = bf["date"].str[:4]
    overall, baseline = _block(df), _block(bf)
    by_year = {}
    for yr, g in df.groupby("year"):
        blk = _block(g)
        bb = _block(bf[bf["year"] == yr])
        blk["baseline_net_pct"] = bb["net_pct"]
        blk["baseline_direction_hit_rate"] = bb["direction_hit_rate"]
        by_year[yr] = blk
    df["month"] = df["date"].str[:7]
    bf["month"] = bf["date"].str[:7]
    m = df.groupby("month").agg(pm=("pct_m", "sum"), pc=("pct_c", "sum"), po=("pct_o", "sum"))
    mb = bf.groupby("month")["pct_m"].sum().reindex(m.index).fillna(0.0)
    curve = [{"month": idx, "cum_pct": round(float(u), 2), "cum_pct_conservative": round(float(v), 2),
              "cum_pct_optimistic": round(float(w), 2), "baseline_cum_pct": round(float(b), 2)}
             for idx, u, v, w, b in zip(m.index, m["pm"].cumsum(), m["pc"].cumsum(), m["po"].cumsum(), mb.cumsum())]
    cum = m["pm"].cumsum().values
    dd = float((cum - np.maximum.accumulate(cum)).min()) if len(cum) else 0.0
    return {
        "method": ("daily-bar proxy of the morning call, walked forward; the 5% news leg is 0 and there is no option "
                   "chain; regressions refit every 20 sessions on a trailing 250; trades scored on O/H/L/C. Headline "
                   "figures let the close decide when the order of stop/target touches is unknown; 'conservative' "
                   "counts any stop touch, 'optimistic' counts a stop only when the day closes through it. The random "
                   "baseline runs the identical geometry with the daily scores shuffled."),
        "window": [real[0]["date"], real[-1]["date"]], "overall": overall, "baseline": baseline, "by_year": by_year,
        "monthly_curve": curve, "max_drawdown_pct": round(dd, 2),
        "bias_distribution": df["bias"].value_counts().to_dict(),
        "generated_at": int(time.time()), "seconds": round(secs, 1), "sample_recent": real[-10:],
    }


def run(daily: dict) -> dict:
    t0 = time.time()
    sig = signals(daily)
    if not sig:
        return {"error": "not enough history"}
    scores = np.array([float(np.clip(0.5 * s["tech"] + 0.45 * s["reg"], -1, 1)) for s in sig])
    real = [plan_and_score(s, sc) for s, sc in zip(sig, scores)]
    shuffled = np.random.default_rng(0).permutation(scores)
    base = [plan_and_score(s, sc) for s, sc in zip(sig, shuffled)]
    return summarise(real, base, time.time() - t0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = ["NIFTY"] + [n for n in LAG_SPEC if n in D.SYMBOLS]
    daily = D.yahoo_batch(names, period="max", interval="1d")
    if "NIFTY" not in daily:
        print("no Nifty daily data", D.STATUS)
        sys.exit(1)
    res = run(daily)
    path = os.path.join(args.out, "backtest.json")
    try:
        with open(path) as f:
            bt = json.load(f)
    except Exception:
        bt = {}
    bt["twenty_year"] = res
    bt["generated_at"] = int(time.time())  # so the dashboard sync notices the change

    def enc(o):
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    with open(path, "w") as f:
        json.dump(bt, f, separators=(",", ":"), default=enc)
    print(json.dumps({"window": res.get("window"), "overall": res.get("overall"), "baseline": res.get("baseline"),
                      "seconds": res.get("seconds"), "sources": D.STATUS}, indent=1, default=str))


if __name__ == "__main__":
    main()
