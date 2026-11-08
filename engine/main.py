"""Entry point: pull data, run everything, write JSON files to --out.

Files written:
  latest.json      everything the page needs for the current state
  daily.json       long Nifty daily OHLC history (for the History tab)
  intraday.json    last ~10 sessions of 5-minute Nifty bars
  backtest.json    walk-forward hit-rate test (refreshed once a day)
  signal_log.json  every run's prediction, evaluated on later runs
  status.json      which data sources worked on this run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D  # noqa: E402
import indicators as ti  # noqa: E402
from analysis import (backtest, build_levels, cluster_zones, combine, forecast_cone, regression_score,  # noqa: E402
                      technical_score, trade_plan)
from regression import daily_model, gap_model, intraday_model, HORIZONS  # noqa: E402
import daycall  # noqa: E402

IST = D.IST


class Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
            return o.isoformat()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)


def clean(o):
    """Replace NaN/inf with None recursively (json.dumps writes NaN otherwise)."""
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(clean(obj), f, cls=Enc, separators=(",", ":"))


def ohlc_records(df: pd.DataFrame) -> list:
    return [[int(t.timestamp()), round(float(r.open), 2), round(float(r.high), 2), round(float(r.low), 2),
             round(float(r.close), 2), int(r.volume) if np.isfinite(r.volume) else 0]
            for t, r in df.iterrows()]


def market_status(now: pd.Timestamp) -> str:
    if now.weekday() >= 5:
        return "closed-weekend"
    t = now.time()
    if t < dt.time(9, 0):
        return "pre-market"
    if t < dt.time(9, 15):
        return "pre-open"
    if t < dt.time(15, 30):
        return "open"
    return "closed"


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--skip-nse", action="store_true")
    ap.add_argument("--skip-fred", action="store_true")
    ap.add_argument("--force-backtest", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    now = pd.Timestamp.now(tz=IST)
    if os.environ.get("ENGINE_NOW"):  # testing hook: pretend it is another time
        now = pd.Timestamp(os.environ["ENGINE_NOW"]).tz_localize(IST)
    run = {"run_at": int(now.timestamp()), "run_at_ist": now.strftime("%Y-%m-%d %H:%M IST"),
           "market_status": market_status(now), "engine_version": "1.1.2"}

    # ------------------------------------------------------------ data ---
    intraday_names = ["NIFTY"] + D.INTRADAY_FACTORS
    intraday = D.yahoo_batch(intraday_names, period="60d", interval="5m")
    daily_names = ["NIFTY"] + D.DAILY_FACTORS
    daily = D.yahoo_batch(daily_names, period="10y", interval="1d")
    if os.environ.get("ENGINE_NOW"):  # testing hook: hide bars from the "future"
        intraday = {k: v[v.index <= now] for k, v in intraday.items()}
        daily = {k: v[v.index <= now] for k, v in daily.items()}
    if "NIFTY" not in intraday or "NIFTY" not in daily:
        run["error"] = "Nifty data unavailable from Yahoo"
        dump(os.path.join(args.out, "status.json"), {"run": run, "sources": D.STATUS})
        print(json.dumps(D.STATUS, indent=1))
        sys.exit(1)
    n5 = intraday["NIFTY"]
    s5 = D.session_filter(n5)
    nd = daily["NIFTY"]
    # drop today's partial daily bar for daily indicators if the session is open (avoid repainting)
    nd_hist = nd[nd.index.normalize() < now.normalize()] if run["market_status"] in ("open", "pre-open") else nd

    oc = {} if args.skip_nse else D.nse_option_chain()
    fut = [] if args.skip_nse else D.nse_futures()
    flows = [] if args.skip_nse else D.nse_fii_dii()
    if not oc and not args.skip_nse:
        oc = D.upstox_pcr()
        if oc:
            oc = {"source": "Upstox (headline only)", "expiries": [{"expiry": "nearest", "pcr_oi": oc["pcr_oi"]}]}
    macro = {} if args.skip_fred else D.fred_macro()

    # ---------------------------------------------------------- market ---
    days = sorted(set(s5.index.normalize()))
    today = days[-1]
    tb = s5[s5.index.normalize() == today]
    prev_bars = s5[s5.index.normalize() == days[-2]] if len(days) > 1 else tb
    spot = float(tb["close"].iloc[-1])
    prev_close = float(prev_bars["close"].iloc[-1])
    market = {
        "spot": spot, "prev_close": prev_close, "change": spot - prev_close, "change_pct": (spot / prev_close - 1) * 100,
        "session_date": str(today.date()), "last_bar": int(tb.index[-1].timestamp()),
        "open": float(tb["open"].iloc[0]), "high": float(tb["high"].max()), "low": float(tb["low"].min()),
        "bars_today": int(len(tb)),
    }
    manual = load_json(os.path.join(args.out, "manual_inputs.json"), {})
    # A newer official print than Yahoo's last bar (special sessions such as
    # Muhurat trading, which Yahoo publishes late or not at all): prefer a
    # manual session override, then NSE's own spot from the option chain.
    official = None
    so = manual.get("session_override") or {}
    try:
        if so.get("close") and pd.Timestamp(so["date"]).date() > today.date():
            official = {"source": "manual entry", "close": float(so["close"]), "date": str(so["date"])}
    except Exception:
        pass
    if official is None and oc.get("spot") and oc.get("as_of"):
        try:
            nse_t = pd.Timestamp(oc["as_of"])
            if nse_t.date() > today.date() and float(oc["spot"]) > 0:
                official = {"source": f"NSE spot as of {oc['as_of']}", "close": float(oc["spot"]), "date": str(nse_t.date())}
        except Exception:
            pass
    call_ref = prev_close if len(tb) else spot  # reference close for the next session's call
    if official and run["market_status"] != "open":
        market["official_close"] = official
        call_ref = official["close"]
    snap = {}
    for nm in ["BANKNIFTY", "INDIAVIX", "USDINR", "ES", "NQ", "NIKKEI", "HSI", "SHANGHAI", "KOSPI", "BRENT", "WTI",
               "GOLD", "SILVER", "COPPER", "DXY", "US10Y", "USDJPY", "EURUSD", "BTC", "FTSE", "DAX", "STOXX50"]:
        if nm in intraday:
            f = intraday[nm]
            last = float(f["close"].iloc[-1])
            fd = sorted(set(f.index.normalize()))
            pc = float(f[f.index.normalize() == fd[-2]]["close"].iloc[-1]) if len(fd) > 1 else last
            snap[nm] = {"last": last, "chg_pct": (last / pc - 1) * 100, "as_of": int(f.index[-1].timestamp()),
                        "spark": [round(float(x), 3) for x in f["close"].tail(75).values]}
    for nm in ["SPX", "NDX", "DJI", "VIX", "INDA", "EEM", "TAIEX", "US5Y", "US30Y"]:
        if nm in daily:
            f = daily[nm]
            if len(f) > 1:
                snap[nm] = {"last": float(f["close"].iloc[-1]), "chg_pct": (float(f["close"].iloc[-1]) / float(f["close"].iloc[-2]) - 1) * 100,
                            "as_of": int(f.index[-1].timestamp()), "spark": [round(float(x), 3) for x in f["close"].tail(30).values]}
    vix_ratio = 1.0
    if "INDIAVIX" in daily and len(daily["INDIAVIX"]) > 25:
        v = daily["INDIAVIX"]["close"]
        vix_ratio = float(v.iloc[-1] / v.tail(21).mean())

    # ------------------------------------------------------- technical ---
    tech = technical_score(n5, nd_hist, oc if oc.get("expiries") else None, manual)
    ind5, indd = tech["_ind5"], tech["_indd"]

    # ------------------------------------------------------ regression ---
    try:
        dm = daily_model(daily, target_date=now if run["market_status"] in ("pre-market", "pre-open") else today)
    except Exception as e:
        dm = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    try:
        im = intraday_model(n5, intraday)
    except Exception as e:
        im = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    try:
        gm = gap_model(n5, daily, intraday)
    except Exception as e:
        gm = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    reg = regression_score(dm, im, spot, prev_close)

    # --------------------------------------------------------- combine ---
    comb = combine(tech["score"], reg["score"], None)
    sigma_5m = im.get("sigma_5m") or float(np.log(s5["close"]).diff().tail(500).std())
    cone = forecast_cone(n5, comb["score"], sigma_5m, vix_ratio)
    levels = build_levels(s5, nd_hist, ind5, indd, oc if oc.get("expiries") else None)
    zones = cluster_zones(levels, spot, atr5=ind5["atr"])
    plan = trade_plan(spot, zones, comb, cone, ind5["atr"])

    # -------------------------------------------------------- backtest ---
    bt_path = os.path.join(args.out, "backtest.json")
    bt_old = load_json(bt_path, None)
    need_bt = args.force_backtest or not bt_old or (now.timestamp() - bt_old.get("generated_at", 0) > 20 * 3600)
    if need_bt:
        try:
            bt = backtest(n5, nd, daily, intraday)
            dump(bt_path, bt)
        except Exception as e:
            bt = bt_old or {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
    else:
        bt = bt_old

    # ------------------------------------------------------ signal log ---
    log = load_json(os.path.join(args.out, "signal_log.json"), [])
    # evaluate earlier, unevaluated signals against what has happened since
    for rec in log:
        if rec.get("evaluated"):
            continue
        t_sig = rec["t"]
        for h, nb in HORIZONS.items():
            if f"actual_{h}" in rec:
                continue
            later = s5[s5.index > pd.Timestamp(t_sig, unit="s", tz=IST)]
            if len(later) >= nb:
                a = float(later["close"].iloc[nb - 1])
                rec[f"actual_{h}"] = round(a, 2)
                rec[f"ret_{h}_pct"] = round((a / rec["spot"] - 1) * 100, 3)
                rec[f"hit_{h}"] = bool(np.sign(rec[f"ret_{h}_pct"]) == np.sign(rec["score"])) if abs(rec["score"]) > 0.15 else None
        if all(f"actual_{h}" in rec for h in HORIZONS):
            rec["evaluated"] = True
    if run["market_status"] == "open":
        log.append({"t": market["last_bar"], "run_at": run["run_at"], "spot": spot, "score": round(comb["score"], 3),
                    "bias": comb["bias"], "tech": round(tech["score"], 3), "reg": round(reg["score"], 3),
                    "exp_1h": cone["expected_move_points"]["1h"], "exp_2h": cone["expected_move_points"]["2h"]})
    log = log[-600:]
    dump(os.path.join(args.out, "signal_log.json"), log)
    ev = [r for r in log if r.get("hit_1h") is not None]
    live_record = {"n": len(ev), "hit_rate_1h": round(sum(1 for r in ev if r["hit_1h"]) / len(ev), 3) if ev else None,
                   "hit_rate_2h": round(sum(1 for r in ev if r.get("hit_2h")) / len([r for r in ev if r.get("hit_2h") is not None]), 3)
                   if any(r.get("hit_2h") is not None for r in ev) else None}

    # -------------------------------------------------------- day call ---
    tb_today = s5[s5.index.normalize() == now.normalize()]
    try:
        dc_log = load_json(os.path.join(args.out, "daily_calls.json"), {"calls": []})
        if len(tb_today):
            call_spot, call_prev, call_gm = spot, prev_close, gm
        else:
            # before the open: the latest official close is both the reference and
            # the base for the gap estimate
            call_spot = call_prev = call_ref
            call_gm = dict(gm)
            if gm.get("ok"):
                call_gm["estimated_open"] = call_ref * float(np.exp(gm["estimated_gap_pct"] / 100))
        dc_log = daycall.update_log(
            dc_log, now, s5, run["market_status"],
            lambda: daycall.make_day_call(now, tb_today, nd_hist, tech, dict(dm), call_gm, im, zones, vix_ratio, call_spot, call_prev))
        dump(os.path.join(args.out, "daily_calls.json"), dc_log)
        today_str = now.strftime("%Y-%m-%d")
        day_call = next((c for c in dc_log["calls"] if c["date"] == today_str), None) or (dc_log["calls"][-1] if dc_log["calls"] else None)
        results_summary = dc_log.get("summary")
    except Exception as e:
        traceback.print_exc()
        day_call, results_summary = {"error": f"{type(e).__name__}: {e}"}, None

    # ------------------------------------------------------------ write ---
    run["seconds"] = round(time.time() - t0, 1)
    latest = {
        "run": run, "market": market, "snapshot": snap, "vix_ratio": vix_ratio,
        "prediction": {"combined": comb, "technical_score": tech["score"], "regression_score": reg["score"],
                       "cone": cone, "plan": plan, "horizons_bars": HORIZONS},
        "day_call": day_call, "results_summary": results_summary,
        "technical": {k: v for k, v in tech.items() if not k.startswith("_")},
        "regression": {"daily": dm, "intraday": im, "gap_model": gm, "score": reg},
        "levels": levels, "zones": zones,
        "option_chain": oc, "futures": fut, "fii_dii": flows, "macro": macro,
        "backtest_summary": (bt or {}).get("summary"), "live_record": live_record,
        "series": {"today_5m": ohlc_records(tb), "prev_5m": ohlc_records(prev_bars),
                   "vwap_today": [[int(t.timestamp()), round(float(v), 2)] for t, v in ti.vwap(tb).items()],
                   "implied_path": im.get("implied_path") if im.get("ok") else None},
        "sources": D.STATUS,
    }
    dump(os.path.join(args.out, "latest.json"), latest)
    dump(os.path.join(args.out, "daily.json"), {"symbol": "NIFTY 50", "bars": ohlc_records(nd.tail(1500))})
    recent_days = days[-10:]
    dump(os.path.join(args.out, "intraday.json"), {"symbol": "NIFTY 50", "interval": "5m",
                                                    "bars": ohlc_records(s5[s5.index.normalize().isin(recent_days)])})
    dump(os.path.join(args.out, "status.json"), {"run": run, "sources": D.STATUS})
    print(json.dumps({"run": run, "spot": spot, "score": comb, "sources": D.STATUS}, indent=1, cls=Enc, default=str))


if __name__ == "__main__":
    main()
