"""Regression layer.

Two models, both fitted fresh on every run and reported with their real
R² and t-stats so the page never hides how weak or strong the fit is:

1. Daily factor model — Nifty's daily return vs. the previous US session,
   same-day Asia, commodities, dollar, yields, VIX, USD/INR and the US-listed
   India ETF (INDA). Gives a factor-implied close for today; the gap between
   spot and that implied level is the daily regression signal.

2. Intraday model — 5-minute Nifty returns vs. contemporaneous moves in ES/NQ
   futures, Asian indices, Brent, gold, DXY, USD/INR, US 10y, Bank Nifty and
   India VIX (betas), plus a small predictive regression of the next 1h/2h
   Nifty return on (a) the gap between spot and the factor-implied path since
   the open, (b) 30-minute factor momentum and (c) Nifty's own 30-minute
   momentum. The predictive fit's R² is reported; it is normally small.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from data import IST, DAILY_FACTORS, INTRADAY_FACTORS, session_filter

BARS_PER_HOUR = 12
HORIZONS = {"1h": 12, "2h": 24}


# -------------------------------------------------------------------- OLS ---
def ols(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6, names=None) -> dict:
    """OLS with intercept, tiny ridge for stability, HC0 robust t-stats."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    Xc = np.column_stack([np.ones(n), X])
    XtX = Xc.T @ Xc + ridge * n * np.eye(k + 1)
    XtX[0, 0] -= ridge * n  # don't shrink intercept
    beta = np.linalg.solve(XtX, Xc.T @ y)
    resid = y - Xc @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12
    r2 = 1 - ss_res / ss_tot
    adj = 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1)
    XtX_inv = np.linalg.inv(XtX)
    meat = (Xc * resid[:, None]).T @ (Xc * resid[:, None])
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    t = beta / se
    out = {"beta": beta, "t": t, "r2": float(r2), "adj_r2": float(adj), "n": int(n),
           "rmse": float(np.sqrt(ss_res / max(n - k - 1, 1))), "resid_std": float(resid.std(ddof=1)) if n > 2 else 0.0}
    if names is not None:
        out["coef_table"] = [{"factor": "intercept", "beta": float(beta[0]), "t": float(t[0])}] + [
            {"factor": nm, "beta": float(b), "t": float(tt)} for nm, b, tt in zip(names, beta[1:], t[1:])]
    return out


def predict(model: dict, x: np.ndarray) -> float:
    return float(model["beta"][0] + np.dot(model["beta"][1:], np.asarray(x, float)))


# ------------------------------------------------------------ daily model ---
def _dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """tz-aware timestamps -> naive IST calendar dates."""
    if idx.tz is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    return pd.DatetimeIndex(idx.normalize())


def _daily_series(close: pd.Series) -> pd.Series:
    r = np.log(close.astype(float)).diff().dropna()
    r.index = _dates(r.index)
    return r[~r.index.duplicated(keep="last")].sort_index()


def _lagged_us_return(close: pd.Series, india_dates: pd.DatetimeIndex) -> pd.Series:
    """Return of the latest session that *ended before* each Indian date (date-only)."""
    r = _daily_series(close)
    dates = r.index.values
    out = []
    for d in india_dates:
        pos = int(np.searchsorted(dates, np.datetime64(d))) - 1
        if pos < 0:
            out.append(0.0)
            continue
        age = (d - pd.Timestamp(dates[pos])).days
        out.append(float(r.iloc[pos]) if age <= 3 else 0.0)
    return pd.Series(out, index=india_dates)


def _same_day_return(close: pd.Series, india_dates: pd.DatetimeIndex) -> pd.Series:
    r = _daily_series(close)
    return r.reindex(india_dates).fillna(0.0)


DAILY_SPEC = {  # name: (alignment, kind)  kind: 'ret' log-return or 'diff' level change
    "SPX": ("lag", "ret"), "NDX": ("lag", "ret"), "INDA": ("lag", "ret"), "EEM": ("lag", "ret"),
    "VIX": ("lag", "diff"), "US10Y": ("lag", "diff"), "DXY": ("lag", "ret"), "BRENT": ("lag", "ret"),
    "GOLD": ("lag", "ret"), "COPPER": ("lag", "ret"), "STOXX50": ("lag", "ret"),
    "NIKKEI": ("same", "ret"), "HSI": ("same", "ret"), "KOSPI": ("same", "ret"), "SHANGHAI": ("same", "ret"),
    "USDINR": ("same", "ret"), "INDIAVIX": ("lag", "diff"),
}


def daily_model(daily: dict[str, pd.DataFrame], lookback: int = 250, asof: pd.Timestamp | None = None,
                target_date: pd.Timestamp | None = None) -> dict:
    nifty = daily["NIFTY"]
    if asof is not None:
        nifty = nifty[nifty.index.normalize() <= asof.normalize()]
    y_full = _daily_series(nifty["close"])
    if target_date is not None:
        td = pd.Timestamp(target_date)
        if td.tzinfo is not None:
            td = td.tz_convert(IST).tz_localize(None)
        td = td.normalize()
        if td > y_full.index[-1]:
            # session hasn't printed a daily bar yet (pre-open): add an empty
            # row so today's factor inputs are still computed.
            y_full = pd.concat([y_full, pd.Series([np.nan], index=pd.DatetimeIndex([td]))])
    feats, names = [], []
    for nm, (align, kind) in DAILY_SPEC.items():
        if nm not in daily:
            continue
        cl = daily[nm]["close"].astype(float)
        if asof is not None:
            cl = cl[cl.index <= asof]
        if kind == "diff":
            # level change in points (VIX / yields): exp(level/10) turns a
            # 1-point change into ~10% in log-return units so the same
            # aligners can be reused.
            cl = np.exp(cl / 10.0)
        s = _lagged_us_return(cl, y_full.index) if align == "lag" else _same_day_return(cl, y_full.index)
        feats.append(s.values)
        names.append(nm)
    if not feats:
        return {"ok": False, "reason": "no factors"}
    X = np.column_stack(feats)
    y = y_full.values
    # fit on the window ending yesterday (today's row isn't complete)
    X_fit, y_fit = X[-(lookback + 1):-1], y[-(lookback + 1):-1]
    mask = ~np.isnan(X_fit).any(axis=1)
    m = ols(X_fit[mask], y_fit[mask], ridge=1e-3, names=names)
    x_today = np.nan_to_num(X[-1])
    pred_today = predict(m, x_today)
    contrib = {nm: float(b * xv) for nm, b, xv in zip(names, m["beta"][1:], x_today)}
    return {
        "ok": True, "n": m["n"], "r2": m["r2"], "adj_r2": m["adj_r2"], "resid_std": m["resid_std"],
        "coef_table": [dict(c, beta_bp_per_pct=round(c["beta"] * 100, 2) if c["factor"] != "intercept" else None)
                       for c in m["coef_table"]],
        "pred_today_log_ret": pred_today, "contributions": contrib, "factor_values_today": dict(zip(names, map(float, x_today))),
        "today": str(y_full.index[-1].date()),
        "actual_today_log_ret": float(y[-1]),
        "fitted_window": [str(y_full.index[-(lookback + 1)].date()), str(y_full.index[-2].date())],
    }


# --------------------------------------------------------- intraday model ---
def align_factors(nifty5: pd.DataFrame, intraday: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Log-price of each factor as-of each Nifty bar close."""
    idx = nifty5.index
    cols = {}
    for nm in INTRADAY_FACTORS:
        if nm not in intraday or nm == "NIFTY":
            continue
        s = np.log(intraday[nm]["close"].astype(float))
        # Yahoo bar timestamps are bar *starts*; shift by one bar so a factor
        # bar is only used once it has closed.
        s.index = s.index + pd.Timedelta(minutes=5)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        aligned = s.reindex(s.index.union(idx)).ffill().reindex(idx)
        cols[nm] = aligned
    return pd.DataFrame(cols, index=idx)


def intraday_model(nifty5: pd.DataFrame, intraday: dict[str, pd.DataFrame], sessions: int = 20,
                   asof: pd.Timestamp | None = None) -> dict:
    n5 = session_filter(nifty5)
    if asof is not None:
        n5 = n5[n5.index <= asof]
    if len(n5) < 300:
        return {"ok": False, "reason": "not enough intraday bars"}
    F = align_factors(n5, intraday)
    F = F.dropna(axis=1, thresh=int(0.8 * len(F)))
    day = n5.index.normalize()
    ln = np.log(n5["close"].astype(float))
    y = ln.groupby(day).diff()  # within-session 5m returns
    fr = F.groupby(day).diff()  # factor returns over the same intervals
    df = pd.concat([y.rename("y"), fr], axis=1).dropna()
    days = sorted(set(df.index.normalize()))
    fit_days = days[-(sessions + 1):-1]  # everything but the current session
    fit = df[df.index.normalize().isin(fit_days)]
    names = list(fr.columns)
    beta_m = ols(fit[names].values, fit["y"].values, ridge=1e-2, names=names)
    betas = dict(zip(names, map(float, beta_m["beta"][1:])))

    # implied path today: open * exp(cumsum(beta·f))
    today = days[-1]
    tdf = df[df.index.normalize() == today]
    full_today = n5[n5.index.normalize() == today]
    implied_log = np.log(float(full_today["open"].iloc[0])) + (tdf[names].values @ beta_m["beta"][1:]).cumsum()
    implied = pd.Series(np.exp(implied_log), index=tdf.index)
    actual = full_today["close"].reindex(tdf.index)
    gap = np.log(actual.values) - implied_log  # +ve: Nifty above factor-implied

    # predictive regression on history: forward return ~ gap, factor mom, own mom
    rows = []
    for d in days[-(60 if sessions < 60 else sessions):]:
        ddf = df[df.index.normalize() == d]
        sess = n5[n5.index.normalize() == d]
        if len(ddf) < 30 or len(sess) < 30:
            continue
        fimp = (ddf[names].values @ beta_m["beta"][1:])
        cum_imp = fimp.cumsum()
        lc = np.log(sess["close"].reindex(ddf.index).values)
        g = lc - (np.log(float(sess["open"].iloc[0])) + cum_imp)
        own = pd.Series(lc).diff().fillna(0).values
        fmom = pd.Series(fimp).rolling(6).sum().values
        omom = pd.Series(own).rolling(6).sum().values
        tod = np.arange(len(ddf)) / 75.0
        for h, nb in HORIZONS.items():
            fwd = np.full(len(lc), np.nan)
            fwd[:-nb] = lc[nb:] - lc[:-nb]
            # last bars of the day: forward to close
            for i in range(max(0, len(lc) - nb), len(lc) - 1):
                fwd[i] = lc[-1] - lc[i]
            for i in range(6, len(lc) - 1):
                if np.isnan(fwd[i]) or np.isnan(fmom[i]):
                    continue
                rows.append((d, h, g[i], fmom[i], omom[i], tod[i], fwd[i]))
    pred_models, signals = {}, {}
    sigma_1bar = float(fit["y"].std())
    for h, nb in HORIZONS.items():
        R = [r for r in rows if r[1] == h and (asof is None or r[0] < today)]
        if len(R) < 200:
            continue
        Xp = np.array([[r[2], r[3], r[4], r[5]] for r in R])
        yp = np.array([r[6] for r in R])
        pm = ols(Xp, yp, ridge=1e-2, names=["fair_value_gap", "factor_mom_30m", "own_mom_30m", "time_of_day"])
        pred_models[h] = pm
        if len(gap):
            fimp_t = (tdf[names].values @ beta_m["beta"][1:])
            x_now = [gap[-1], float(np.nansum(fimp_t[-6:])), float(np.log(actual.values[-1]) - np.log(actual.values[-7])) if len(actual) > 7 else 0.0,
                     len(tdf) / 75.0]
            p = predict(pm, x_now)
            signals[h] = {"pred_log_ret": p, "pred_points": float(actual.values[-1]) * p,
                          "sigma_h": float(pm["resid_std"]), "z": p / pm["resid_std"] if pm["resid_std"] else 0.0,
                          "inputs": dict(zip(["fair_value_gap", "factor_mom_30m", "own_mom_30m", "time_of_day"], map(float, x_now)))}
    return {
        "ok": True, "fit_sessions": len(fit_days), "n_bars": beta_m["n"], "r2_contemporaneous": beta_m["r2"],
        "betas": betas, "beta_table": beta_m["coef_table"], "sigma_5m": sigma_1bar,
        "implied_path": [[int(t.timestamp()), round(float(v), 2)] for t, v in implied.items()],
        "gap_now_pct": float(gap[-1] * 100) if len(gap) else None,
        "gap_series": [[int(t.timestamp()), round(float(g) * 100, 3)] for t, g in zip(tdf.index, gap)],
        "predictive": {h: {"r2": pm["r2"], "n": pm["n"], "coef_table": pm["coef_table"]} for h, pm in pred_models.items()},
        "signals": signals,
        "factor_moves_today_pct": {nm: round(float(tdf[nm].sum() * 100), 3) for nm in names} if len(tdf) else {},
        "factor_moves_30m_pct": {nm: round(float(tdf[nm].tail(6).sum() * 100), 3) for nm in names} if len(tdf) else {},
    }


def gap_model(nifty5: pd.DataFrame, daily: dict[str, pd.DataFrame], intraday: dict[str, pd.DataFrame]) -> dict:
    """Pre-open gap estimate (GIFT-Nifty proxy): Nifty open gap ~ ES overnight + INDA + SPX."""
    n5 = session_filter(nifty5)
    days = sorted(set(n5.index.normalize()))
    if len(days) < 15 or "ES" not in intraday:
        return {"ok": False}
    es = np.log(intraday["ES"]["close"].astype(float))
    rows = []
    for i in range(1, len(days)):
        d0, d1 = days[i - 1], days[i]
        s0 = n5[n5.index.normalize() == d0]
        s1 = n5[n5.index.normalize() == d1]
        if not len(s0) or not len(s1):
            continue
        t_close = s0.index[-1] + pd.Timedelta(minutes=5)
        t_open = s1.index[0]
        es0 = es[es.index <= t_close]
        es1 = es[es.index <= t_open]
        if not len(es0) or not len(es1):
            continue
        es_on = float(es1.iloc[-1] - es0.iloc[-1])
        gap = float(np.log(s1["open"].iloc[0]) - np.log(s0["close"].iloc[-1]))
        inda = 0.0
        if "INDA" in daily:
            r = np.log(daily["INDA"]["close"]).diff().dropna()
            r.index = r.index.tz_convert(IST).normalize()
            prior = r[r.index < d1]
            if len(prior) and (d1 - prior.index[-1]).days <= 3:
                inda = float(prior.iloc[-1])
        rows.append((d1, es_on, inda, gap))
    if len(rows) < 12:
        return {"ok": False}
    X = np.array([[r[1], r[2]] for r in rows[:-1]])
    y = np.array([r[3] for r in rows[:-1]])
    m = ols(X, y, ridge=1e-3, names=["ES_overnight", "INDA_prev_session"])
    # current estimate: ES since last Nifty close, INDA last session
    last_sess = n5[n5.index.normalize() == days[-1]]
    t_close = last_sess.index[-1] + pd.Timedelta(minutes=5)
    es_now = float(es.iloc[-1] - es[es.index <= t_close].iloc[-1]) if len(es[es.index <= t_close]) else 0.0
    inda_now = 0.0
    if "INDA" in daily:
        r = np.log(daily["INDA"]["close"]).diff().dropna()
        inda_now = float(r.iloc[-1]) if len(r) else 0.0
    est = predict(m, [es_now, inda_now])
    return {"ok": True, "r2": m["r2"], "n": m["n"], "coef_table": m["coef_table"],
            "es_move_since_nifty_close_pct": es_now * 100, "inda_last_session_pct": inda_now * 100,
            "estimated_gap_pct": est * 100, "last_close": float(last_sess["close"].iloc[-1]),
            "estimated_open": float(last_sess["close"].iloc[-1]) * float(np.exp(est))}
