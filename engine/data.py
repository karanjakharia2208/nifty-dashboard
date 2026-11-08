"""Data access: Yahoo Finance (via yfinance), NSE (best effort), FRED (macro).

Everything returns tz-aware (Asia/Kolkata) frames with lowercase columns.
Failures never raise out of here: callers get None / {} and a note in
`STATUS` so the page can show what was and wasn't available.
"""
from __future__ import annotations

import io
import json
import time
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

IST = "Asia/Kolkata"
STATUS: dict[str, str] = {}

# Yahoo symbols. Key = our short name.
SYMBOLS = {
    # India
    "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN", "INDIAVIX": "^INDIAVIX",
    "USDINR": "INR=X", "INDA": "INDA",
    # US / global equities
    "ES": "ES=F", "NQ": "NQ=F", "SPX": "^GSPC", "NDX": "^IXIC", "DJI": "^DJI", "VIX": "^VIX",
    "NIKKEI": "^N225", "HSI": "^HSI", "SHANGHAI": "000001.SS", "KOSPI": "^KS11", "TAIEX": "^TWII",
    "FTSE": "^FTSE", "DAX": "^GDAXI", "STOXX50": "^STOXX50E", "EEM": "EEM",
    # commodities / FX / rates
    "BRENT": "BZ=F", "WTI": "CL=F", "GOLD": "GC=F", "SILVER": "SI=F", "COPPER": "HG=F",
    "DXY": "DX-Y.NYB", "EURUSD": "EURUSD=X", "USDJPY": "JPY=X", "USDCNY": "CNY=X",
    "US10Y": "^TNX", "US5Y": "^FVX", "US30Y": "^TYX", "US13W": "^IRX",
    "BTC": "BTC-USD",
}
# Factors that trade around the clock or overlap Indian hours -> usable intraday
INTRADAY_FACTORS = ["ES", "NQ", "NIKKEI", "HSI", "SHANGHAI", "KOSPI", "BRENT", "WTI", "GOLD", "SILVER",
                    "COPPER", "DXY", "EURUSD", "USDJPY", "US10Y", "USDINR", "BANKNIFTY", "INDIAVIX", "BTC",
                    "FTSE", "DAX", "STOXX50"]
DAILY_FACTORS = ["SPX", "NDX", "DJI", "VIX", "ES", "NQ", "NIKKEI", "HSI", "SHANGHAI", "KOSPI", "TAIEX", "FTSE",
                 "DAX", "STOXX50", "EEM", "INDA", "BRENT", "WTI", "GOLD", "SILVER", "COPPER", "DXY", "EURUSD",
                 "USDJPY", "USDCNY", "US10Y", "US5Y", "US30Y", "US13W", "BTC", "USDINR", "INDIAVIX", "BANKNIFTY"]

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/124.0 Safari/537.36")


def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]
    if "volume" not in df:
        df["volume"] = 0.0
    df = df.dropna(subset=["close"])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df if len(df) else None


def yahoo_batch(names: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    """Download several symbols at once; returns {name: frame}."""
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    syms = [SYMBOLS[n] for n in names]
    for attempt in range(3):
        try:
            raw = yf.download(syms, period=period, interval=interval, group_by="ticker",
                              auto_adjust=False, threads=True, progress=False)
            break
        except Exception as e:  # pragma: no cover
            STATUS[f"yahoo_{interval}"] = f"attempt {attempt + 1} failed: {e}"
            time.sleep(3 + 3 * attempt)
            raw = None
    if raw is None or len(raw) == 0:
        STATUS[f"yahoo_{interval}"] = "download failed"
        return out
    for n, s in zip(names, syms):
        try:
            sub = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
            cl = _clean(sub)
            if cl is not None and len(cl) > 5:
                out[n] = cl
        except Exception:
            pass
    missing = [n for n in names if n not in out]
    STATUS[f"yahoo_{interval}"] = f"ok {len(out)}/{len(names)}" + (f"; missing {missing}" if missing else "")
    return out


def session_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only NSE regular-session bars (09:15–15:30 IST)."""
    t = df.index.time
    import datetime as dt
    m = (t >= dt.time(9, 15)) & (t < dt.time(15, 30))
    return df[m]


# ------------------------------------------------------------------- NSE ----
def _nse_session():
    s = requests.Session()
    s.headers.update({"User-Agent": _UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9",
                      "Referer": "https://www.nseindia.com/option-chain"})
    s.get("https://www.nseindia.com", timeout=12)
    s.get("https://www.nseindia.com/option-chain", timeout=12)
    return s


def nse_option_chain(symbol="NIFTY") -> dict:
    """PCR / OI by expiry, max pain, OI walls. Best effort; {} on failure."""
    try:
        s = _nse_session()
        r = s.get(f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}", timeout=15)
        r.raise_for_status()
        js = r.json()
        recs = js["records"]
        spot = float(recs.get("underlyingValue") or 0)
        rows = recs["data"]
        by_exp: dict[str, dict] = {}
        for row in rows:
            e = row["expiryDate"]
            d = by_exp.setdefault(e, {"strikes": {}, "ce_oi": 0.0, "pe_oi": 0.0, "ce_vol": 0.0, "pe_vol": 0.0,
                                      "ce_chg": 0.0, "pe_chg": 0.0})
            k = float(row["strikePrice"])
            ce, pe = row.get("CE") or {}, row.get("PE") or {}
            ceoi, peoi = float(ce.get("openInterest", 0) or 0), float(pe.get("openInterest", 0) or 0)
            d["strikes"][k] = (ceoi, peoi)
            d["ce_oi"] += ceoi
            d["pe_oi"] += peoi
            d["ce_vol"] += float(ce.get("totalTradedVolume", 0) or 0)
            d["pe_vol"] += float(pe.get("totalTradedVolume", 0) or 0)
            d["ce_chg"] += float(ce.get("changeinOpenInterest", 0) or 0)
            d["pe_chg"] += float(pe.get("changeinOpenInterest", 0) or 0)
        expiries = []
        for e in recs.get("expiryDates", [])[:6]:
            d = by_exp.get(e)
            if not d or not d["strikes"]:
                continue
            ks = sorted(d["strikes"])
            # max pain: strike minimising total option writers' payout
            pains = []
            for k in ks:
                pain = 0.0
                for kk, (coi, poi) in d["strikes"].items():
                    pain += coi * max(0.0, k - kk) + poi * max(0.0, kk - k)
                pains.append((pain, k))
            max_pain = min(pains)[1] if pains else None
            near = [k for k in ks if abs(k - spot) <= 0.04 * spot] if spot else ks
            call_wall = max(near, key=lambda k: d["strikes"][k][0]) if near else None
            put_wall = max(near, key=lambda k: d["strikes"][k][1]) if near else None
            expiries.append({
                "expiry": e,
                "pcr_oi": round(d["pe_oi"] / d["ce_oi"], 3) if d["ce_oi"] else None,
                "pcr_vol": round(d["pe_vol"] / d["ce_vol"], 3) if d["ce_vol"] else None,
                "ce_oi": d["ce_oi"], "pe_oi": d["pe_oi"], "ce_oi_chg": d["ce_chg"], "pe_oi_chg": d["pe_chg"],
                "max_pain": max_pain, "call_wall": call_wall, "put_wall": put_wall,
                "top_calls": sorted(near, key=lambda k: -d["strikes"][k][0])[:3] if near else [],
                "top_puts": sorted(near, key=lambda k: -d["strikes"][k][1])[:3] if near else [],
            })
        STATUS["nse_option_chain"] = f"ok {len(expiries)} expiries"
        return {"source": "NSE", "spot": spot, "as_of": recs.get("timestamp"), "expiries": expiries}
    except Exception as e:
        STATUS["nse_option_chain"] = f"failed: {type(e).__name__}: {str(e)[:120]}"
        return {}


def nse_futures(symbol="NIFTY") -> list[dict]:
    """Nifty futures OI/volume by expiry. Best effort."""
    try:
        s = _nse_session()
        r = s.get(f"https://www.nseindia.com/api/quote-derivative?symbol={symbol}", timeout=15)
        r.raise_for_status()
        js = r.json()
        out = []
        for st in js.get("stocks", []):
            md = st.get("metadata", {})
            if md.get("instrumentType", "").lower().startswith("index fut"):
                mi = st.get("marketDeptOrderBook", {}).get("tradeInfo", {})
                out.append({
                    "expiry": md.get("expiryDate"), "last": md.get("lastPrice"),
                    "change_pct": md.get("pChange"), "oi": mi.get("openInterest"),
                    "oi_change": mi.get("changeinOpenInterest"), "oi_change_pct": mi.get("pchangeinOpenInterest"),
                    "volume_contracts": mi.get("tradedVolume"), "value_cr": mi.get("value"),
                })
        STATUS["nse_futures"] = f"ok {len(out)} contracts"
        return out
    except Exception as e:
        STATUS["nse_futures"] = f"failed: {type(e).__name__}"
        return []


def nse_fii_dii() -> list[dict]:
    try:
        s = _nse_session()
        r = s.get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=15)
        r.raise_for_status()
        js = r.json()
        STATUS["nse_fii_dii"] = "ok"
        return js if isinstance(js, list) else []
    except Exception as e:
        STATUS["nse_fii_dii"] = f"failed: {type(e).__name__}"
        return []


# -------------------------------------------------------------- Upstox PCR --
def upstox_pcr() -> dict:
    """Headline Nifty PCR from Upstox's public page (fallback when NSE blocks)."""
    try:
        r = requests.get("https://upstox.com/fno-discovery/open-interest-analysis/nifty-pcr/",
                         headers={"User-Agent": _UA}, timeout=15)
        r.raise_for_status()
        import re
        txt = r.text
        m = re.search(r"PCR[^0-9]{0,80}?([0-9]\.[0-9]{2,6})", txt)
        if m:
            STATUS["upstox_pcr"] = "ok"
            return {"source": "Upstox", "pcr_oi": float(m.group(1))}
        STATUS["upstox_pcr"] = "no number found"
    except Exception as e:
        STATUS["upstox_pcr"] = f"failed: {type(e).__name__}"
    return {}


# ------------------------------------------------------------------ FRED ----
FRED_SERIES = {
    "CPIAUCSL": "US CPI (index)", "UNRATE": "US unemployment rate %", "FEDFUNDS": "Fed funds rate %",
    "ICSA": "US initial jobless claims", "INDPRO": "US industrial production", "HOUST": "US housing starts (k)",
    "UMCSENT": "US consumer sentiment", "T10Y2Y": "US 10y–2y spread", "DGS10": "US 10y yield %",
    "DGS2": "US 2y yield %", "BOPGSTB": "US trade balance ($m)", "GDPC1": "US real GDP (bn, saar)",
    "DTWEXBGS": "Broad dollar index",
}


def fred_macro() -> dict:
    out = {}
    for sid, label in FRED_SERIES.items():
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=15,
                             headers={"User-Agent": _UA})
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna().tail(14)
            if len(df) >= 2:
                last, prev = df.iloc[-1], df.iloc[-2]
                out[sid] = {"label": label, "date": str(last["date"]), "value": float(last["value"]),
                            "prev": float(prev["value"]), "prev_date": str(prev["date"]),
                            "yoy": (float(last["value"]) / float(df.iloc[-13]["value"]) - 1) * 100
                            if len(df) >= 13 and sid in ("CPIAUCSL", "INDPRO") else None,
                            "history": [[str(d), float(v)] for d, v in zip(df["date"], df["value"])]}
        except Exception as e:
            out[sid] = {"label": label, "error": type(e).__name__}
    STATUS["fred"] = f"ok {sum(1 for v in out.values() if 'value' in v)}/{len(FRED_SERIES)}"
    return out
