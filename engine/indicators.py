"""Technical indicators and their directional votes.

Every indicator function takes an OHLCV DataFrame (columns: open, high, low,
close, volume) and returns pandas Series. `votes()` turns the current state of
each indicator into a vote in [-1, 1] (bullish > 0) plus a short note so the
page can show *why*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- helpers --
def _s(x):
    return pd.Series(x) if not isinstance(x, pd.Series) else x


def sma(s, n):
    return _s(s).rolling(n, min_periods=n).mean()


def ema(s, n):
    return _s(s).ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s, n):
    return _s(s).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def true_range(df):
    pc = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)


def atr(df, n=14):
    return wilder(true_range(df), n)


def rsi(s, n=14):
    d = _s(s).diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder(-d.clip(upper=0), n)
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(s, fast=12, slow=26, sig=9):
    line = ema(s, fast) - ema(s, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


def stochastic(df, k=14, d=3, smooth=3):
    ll = df["low"].rolling(k).min()
    hh = df["high"].rolling(k).max()
    raw = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    kk = raw.rolling(smooth).mean()
    return kk, kk.rolling(d).mean()


def bollinger(s, n=20, k=2.0):
    m = sma(s, n)
    sd = _s(s).rolling(n).std(ddof=0)
    return m, m + k * sd, m - k * sd, sd


def adx(df, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df)
    atr_ = wilder(tr, n)
    pdi = 100 * wilder(pd.Series(plus_dm, index=df.index), n) / atr_
    mdi = 100 * wilder(pd.Series(minus_dm, index=df.index), n) / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx, n), pdi, mdi


def obv(df):
    sign = np.sign(df["close"].diff()).fillna(0)
    return (sign * df["volume"]).cumsum()


def mfi(df, n=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def cmf(df, n=20):
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    return (mfm * df["volume"]).rolling(n).sum() / df["volume"].rolling(n).sum().replace(0, np.nan)


def cci(df, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    m = tp.rolling(n).mean()
    md = (tp - m).abs().rolling(n).mean()
    return (tp - m) / (0.015 * md.replace(0, np.nan))


def williams_r(df, n=14):
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def roc(s, n=12):
    s = _s(s)
    return 100 * (s / s.shift(n) - 1)


def parabolic_sar(df, af0=0.02, af_max=0.2):
    h, l = df["high"].values, df["low"].values
    n = len(df)
    sar = np.full(n, np.nan)
    trend = np.ones(n, dtype=int)
    if n < 3:
        return pd.Series(sar, index=df.index), pd.Series(trend, index=df.index)
    up = h[1] > h[0]
    ep = h[1] if up else l[1]
    sar[1] = l[0] if up else h[0]
    af = af0
    for i in range(2, n):
        prev = sar[i - 1]
        cur = prev + af * (ep - prev)
        if up:
            cur = min(cur, l[i - 1], l[i - 2])
            if l[i] < cur:  # reversal
                up, cur, ep, af = False, ep, l[i], af0
            else:
                if h[i] > ep:
                    ep, af = h[i], min(af + af0, af_max)
        else:
            cur = max(cur, h[i - 1], h[i - 2])
            if h[i] > cur:
                up, cur, ep, af = True, ep, h[i], af0
            else:
                if l[i] < ep:
                    ep, af = l[i], min(af + af0, af_max)
        sar[i] = cur
        trend[i] = 1 if up else -1
    return pd.Series(sar, index=df.index), pd.Series(trend, index=df.index)


def ichimoku(df, t=9, k=26, s=52):
    conv = (df["high"].rolling(t).max() + df["low"].rolling(t).min()) / 2
    base = (df["high"].rolling(k).max() + df["low"].rolling(k).min()) / 2
    span_a = ((conv + base) / 2).shift(k)
    span_b = ((df["high"].rolling(s).max() + df["low"].rolling(s).min()) / 2).shift(k)
    lagging = df["close"].shift(-k)
    return conv, base, span_a, span_b, lagging


def supertrend(df, n=7, mult=3.0):
    a = atr(df, n)
    hl2 = (df["high"] + df["low"]) / 2
    ub = (hl2 + mult * a).values
    lb = (hl2 - mult * a).values
    c = df["close"].values
    m = len(df)
    fub, flb = ub.copy(), lb.copy()
    st = np.full(m, np.nan)
    dirn = np.ones(m, dtype=int)
    for i in range(1, m):
        if np.isnan(ub[i]):
            continue
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if np.isnan(st[i - 1]):
            dirn[i] = 1 if c[i] > fub[i] else -1
        elif st[i - 1] == fub[i - 1]:
            dirn[i] = 1 if c[i] > fub[i] else -1
        else:
            dirn[i] = -1 if c[i] < flb[i] else 1
        st[i] = flb[i] if dirn[i] == 1 else fub[i]
    return pd.Series(st, index=df.index), pd.Series(dirn, index=df.index)


def keltner(df, n=20, mult=2.0):
    m = ema(df["close"], n)
    a = atr(df, n)
    return m, m + mult * a, m - mult * a


def vwap(df):
    """Session VWAP: resets each calendar day (index must be tz-aware datetimes)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.normalize()
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    out = pv / vv
    # Yahoo often reports zero volume for the index itself: fall back to a
    # cumulative mean of typical price so the line still exists.
    fallback = tp.groupby(day).expanding().mean().reset_index(level=0, drop=True)
    return out.fillna(fallback)


def fib_levels(hi, lo, up=True):
    """Retracement levels of the move lo->hi (up=True) or hi->lo."""
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    out = {}
    for r in ratios:
        out[f"{r:.3f}"] = hi - (hi - lo) * r if up else lo + (hi - lo) * r
    # extensions
    out["1.272"] = hi + (hi - lo) * 0.272 if up else lo - (hi - lo) * 0.272
    out["1.618"] = hi + (hi - lo) * 0.618 if up else lo - (hi - lo) * 0.618
    return out


def swing_points(df, lookback=None, atr_mult=1.0):
    """ZigZag-style swings using an ATR threshold. Returns list of (idx, price, kind)."""
    a = atr(df, 14).bfill()
    c = df["close"].values
    h, l = df["high"].values, df["low"].values
    n = len(df)
    if n < 5:
        return []
    piv = []
    mode = None
    ext_i = 0
    for i in range(1, n):
        th = atr_mult * a.iloc[i]
        if mode is None:
            if h[i] > h[ext_i] + th:
                mode, ext_i = "up", i
                piv.append((0, l[0], "L"))
            elif l[i] < l[ext_i] - th:
                mode, ext_i = "down", i
                piv.append((0, h[0], "H"))
            continue
        if mode == "up":
            if h[i] >= h[ext_i]:
                ext_i = i
            elif l[i] < h[ext_i] - th:
                piv.append((ext_i, h[ext_i], "H"))
                mode, ext_i = "down", i
        else:
            if l[i] <= l[ext_i]:
                ext_i = i
            elif h[i] > l[ext_i] + th:
                piv.append((ext_i, l[ext_i], "L"))
                mode, ext_i = "up", i
    # provisional last extreme
    if mode == "up":
        piv.append((ext_i, h[ext_i], "H*"))
    elif mode == "down":
        piv.append((ext_i, l[ext_i], "L*"))
    return piv


# ------------------------------------------------------------------- votes --
def _clip(x):
    return float(max(-1.0, min(1.0, x)))


def _last(s):
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def compute_all(df: pd.DataFrame, tf: str) -> dict:
    """Compute indicator values for the last bar. tf = 'intraday' or 'daily'."""
    c = df["close"]
    out = {}
    px = float(c.iloc[-1])
    a14 = _last(atr(df, 14))
    out["atr"] = a14

    ma_set = (9, 21, 50, 100, 200) if tf == "intraday" else (20, 50, 100, 200)
    for n in ma_set:
        out[f"sma{n}"] = _last(sma(c, n))
    for n in (9, 21, 50):
        out[f"ema{n}"] = _last(ema(c, n))
    out["rsi14"] = _last(rsi(c, 14))
    ml, ms, mh = macd(c)
    out["macd"], out["macd_signal"], out["macd_hist"] = _last(ml), _last(ms), _last(mh)
    out["macd_hist_prev"] = float(mh.dropna().iloc[-2]) if mh.notna().sum() > 1 else np.nan
    k, d = stochastic(df)
    out["stoch_k"], out["stoch_d"] = _last(k), _last(d)
    bm, bu, bl, bsd = bollinger(c)
    out["bb_mid"], out["bb_upper"], out["bb_lower"] = _last(bm), _last(bu), _last(bl)
    out["bb_width_pct"] = float((bu - bl).dropna().iloc[-1] / px * 100) if bu.notna().any() else np.nan
    bw = ((bu - bl) / bm).dropna()
    out["bb_squeeze"] = bool(len(bw) > 60 and bw.iloc[-1] <= bw.iloc[-120:].quantile(0.15))
    ax, pdi, mdi = adx(df)
    out["adx"], out["plus_di"], out["minus_di"] = _last(ax), _last(pdi), _last(mdi)
    has_vol = df["volume"].fillna(0).sum() > 0
    out["has_volume"] = bool(has_vol)
    if has_vol:
        o = obv(df)
        out["obv"] = _last(o)
        out["obv_slope"] = float(o.diff().tail(10).mean()) if len(o) > 10 else np.nan
        out["obv_ma_rel"] = float(_last(o) - _last(sma(o, 20))) if len(o) > 20 else np.nan
        out["mfi14"] = _last(mfi(df))
        out["cmf20"] = _last(cmf(df))
    out["cci20"] = _last(cci(df))
    out["williams_r"] = _last(williams_r(df))
    out["roc12"] = _last(roc(c, 12))
    psar, ptrend = parabolic_sar(df)
    out["psar"], out["psar_trend"] = _last(psar), int(ptrend.iloc[-1])
    conv, base, spa, spb, lag = ichimoku(df)
    out["ichi_conv"], out["ichi_base"] = _last(conv), _last(base)
    out["ichi_span_a"], out["ichi_span_b"] = _last(spa), _last(spb)
    st, sd_ = supertrend(df)
    out["supertrend"], out["supertrend_dir"] = _last(st), int(sd_.iloc[-1])
    km, ku, kl = keltner(df)
    out["kc_upper"], out["kc_lower"] = _last(ku), _last(kl)
    if tf == "intraday":
        v = vwap(df)
        out["vwap"] = _last(v)
    # swings and fib
    piv = swing_points(df, atr_mult=1.5 if tf == "intraday" else 1.0)
    pts = [p for p in piv if not p[2].endswith("*")]
    # Dominant swing for Fibonacci: the most recent confirmed pivot pair at
    # least 3 ATR tall inside the window; otherwise the window's high/low.
    win = 150 if tf == "intraday" else 120  # ~2 sessions / ~6 months
    seg = df.iloc[-win:]
    hi_i, lo_i = int(seg["high"].values.argmax()), int(seg["low"].values.argmin())
    hi, lo, up_move = float(seg["high"].iloc[hi_i]), float(seg["low"].iloc[lo_i]), lo_i < hi_i
    for a, b in zip(pts[-7:-1], pts[-6:]):
        if a[2] != b[2] and abs(b[1] - a[1]) >= 3 * a14 and b[0] >= len(df) - win:
            up_move = a[2] == "L"
            hi, lo = (b[1], a[1]) if up_move else (a[1], b[1])
    if hi > lo:
        out["swing_high"], out["swing_low"] = hi, lo
        out["fib"] = fib_levels(hi, lo, up=up_move)
        out["fib_direction"] = "up" if up_move else "down"
    out["price"] = px
    out["swings"] = [(int(i), float(p), k) for i, p, k in piv[-12:]]
    return out


def votes(ind: dict, tf: str) -> list[dict]:
    """Turn indicator values into votes. Each vote: name, vote[-1..1], weight, note."""
    px = ind["price"]
    v = []

    def add(name, vote, weight, note):
        if vote is None or (isinstance(vote, float) and np.isnan(vote)):
            return
        v.append({"name": name, "vote": round(_clip(vote), 3), "weight": weight, "note": note})

    # Moving averages: price vs MA and MA slope/stack
    ma_keys = [k for k in ind if k.startswith("sma") and not np.isnan(ind[k])]
    stack = 0
    for k in ma_keys:
        stack += 1 if px > ind[k] else -1
    if ma_keys:
        add("MA stack", stack / len(ma_keys), 1.2,
            f"Price above {sum(1 for k in ma_keys if px > ind[k])}/{len(ma_keys)} SMAs")
    if not np.isnan(ind.get("ema9", np.nan)) and not np.isnan(ind.get("ema21", np.nan)):
        d = (ind["ema9"] - ind["ema21"]) / px * 100
        add("EMA 9/21 cross", np.tanh(d * (8 if tf == "intraday" else 2)), 1.0,
            f"EMA9 {'above' if d > 0 else 'below'} EMA21 by {abs(d):.2f}%")
    if not np.isnan(ind.get("ema50", np.nan)):
        d = (px - ind["ema50"]) / px * 100
        add("EMA50 location", np.tanh(d * (4 if tf == "intraday" else 1)), 0.8,
            f"Price {'above' if d > 0 else 'below'} EMA50 by {abs(d):.2f}%")
    r = ind.get("rsi14")
    if r is not None and not np.isnan(r):
        # momentum vote; fade extremes a little
        vote = (r - 50) / 25
        if r > 75:
            vote = 0.4
        if r < 25:
            vote = -0.4
        add("RSI 14", vote, 1.0, f"RSI {r:.1f} ({'overbought' if r > 70 else 'oversold' if r < 30 else 'neutral zone'})")
    mh, mhp = ind.get("macd_hist"), ind.get("macd_hist_prev")
    if mh is not None and not np.isnan(mh):
        base = np.sign(mh) * 0.6
        if mhp is not None and not np.isnan(mhp):
            base += 0.4 * np.sign(mh - mhp)
        add("MACD", base, 1.1, f"Histogram {mh:+.1f}, {'rising' if (mhp is not None and mh > mhp) else 'falling'}")
    k, d = ind.get("stoch_k"), ind.get("stoch_d")
    if k is not None and not np.isnan(k) and not np.isnan(d):
        vote = np.sign(k - d) * 0.5 + (0.5 if k < 20 else -0.5 if k > 80 else 0)
        add("Stochastic", vote, 0.7, f"%K {k:.0f} / %D {d:.0f}")
    bu, bl, bm = ind.get("bb_upper"), ind.get("bb_lower"), ind.get("bb_mid")
    if bu is not None and not np.isnan(bu):
        pos = (px - bl) / (bu - bl) if bu != bl else 0.5  # 0..1
        # trend-following inside bands, mean-reversion at extremes
        vote = (pos - 0.5) * 1.2
        if pos > 1.0:
            vote = -0.3
        if pos < 0.0:
            vote = 0.3
        note = f"{pos * 100:.0f}% of band" + (" · squeeze" if ind.get("bb_squeeze") else "")
        add("Bollinger", vote, 0.9, note)
    ax, pdi, mdi = ind.get("adx"), ind.get("plus_di"), ind.get("minus_di")
    if ax is not None and not np.isnan(ax):
        strength = min(1.0, max(0.0, (ax - 15) / 25))
        vote = np.sign(pdi - mdi) * strength
        add("ADX / DI", vote, 1.0, f"ADX {ax:.0f}, +DI {pdi:.0f} / -DI {mdi:.0f} ({'trending' if ax > 25 else 'weak trend'})")
    if ind.get("has_volume"):
        if not np.isnan(ind.get("obv_ma_rel", np.nan)):
            add("OBV", np.sign(ind["obv_ma_rel"]) * 0.6 + np.sign(ind.get("obv_slope", 0) or 0) * 0.4, 0.7,
                f"OBV {'above' if ind['obv_ma_rel'] > 0 else 'below'} its 20-MA")
        m = ind.get("mfi14")
        if m is not None and not np.isnan(m):
            add("MFI 14", (m - 50) / 30 if 20 <= m <= 80 else (-0.4 if m > 80 else 0.4), 0.6, f"MFI {m:.0f}")
        cm = ind.get("cmf20")
        if cm is not None and not np.isnan(cm):
            add("Chaikin MF", np.tanh(cm * 6), 0.6, f"CMF {cm:+.2f}")
    cc = ind.get("cci20")
    if cc is not None and not np.isnan(cc):
        add("CCI 20", np.tanh(cc / 150) if abs(cc) < 200 else -np.sign(cc) * 0.3, 0.6, f"CCI {cc:+.0f}")
    wr = ind.get("williams_r")
    if wr is not None and not np.isnan(wr):
        add("Williams %R", (wr + 50) / 30 if -80 < wr < -20 else (0.4 if wr <= -80 else -0.4), 0.5,
            f"%R {wr:.0f} ({'overbought' if wr > -20 else 'oversold' if wr < -80 else 'mid-range'})")
    rc = ind.get("roc12")
    if rc is not None and not np.isnan(rc):
        add("ROC 12", np.tanh(rc * (3 if tf == "intraday" else 0.5)), 0.6, f"ROC {rc:+.2f}%")
    if ind.get("psar") is not None and not np.isnan(ind["psar"]):
        add("Parabolic SAR", 0.8 * ind["psar_trend"], 0.8,
            f"SAR {ind['psar']:.0f} ({'below' if ind['psar_trend'] > 0 else 'above'} price)")
    spa, spb, conv, base = ind.get("ichi_span_a"), ind.get("ichi_span_b"), ind.get("ichi_conv"), ind.get("ichi_base")
    if spa is not None and not np.isnan(spa) and not np.isnan(spb):
        top, bot = max(spa, spb), min(spa, spb)
        vote = 0.7 if px > top else -0.7 if px < bot else 0.0
        if not np.isnan(conv) and not np.isnan(base):
            vote += 0.3 * np.sign(conv - base)
        add("Ichimoku", vote, 1.0, f"Price {'above' if px > top else 'below' if px < bot else 'inside'} cloud")
    if ind.get("supertrend") is not None and not np.isnan(ind["supertrend"]):
        add("Supertrend 7,3", 0.9 * ind["supertrend_dir"], 1.0,
            f"Supertrend {ind['supertrend']:.0f} ({'bullish' if ind['supertrend_dir'] > 0 else 'bearish'})")
    if tf == "intraday" and ind.get("vwap") is not None and not np.isnan(ind["vwap"]):
        d = (px - ind["vwap"]) / px * 100
        add("VWAP", np.tanh(d * 6), 1.2, f"Price {'above' if d > 0 else 'below'} VWAP by {abs(d):.2f}%")
    fib = ind.get("fib")
    if fib:
        # where is price within the retracement of the last swing?
        hi, lo = ind["swing_high"], ind["swing_low"]
        if hi > lo:
            pos = (px - lo) / (hi - lo)
            if pos > 1.0:
                vote, note = 0.5, f"Above the swing high {hi:.0f} — extension zone (127.2% {fib['1.272']:.0f}, 161.8% {fib['1.618']:.0f})" if ind["fib_direction"] == "up" else f"Back above the swing high {hi:.0f}; down-swing negated"
            elif pos < 0.0:
                vote, note = -0.5, f"Below the swing low {lo:.0f} — extension zone" if ind["fib_direction"] == "down" else f"Below the swing low {lo:.0f}; up-swing negated"
            elif ind["fib_direction"] == "up":
                # pullback in an up-move: deeper than 61.8% weakens the case
                vote = 0.6 if pos > 0.618 else 0.2 if pos > 0.5 else -0.2 if pos > 0.382 else -0.6
                note = f"Up-swing {lo:.0f}→{hi:.0f}; retraced {100 * (1 - pos):.0f}% of it"
            else:
                vote = -0.6 if pos < 0.382 else -0.2 if pos < 0.5 else 0.2 if pos < 0.618 else 0.6
                note = f"Down-swing {hi:.0f}→{lo:.0f}; retraced {100 * pos:.0f}% of it"
            add("Fibonacci", vote, 0.8, note)
    return v


def aggregate(votes_list: list[dict]) -> float:
    w = sum(x["weight"] for x in votes_list)
    if w == 0:
        return 0.0
    return _clip(sum(x["vote"] * x["weight"] for x in votes_list) / w)
