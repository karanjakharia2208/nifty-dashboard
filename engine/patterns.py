"""Chart-pattern detection on swing points.

Patterns are detected from ZigZag swings (see indicators.swing_points) plus a
few trendline fits. Each hit: {name, bias (+1/-1/0), status (forming|confirmed
|failed), confidence 0..1, target, neckline/trigger, note}. Detection is
deliberately conservative: a pattern only counts when the geometry is close
to the textbook shape, and most detections stay "forming" until the breakout
level is crossed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import swing_points, atr


def _near(a, b, tol):
    return abs(a - b) <= tol


def _fit_line(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 2:
        return 0.0, float(ys[0]) if len(ys) else 0.0
    m, c = np.polyfit(xs, ys, 1)
    return float(m), float(c)


def detect(df: pd.DataFrame, tf: str) -> list[dict]:
    n = len(df)
    if n < 60:
        return []
    a = float(atr(df, 14).dropna().iloc[-1])
    px = float(df["close"].iloc[-1])
    piv = swing_points(df, atr_mult=1.5 if tf == "intraday" else 1.0)
    pts = [(i, p, k[0]) for i, p, k in piv]
    tol = 0.9 * a  # equality tolerance for tops/bottoms
    recent = 60 if tf == "intraday" else 40  # last pivot must be this recent (bars)
    near_pct = 0.015 if tf == "intraday" else 0.04  # trigger must be this close to price
    hits = []

    def add(name, bias, status, conf, target, trigger, note, idx_from):
        hits.append({
            "name": name, "bias": bias, "status": status, "confidence": round(float(conf), 2),
            "target": None if target is None else round(float(target), 1),
            "trigger": None if trigger is None else round(float(trigger), 1),
            "note": note, "bars_ago": int(n - 1 - idx_from), "timeframe": tf,
        })

    # ---- reversal patterns on the last 3-7 swings ----
    if len(pts) >= 4:
        last = pts[-6:]
        # Double top: H, L, H (equal highs) then price below the middle low
        for j in range(len(last) - 2):
            a1, b, a2 = last[j], last[j + 1], last[j + 2]
            if a1[2] == "H" and b[2] == "L" and a2[2] == "H" and _near(a1[1], a2[1], tol) and a2[0] >= n - recent:
                neck = b[1]
                height = max(a1[1], a2[1]) - neck
                status = "confirmed" if px < neck else "forming"
                add("Double top", -1, status, 0.55 if status == "forming" else 0.75,
                    neck - height, neck, f"Twin highs near {a1[1]:.0f}; neckline {neck:.0f}", a1[0])
            if a1[2] == "L" and b[2] == "H" and a2[2] == "L" and _near(a1[1], a2[1], tol) and a2[0] >= n - recent:
                neck = b[1]
                height = neck - min(a1[1], a2[1])
                status = "confirmed" if px > neck else "forming"
                add("Double bottom", 1, status, 0.55 if status == "forming" else 0.75,
                    neck + height, neck, f"Twin lows near {a1[1]:.0f}; neckline {neck:.0f}", a1[0])
    if len(pts) >= 5:
        last = pts[-7:]
        for j in range(len(last) - 4):
            s = last[j:j + 5]
            if s[-1][0] < n - recent:
                continue
            kinds = "".join(p[2] for p in s)
            if kinds == "HLHLH":
                h1, l1, h2, l2, h3 = s
                # Head & shoulders: middle high highest, shoulders similar
                if h2[1] > h1[1] + 0.3 * a and h2[1] > h3[1] + 0.3 * a and _near(h1[1], h3[1], 2 * tol):
                    m, c = _fit_line([l1[0], l2[0]], [l1[1], l2[1]])
                    neck = m * (n - 1) + c
                    height = h2[1] - (l1[1] + l2[1]) / 2
                    status = "confirmed" if px < neck else "forming"
                    add("Head & shoulders", -1, status, 0.5 if status == "forming" else 0.75,
                        neck - height, neck, f"Head {h2[1]:.0f}, shoulders {h1[1]:.0f}/{h3[1]:.0f}, neckline ≈{neck:.0f}", h1[0])
                # Triple top
                elif _near(h1[1], h2[1], tol) and _near(h2[1], h3[1], tol):
                    neck = min(l1[1], l2[1])
                    height = max(h1[1], h2[1], h3[1]) - neck
                    status = "confirmed" if px < neck else "forming"
                    add("Triple top", -1, status, 0.5 if status == "forming" else 0.7, neck - height, neck,
                        f"Three highs near {h2[1]:.0f}", h1[0])
            if kinds == "LHLHL":
                l1, h1, l2, h2, l3 = s
                if l2[1] < l1[1] - 0.3 * a and l2[1] < l3[1] - 0.3 * a and _near(l1[1], l3[1], 2 * tol):
                    m, c = _fit_line([h1[0], h2[0]], [h1[1], h2[1]])
                    neck = m * (n - 1) + c
                    height = (h1[1] + h2[1]) / 2 - l2[1]
                    status = "confirmed" if px > neck else "forming"
                    add("Inverse head & shoulders", 1, status, 0.5 if status == "forming" else 0.75,
                        neck + height, neck, f"Head {l2[1]:.0f}, shoulders {l1[1]:.0f}/{l3[1]:.0f}, neckline ≈{neck:.0f}", l1[0])
                elif _near(l1[1], l2[1], tol) and _near(l2[1], l3[1], tol):
                    neck = max(h1[1], h2[1])
                    height = neck - min(l1[1], l2[1], l3[1])
                    status = "confirmed" if px > neck else "forming"
                    add("Triple bottom", 1, status, 0.5 if status == "forming" else 0.7, neck + height, neck,
                        f"Three lows near {l2[1]:.0f}", l1[0])

    # ---- continuation / consolidation via trendlines over the last window ----
    win = 60 if tf == "intraday" else 40
    seg = df.iloc[-win:]
    recent_pts = [p for p in pts if p[0] >= n - win]
    hs = [(i, p) for i, p, k in recent_pts if k == "H"]
    ls = [(i, p) for i, p, k in recent_pts if k == "L"]
    if len(hs) >= 2 and len(ls) >= 2:
        mh, ch = _fit_line([i for i, _ in hs], [p for _, p in hs])
        ml, cl = _fit_line([i for i, _ in ls], [p for _, p in ls])
        x = n - 1
        upper, lower = mh * x + ch, ml * x + cl
        slope_h, slope_l = mh / a, ml / a  # ATR per bar
        width0 = (mh * hs[0][0] + ch) - (ml * ls[0][0] + cl)
        width1 = upper - lower
        converging = width1 < 0.7 * width0 and width1 > 0
        flat_h, flat_l = abs(slope_h) < 0.04, abs(slope_l) < 0.04
        prior = seg["close"].iloc[0] - df["close"].iloc[max(0, n - win - win // 2)]
        start_idx = min(hs[0][0], ls[0][0])
        if converging:
            if flat_h and slope_l > 0.04:
                st = "confirmed" if px > upper else "forming"
                add("Ascending triangle", 1, st, 0.55 if st == "forming" else 0.75, upper + width0, upper,
                    f"Flat top ≈{upper:.0f}, rising lows", start_idx)
            elif flat_l and slope_h < -0.04:
                st = "confirmed" if px < lower else "forming"
                add("Descending triangle", -1, st, 0.55 if st == "forming" else 0.75, lower - width0, lower,
                    f"Flat base ≈{lower:.0f}, falling highs", start_idx)
            elif slope_h < -0.04 and slope_l > 0.04:
                bias = 1 if prior > 0 else -1
                st = "confirmed" if (px > upper or px < lower) else "forming"
                add("Symmetrical triangle", bias if st == "forming" else (1 if px > upper else -1), st, 0.45,
                    (upper + width0) if bias > 0 else (lower - width0), upper if bias > 0 else lower,
                    f"Apex forming between {lower:.0f} and {upper:.0f}", start_idx)
            elif slope_h > 0.04 and slope_l > 0.04:
                st = "confirmed" if px < lower else "forming"
                add("Rising wedge", -1, st, 0.45 if st == "forming" else 0.65, lower - width0, lower,
                    f"Both lines rising, narrowing; support ≈{lower:.0f}", start_idx)
            elif slope_h < -0.04 and slope_l < -0.04:
                st = "confirmed" if px > upper else "forming"
                add("Falling wedge", 1, st, 0.45 if st == "forming" else 0.65, upper + width0, upper,
                    f"Both lines falling, narrowing; resistance ≈{upper:.0f}", start_idx)
            # flags / pennants: strong prior impulse + tight consolidation
            impulse = abs(prior) > 4 * a
            if impulse and width1 < 2.5 * a:
                bias = 1 if prior > 0 else -1
                name = ("Bullish" if bias > 0 else "Bearish") + (" pennant" if (slope_h < 0 < slope_l) else " flag")
                trig = upper if bias > 0 else lower
                st = "confirmed" if ((bias > 0 and px > upper) or (bias < 0 and px < lower)) else "forming"
                add(name, bias, st, 0.5 if st == "forming" else 0.7, trig + bias * abs(prior), trig,
                    f"Pole {prior:+.0f} pts then tight range", start_idx)
        else:
            # parallel-ish lines
            if abs(slope_h - slope_l) < 0.08 and width1 > 1.5 * a:
                if slope_h > 0.06:
                    add("Channel up", 1, "forming", 0.5, upper, lower,
                        f"Rising channel {lower:.0f}–{upper:.0f}", start_idx)
                elif slope_h < -0.06:
                    add("Channel down", -1, "forming", 0.5, lower, upper,
                        f"Falling channel {lower:.0f}–{upper:.0f}", start_idx)
                elif flat_h and flat_l:
                    pos = (px - lower) / (upper - lower) if upper > lower else 0.5
                    bias = 1 if pos > 0.8 else -1 if pos < 0.2 else 0
                    add("Rectangle", bias, "forming", 0.4, None, upper if pos > 0.5 else lower,
                        f"Range {lower:.0f}–{upper:.0f}, price at {pos * 100:.0f}% of range", start_idx)

    # ---- rounding bottom / cup (quadratic fit of lows over a longer window) ----
    win2 = 120 if tf == "intraday" else 90
    if n >= win2:
        seg = df.iloc[-win2:]
        y = seg["low"].values
        x = np.arange(win2)
        coef = np.polyfit(x, y, 2)
        resid = y - np.polyval(coef, x)
        fit_q = 1 - resid.var() / max(y.var(), 1e-9)
        if coef[0] > 0 and fit_q > 0.6:
            vertex = -coef[1] / (2 * coef[0])
            if 0.25 * win2 < vertex < 0.75 * win2:
                rim = max(y[0], seg["high"].iloc[:10].max())
                depth = rim - np.polyval(coef, vertex)
                if depth > 3 * a:
                    st = "confirmed" if px > rim else "forming"
                    add("Rounding bottom / cup", 1, st, 0.45 if st == "forming" else 0.65, rim + depth, rim,
                        f"Saucer base, rim ≈{rim:.0f}", n - win2)
        elif coef[0] < 0 and fit_q > 0.6:
            vertex = -coef[1] / (2 * coef[0])
            if 0.25 * win2 < vertex < 0.75 * win2:
                rim = min(y[0], seg["low"].iloc[:10].min())
                depth = np.polyval(coef, vertex) - rim
                if depth > 3 * a:
                    st = "confirmed" if px < rim else "forming"
                    add("Rounding top / inverse cup", -1, st, 0.45 if st == "forming" else 0.65, rim - depth, rim,
                        f"Dome top, rim ≈{rim:.0f}", n - win2)

    # drop patterns whose trigger is far from where price is now
    hits = [h for h in hits if h["trigger"] is None or abs(h["trigger"] - px) / px <= near_pct]
    # de-duplicate by name keeping highest confidence
    best = {}
    for h in hits:
        if h["name"] not in best or h["confidence"] > best[h["name"]]["confidence"]:
            best[h["name"]] = h
    out = sorted(best.values(), key=lambda h: (-(h["status"] == "confirmed"), -h["confidence"]))
    return out[:3]


def pattern_vote(hits: list[dict]) -> tuple[float, str]:
    if not hits:
        return 0.0, "No clean pattern"
    num = sum(h["bias"] * h["confidence"] * (1.3 if h["status"] == "confirmed" else 1.0) for h in hits)
    den = sum(h["confidence"] * (1.3 if h["status"] == "confirmed" else 1.0) for h in hits)
    v = num / den if den else 0.0
    names = ", ".join(f"{h['name']} ({h['status']})" for h in hits[:3])
    return float(max(-1, min(1, v))), names
