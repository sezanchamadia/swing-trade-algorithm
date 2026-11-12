#!/usr/bin/env python3
"""
v5 paper-trading bot for NSE stocks.
Runs once per weekday evening (GitHub Actions), downloads end-of-day prices from Yahoo Finance,
updates the paper account in state/paper_ledger.xlsx and writes docs/REPORT.md (+ charts).
Strategy: Oversold leader always; 20-day breakout, Supertrend (10,3) and EMA 10/30 only when
>= 70% of stocks are above their 50 EMA. Paper trading only. Not investment advice.
"""


import base64, io, json, math, os, sys, time, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ======================================================================================
# Settings (change here or with command-line flags)
# ======================================================================================
CONFIG = {
    "capital": 100_000,        # trading capital in rupees, used for position size
    "risk_pct": 1.0,           # % of capital you lose if the stop is hit
    "max_risk_pct": 8.0,       # skip trades whose stop is more than this % below entry
    "min_price": 30,           # ignore stocks below this price
    "min_turnover_cr": 5,      # ignore stocks trading less than this many crore/day (20-day avg)
    "min_score": 55,           # minimum technical score for a trade idea
    "top_n": 15,               # number of trades to list (and log) each day
    "entry_valid_days": 3,     # an unfilled entry order is cancelled after this many sessions
    "max_hold_days": 20,       # close the trade at market after this many sessions
    "history": "2y",           # download length for the daily scan
    "index": "^NSEI",          # Nifty 50, used for market filter and relative strength
}


# ======================================================================================
# Data
# ======================================================================================
def read_universe_file(path):
    syms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.split("#")[0].strip().upper()
            if s and s not in syms:
                syms.append(s)
    return syms


def _clean(df):
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df.columns = [str(c).capitalize() for c in df.columns]
    need = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in need):
        return None
    df = df[need].apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df["Close"] > 0) & (df["High"] >= df["Low"])]
    df["Volume"] = df["Volume"].fillna(0)
    return df if len(df) else None


def download(symbols, start, end, index_symbol):
    """Download daily OHLCV from Yahoo Finance. Returns ({symbol: df}, index df, failed list)."""
    import yfinance as yf
    tickers = {s: s + ".NS" for s in symbols}
    out, failed = {}, []
    keys = list(tickers)
    for b in range(0, len(keys), 40):
        chunk = keys[b:b + 40]
        print(f"  downloading {b + 1}-{b + len(chunk)} of {len(keys)} ...", flush=True)
        try:
            data = yf.download([tickers[s] for s in chunk], start=start, end=end, interval="1d",
                               group_by="ticker", auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print("   batch failed:", e); data = None
        for s in chunk:
            df = None
            if data is not None and len(data):
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        lv0 = data.columns.get_level_values(0)
                        df = data[tickers[s]] if tickers[s] in lv0 else data.xs(tickers[s], axis=1, level=1)
                    else:
                        df = data
                except Exception:
                    df = None
            df = _clean(df)
            if df is None or len(df) < 60:
                try:
                    df = _clean(yf.Ticker(tickers[s]).history(start=start, end=end, interval="1d", auto_adjust=True))
                except Exception:
                    df = None
            if df is None or len(df) < 60:
                failed.append(s)
            else:
                out[s] = df
        time.sleep(1.0)
    idx = None
    try:
        idx = _clean(yf.Ticker(index_symbol).history(start=start, end=end, interval="1d", auto_adjust=True))
    except Exception:
        pass
    return out, idx, failed


def quality_check(data, idx):
    """Drop stale or broken series. Returns (clean data, list of warnings)."""
    warns, clean = [], {}
    last_market_day = idx.index[-1] if idx is not None and len(idx) else max(d.index[-1] for d in data.values())
    for s, df in data.items():
        if df.index[-1] < last_market_day - pd.Timedelta(days=4):
            warns.append(f"{s}: last price is from {df.index[-1].date()}, older than the market ({last_market_day.date()}). Skipped.")
            continue
        r = df["Close"].pct_change().abs()
        big = r[r > 0.25]
        if len(big):
            warns.append(f"{s}: one-day move over 25% on {big.index[-1].date()} ({big.iloc[-1]*100:.0f}%). Check for a split/bonus or bad data.")
        if (df["Volume"].tail(20) == 0).sum() > 5:
            warns.append(f"{s}: volume missing on several recent days. Volume checks may be off.")
        clean[s] = df
    return clean, warns, last_market_day



# ======================================================================================
# Indicators
# ======================================================================================
def indicators(df):
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    for n in (10, 20, 50, 200):
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    d["rsi"] = d["rsi"].fillna(100)
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd"], d["macd_sig"] = m, m.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_s = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["pdi"] = 100 * pd.Series(pdm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_s
    d["mdi"] = 100 * pd.Series(mdm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_s
    dx = 100 * (d["pdi"] - d["mdi"]).abs() / (d["pdi"] + d["mdi"]).replace(0, np.nan)
    d["adx"] = dx.fillna(0).ewm(alpha=1 / 14, adjust=False).mean()
    d["vol20"] = v.rolling(20).mean()
    mid = c.rolling(20).mean(); sd = c.rolling(20).std()
    d["bb_up"], d["bb_width"] = mid + 2 * sd, (4 * sd) / mid
    d["turnover_cr"] = (c * v).rolling(20).mean() / 1e7
    return d


class Bars:
    """Fast numpy view of an indicator DataFrame."""
    def __init__(self, d):
        self.d = d
        for col in d.columns:
            setattr(self, col.lower(), d[col].to_numpy(dtype=float))
        self.n = len(d)
        self.dates = d.index


def swings(arr, lo, hi, k=3, kind="high", upto=None):
    """Indices of swing highs/lows in arr[lo:hi] (needs k bars on each side).
    upto = last bar that may be looked at (the scan day). Prevents look-ahead in backtests."""
    out = []
    last = len(arr) - 1 if upto is None else upto
    for j in range(max(lo, k), min(hi, last - k + 1)):
        w = arr[j - k:j + k + 1]
        if (kind == "high" and arr[j] == w.max()) or (kind == "low" and arr[j] == w.min()):
            if not out or j - out[-1] > k:
                out.append(j)
    return out


# ======================================================================================
# Technical score (0-100), same rules as the Swing Trade Scorecard page
# ======================================================================================
def tech_score(b, i):
    pts, mx, notes = 0, 0, []
    c = b.close[i]

    def add(p, m, ok, note):
        nonlocal pts, mx
        pts += p; mx += m
        notes.append(("✓" if ok else "✗") + " " + note)

    add(5 if c > b.ema20[i] else 0, 5, c > b.ema20[i], "above 20 EMA")
    add(5 if c > b.ema50[i] else 0, 5, c > b.ema50[i], "above 50 EMA")
    if i >= 200:
        add(5 if c > b.ema200[i] else 0, 5, c > b.ema200[i], "above 200 EMA")
    add(5 if b.ema20[i] > b.ema50[i] else 0, 5, b.ema20[i] > b.ema50[i], "20 EMA above 50 EMA")
    rising = b.ema50[i] > b.ema50[i - 10]
    add(5 if rising else 0, 5, rising, "50 EMA rising")
    r = b.rsi[i]
    rp = 0 if r > 80 else 3 if r > 75 else 8 if r > 68 else 15 if r >= 55 else 12 if r >= 50 else 6 if r >= 45 else 3 if r >= 35 else 2
    add(rp, 15, rp >= 12, f"RSI {r:.0f}")
    ms = b.macd[i] > b.macd_sig[i]
    add(6 if ms else 0, 6, ms, "MACD above signal")
    add(4 if b.macd[i] > 0 else 0, 4, b.macd[i] > 0, "MACD above zero")
    hr = b.macd_hist[i] > b.macd_hist[i - 1]
    add(3 if hr else 0, 3, hr, "MACD histogram rising")
    cross = any(b.macd_hist[j] > 0 >= b.macd_hist[j - 1] for j in range(i - 4, i + 1))
    add(2 if cross else 0, 2, cross, "MACD bullish cross in 5 days")
    bull = b.pdi[i] > b.mdi[i]
    add(7 if bull else 0, 7, bull, "+DI above -DI")
    a = b.adx[i]
    ap = (8 if a >= 25 else 5 if a >= 20 else 2) if bull else (2 if a < 20 else 0)
    add(ap, 8, ap >= 5, f"ADX {a:.0f}")
    if b.volume[i - 25:i + 1].sum() > 0:
        upv = sum(b.volume[j] for j in range(i - 19, i + 1) if b.close[j] > b.close[j - 1])
        dnv = sum(b.volume[j] for j in range(i - 19, i + 1) if b.close[j] < b.close[j - 1])
        udr = 3 if dnv == 0 else upv / dnv
        add(8 if udr >= 1.5 else 5 if udr >= 1.1 else 3 if udr >= 0.9 else 0, 8, udr >= 1.1, f"up/down volume {udr:.2f}")
        vr = b.volume[i] / b.vol20[i - 1] if b.vol20[i - 1] > 0 else 1
        upday = c > b.close[i - 1]
        lp = (7 if vr >= 1.5 else 4 if vr >= 1 else 2) if upday else (0 if vr >= 1.5 else 2)
        add(lp, 7, lp >= 4, f"volume {vr:.1f}x avg on {'up' if upday else 'down'} day")
    ext = (c / b.ema20[i] - 1) * 100
    xp = 8 if 0 <= ext <= 4 else 5 if 4 < ext <= 8 or -2 <= ext < 0 else 2 if 8 < ext <= 12 else 0
    add(xp, 8, xp >= 5, f"{ext:+.1f}% from 20 EMA")
    hh = b.high[i - 9:i + 1].max() > b.high[i - 19:i - 9].max()
    hl = b.low[i - 9:i + 1].min() > b.low[i - 19:i - 9].min()
    add(4 if hh else 0, 4, hh, "higher high")
    add(3 if hl else 0, 3, hl, "higher low")
    return round(pts / mx * 100), notes


# ======================================================================================
# Chart patterns. Each returns dict(name, status, pivot, base_low, height, note) or None.
# status: "breakout" (closed above the trigger today), "forming" (just below the trigger),
#         "pullback" (trend-continuation buy).
# ======================================================================================
def _vol_ok(b, i, mult=1.4):
    return b.vol20[i - 1] > 0 and b.volume[i] >= mult * b.vol20[i - 1]


def _status(b, i, pivot, near=0.03, vmult=1.4):
    c = b.close[i]
    if c > pivot and b.close[i - 1] <= pivot * 1.01 and _vol_ok(b, i, vmult):
        return "breakout"
    if pivot * (1 - near) <= c <= pivot:
        return "forming"
    return None


def p_52w_high(b, i):
    if i < 260:
        return None
    pivot = b.high[i - 252:i].max()
    if b.close[i] < b.ema50[i]:
        return None
    st = _status(b, i, pivot, near=0.03, vmult=1.5)
    if not st:
        return None
    base_low = b.low[i - 20:i].min()
    return dict(name="52-week high breakout", status=st, pivot=pivot, base_low=base_low,
                height=pivot - base_low, note=f"52-week high {pivot:.2f}")


def p_flat_base(b, i):
    for w in (40, 30, 20, 15):
        if i - w - 25 < 0:
            continue
        hi = b.high[i - w:i].max(); lo = b.low[i - w:i].min()
        if (hi - lo) / hi > 0.12:
            continue
        if not (b.ema50[i - w] > b.ema50[i - w - 20] and b.close[i - w] > b.ema50[i - w]):
            continue
        st = _status(b, i, hi, near=0.03)
        if st:
            return dict(name="Flat base breakout", status=st, pivot=hi, base_low=lo, height=hi - lo,
                        note=f"{w}-day base, range {(hi - lo) / hi * 100:.1f}%")
    return None


def p_vcp(b, i):
    lo0 = i - 70
    if lo0 < 200 or not (b.close[i] > b.ema50[i] > b.ema200[i]):
        return None
    sh = swings(b.high, lo0, i + 1, 3, "high", upto=i)
    sl = swings(b.low, lo0, i + 1, 3, "low", upto=i)
    depths = []
    for h_ in sh:
        nxt = [l_ for l_ in sl if l_ > h_]
        if nxt:
            l_ = nxt[0]
            depths.append((h_, l_, 1 - b.low[l_] / b.high[h_]))
    if len(depths) < 2:
        return None
    last = depths[-3:]
    if not all(last[k + 1][2] < last[k][2] for k in range(len(last) - 1)):
        return None
    if last[-1][2] > 0.10 or last[0][2] < 0.06:
        return None
    if b.volume[i - 10:i].mean() > b.volume[i - 50:i].mean():
        return None
    pivot = b.high[last[-1][0]:i].max()
    st = _status(b, i, pivot, near=0.05)
    if not st:
        return None
    base_low = b.low[last[-1][1]]
    return dict(name="Volatility contraction (VCP)", status=st, pivot=pivot, base_low=base_low,
                height=b.high[last[0][0]] - b.low[last[0][1]],
                note="pullbacks " + " → ".join(f"{d[2] * 100:.0f}%" for d in last))


def p_bull_flag(b, i):
    if i < 40:
        return None
    t = i - 15 + int(np.argmax(b.high[i - 15:i - 2]))
    s = t - 15 + int(np.argmin(b.low[t - 15:t - 2]))
    pole = b.high[t] - b.low[s]
    if b.high[t] / b.low[s] - 1 < 0.12:
        return None
    flag_lo = b.low[t + 1:i + 1].min()
    if flag_lo < b.high[t] - 0.5 * pole:
        return None
    if b.high[t + 1:i + 1].max() > b.high[t] * 1.01:
        return None
    if b.volume[t + 1:i + 1].mean() > b.volume[s:t + 1].mean():
        return None
    pivot = b.high[max(t + 1, i - 3):i].max() if i - 1 > t else b.high[t]
    st = "breakout" if b.close[i] > pivot and _vol_ok(b, i, 1.2) else ("forming" if b.close[i] >= pivot * 0.96 else None)
    if not st:
        return None
    return dict(name="Bull flag", status=st, pivot=pivot, base_low=flag_lo, height=pole,
                note=f"pole +{(b.high[t] / b.low[s] - 1) * 100:.0f}% in {t - s} days, flag {i - t} days")


def p_asc_triangle(b, i, w=50):
    if i < w + 10:
        return None
    lo0 = i - w
    res = b.high[lo0:i].max()
    sh = [j for j in swings(b.high, lo0, i, 3, "high", upto=i) if b.high[j] >= res * 0.98]
    if len(sh) < 2 or sh[-1] - sh[0] < 10:
        return None
    sl = swings(b.low, lo0, i, 3, "low", upto=i)
    if len(sl) < 2:
        return None
    x = np.array(sl, dtype=float); y = b.low[sl]
    slope = np.polyfit(x, y, 1)[0]
    if slope <= 0 or y[-1] < y[0] * 1.03:
        return None
    st = _status(b, i, res, near=0.04)
    if not st:
        return None
    return dict(name="Ascending triangle", status=st, pivot=res, base_low=y[-1], height=res - y[0],
                note=f"{len(sh)} touches of {res:.2f}, rising lows")


def p_cup_handle(b, i):
    if i < 160:
        return None
    lip = i - 150 + int(np.argmax(b.high[i - 150:i - 40]))
    bot = lip + int(np.argmin(b.low[lip:i - 5]))
    depth = 1 - b.low[bot] / b.high[lip]
    if not (0.12 <= depth <= 0.40) or bot - lip < 10:
        return None
    r = bot + int(np.argmax(b.high[bot:i - 2]))
    if r - bot < 10 or b.high[r] < b.high[lip] * 0.95:
        return None
    hl = i - r
    if not (3 <= hl <= 25):
        return None
    handle_lo = b.low[r + 1:i + 1].min()
    if handle_lo < b.high[r] * 0.88 or handle_lo < b.low[bot] + (b.high[lip] - b.low[bot]) / 2:
        return None
    pivot = b.high[r]
    st = _status(b, i, pivot, near=0.05)
    if not st:
        return None
    return dict(name="Cup and handle", status=st, pivot=pivot, base_low=handle_lo,
                height=b.high[lip] - b.low[bot], note=f"cup depth {depth * 100:.0f}%, handle {hl} days")


def p_double_bottom(b, i):
    if i < 70:
        return None
    sl = swings(b.low, i - 60, i, 3, "low", upto=i)
    for a_ in range(len(sl) - 1, 0, -1):
        l2 = sl[a_]
        if i - l2 > 25:
            break
        for l1 in sl[:a_]:
            if l2 - l1 < 10:
                continue
            if abs(b.low[l2] / b.low[l1] - 1) > 0.03:
                continue
            neck = b.high[l1:l2 + 1].max()
            bottom = min(b.low[l1], b.low[l2])
            if neck < bottom * 1.06:
                continue
            if b.close[i] < b.low[l2]:
                continue
            st = _status(b, i, neck, near=0.05)
            if st:
                return dict(name="Double bottom", status=st, pivot=neck, base_low=bottom,
                            height=neck - bottom, note=f"lows {b.low[l1]:.2f} / {b.low[l2]:.2f}")
    return None


def p_squeeze(b, i):
    if i < 140:
        return None
    w = b.bb_width
    if np.nanmin(w[i - 5:i]) > np.nanmin(w[i - 125:i]) * 1.1:
        return None
    if b.close[i] < b.ema50[i]:
        return None
    pivot = b.high[i - 20:i].max()
    st = _status(b, i, pivot, near=0.03, vmult=1.3)
    if not st:
        return None
    lo = b.low[i - 20:i].min()
    return dict(name="Bollinger squeeze breakout", status=st, pivot=pivot, base_low=lo, height=pivot - lo,
                note="volatility at a 6-month low")


def p_golden_cross(b, i):
    if i < 210:
        return None
    crossed = any(b.ema50[j] > b.ema200[j] and b.ema50[j - 1] <= b.ema200[j - 1] for j in range(i - 9, i + 1))
    if not crossed or b.close[i] < b.ema50[i]:
        return None
    return dict(name="Golden cross (50/200 EMA)", status="pullback", pivot=b.high[i],
                base_low=b.low[i - 4:i + 1].min(), height=0, note="50 EMA crossed above 200 EMA in the last 10 days")


def p_ema_pullback(b, i):
    if i < 210 or not (b.close[i] > b.ema50[i] > b.ema200[i] and b.ema50[i] > b.ema50[i - 10]):
        return None
    touched = None
    for e, nm in ((b.ema20, "20 EMA"), (b.ema50, "50 EMA")):
        if b.low[i - 2:i + 1].min() <= e[i] * 1.01 and b.close[i] > e[i]:
            touched = nm; break
    if not touched:
        return None
    if not (b.close[i] > b.open[i] and b.close[i] > b.close[i - 1]):
        return None
    if not (40 <= b.rsi[i] <= 62):
        return None
    return dict(name=f"Pullback to rising {touched}", status="pullback", pivot=b.high[i],
                base_low=b.low[i - 4:i + 1].min(), height=0, note="bounce candle off support in an uptrend")


PATTERNS = [p_52w_high, p_cup_handle, p_vcp, p_flat_base, p_asc_triangle, p_bull_flag,
            p_double_bottom, p_squeeze, p_ema_pullback, p_golden_cross]
PRIORITY = {f: k for k, f in enumerate(PATTERNS)}


def find_patterns(b, i):
    out = []
    for f in PATTERNS:
        try:
            p = f(b, i)
        except Exception:
            p = None
        if p:
            p["prio"] = PRIORITY[f]
            out.append(p)
    return out


# ======================================================================================
# Trade plan: entry / stop / targets
# ======================================================================================
def trade_plan(b, i, p, cfg):
    c, atr = b.close[i], b.atr[i]
    if p["status"] == "forming":
        entry = p["pivot"] + 0.1 * atr
        order = f"Buy stop at {entry:.2f} (limit {entry + 0.3 * atr:.2f}). Fills only if price breaks out."
    elif p["status"] == "breakout":
        if c - p["pivot"] <= 0.5 * atr:
            entry = max(p["pivot"], c - 0.25 * atr)
        else:
            entry = p["pivot"] + 0.25 * atr
        order = f"Limit buy at {entry:.2f} on a dip/retest. Don't chase the open."
    else:  # pullback
        entry = b.high[i] + 0.05 * atr
        order = f"Buy stop at {entry:.2f}, above today's high, to confirm the bounce."

    if p["status"] == "pullback":
        stop = p["base_low"] - 0.2 * atr
    else:
        stop = max(p["base_low"], p["pivot"] - 1.5 * atr) - 0.1 * atr
    if entry - stop < atr:
        stop = entry - atr
    risk = entry - stop
    risk_pct = risk / entry * 100
    if risk_pct > cfg["max_risk_pct"] or risk <= 0:
        return None
    t1 = entry + 2 * risk
    t2 = max(entry + 3 * risk, p["pivot"] + p["height"]) if p.get("height") else entry + 3 * risk
    qty = int((cfg["capital"] * cfg["risk_pct"] / 100) // risk)
    qty = min(qty, int(cfg["capital"] // entry))
    return dict(entry=round(entry, 2), stop=round(stop, 2), t1=round(t1, 2), t2=round(t2, 2),
                risk_pct=round(risk_pct, 2), qty=qty, order=order,
                order_type="stop" if p["status"] in ("forming", "pullback") else "limit")


# ======================================================================================
# Analysis of one stock at bar i
# ======================================================================================
def analyze(sym, b, i, cfg, idx_ret63=None, market_ok=True):
    if i < 120:
        return None
    score, notes = tech_score(b, i)
    rs = (b.close[i] / b.close[i - 63] - 1) * 100 - idx_ret63 if idx_ret63 is not None and i >= 63 else 0.0
    pats = find_patterns(b, i)
    row = dict(symbol=sym, date=b.dates[i], close=b.close[i], chg=(b.close[i] / b.close[i - 1] - 1) * 100,
               score=score, rsi=b.rsi[i], macd=b.macd[i], macd_sig=b.macd_sig[i], adx=b.adx[i],
               pdi=b.pdi[i], mdi=b.mdi[i], vol_ratio=b.volume[i] / b.vol20[i - 1] if b.vol20[i - 1] else np.nan,
               ext20=(b.close[i] / b.ema20[i] - 1) * 100, rs63=rs, turnover_cr=b.turnover_cr[i],
               patterns=pats, notes=notes, plan=None, pattern=None, trade_score=None)
    liquid = b.close[i] >= cfg["min_price"] and (np.isnan(b.turnover_cr[i]) or b.turnover_cr[i] >= cfg["min_turnover_cr"])
    if pats and liquid and score >= cfg["min_score"]:
        best = None
        for p in sorted(pats, key=lambda p: (p["status"] != "breakout", p["prio"])):
            plan = trade_plan(b, i, p, cfg)
            if plan:
                best = (p, plan); break
        if best:
            p, plan = best
            ts = score + (10 if p["status"] == "breakout" else 5) + (10 if rs > 10 else 5 if rs > 0 else 0)
            ts += 3 * (len(pats) - 1)
            ts -= 0 if market_ok else 10
            row.update(pattern=p, plan=plan, trade_score=min(int(round(ts)), 130))
    return row


def market_state(idx):
    if idx is None or len(idx) < 60:
        return True, "Nifty data unavailable; market filter off.", None
    d = indicators(idx)
    c, e50 = d["Close"].iloc[-1], d["ema50"].iloc[-1]
    rising = e50 > d["ema50"].iloc[-11]
    ok = c > e50 and rising
    txt = (f"Nifty 50 at {c:,.0f} is {'above' if c > e50 else 'below'} its 50 EMA ({e50:,.0f}), "
           f"which is {'rising' if rising else 'falling'}. "
           + ("Market supports long swing trades." if ok else "Market is weak: take fewer, smaller trades."))
    return ok, txt, d


# ======================================================================================
# Signal log: forward test of every suggested trade
# ======================================================================================
LOG_COLS = ["signal_date", "symbol", "pattern", "status", "order_type", "entry", "stop", "t1", "t2",
            "trade_score", "fill_date", "fill_price", "exit_date", "exit_price", "result", "r_multiple"]


def simulate(b, start, plan, cfg):
    """Walk forward from bar start (the session after the signal). Returns outcome dict."""
    entry, stop, t1 = plan["entry"], plan["stop"], plan["t1"]
    fill_i = fill = None
    for k in range(start, min(start + cfg["entry_valid_days"], b.n)):
        o, h, l = b.open[k], b.high[k], b.low[k]
        if plan["order_type"] == "stop" and h >= entry:
            fill_i, fill = k, max(entry, o); break
        if plan["order_type"] == "limit" and l <= entry:
            fill_i, fill = k, min(entry, o); break
    if fill_i is None:
        done = start + cfg["entry_valid_days"] <= b.n
        return dict(result="Expired" if done else "Pending")
    risk = fill - stop if fill > stop else plan["entry"] - stop
    mfe = 0.0
    for k in range(fill_i, min(fill_i + cfg["max_hold_days"], b.n)):
        o, h, l, c = b.open[k], b.high[k], b.low[k], b.close[k]
        mfe = max(mfe, h / fill - 1)
        if l <= stop:
            px = min(o, stop) if k > fill_i else stop
            return dict(result="Stop", fill_i=fill_i, fill=fill, exit_i=k, exit=px, r=(px - fill) / risk, mfe=mfe)
        if h >= t1:
            px = max(o, t1) if k > fill_i else t1
            return dict(result="Target 1", fill_i=fill_i, fill=fill, exit_i=k, exit=px, r=(px - fill) / risk, mfe=mfe)
    if fill_i + cfg["max_hold_days"] <= b.n:
        k = fill_i + cfg["max_hold_days"] - 1
        return dict(result="Time exit", fill_i=fill_i, fill=fill, exit_i=k, exit=b.close[k], r=(b.close[k] - fill) / risk, mfe=mfe)
    return dict(result="Open", fill_i=fill_i, fill=fill, r=(b.close[-1] - fill) / risk, mfe=mfe)


def update_log(path, bars_by_sym, new_rows, cfg):
    log = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=LOG_COLS)
    for c in LOG_COLS:
        if c not in log.columns:
            log[c] = np.nan
    for c in ("signal_date", "symbol", "pattern", "status", "order_type", "fill_date", "exit_date", "result"):
        log[c] = log[c].astype(object)
    for c in ("entry", "stop", "t1", "t2", "trade_score", "fill_price", "exit_price", "r_multiple"):
        log[c] = pd.to_numeric(log[c], errors="coerce").astype(float)
    # 1) update unfinished signals with new prices
    for k, r in log.iterrows():
        if r["result"] in ("Stop", "Target 1", "Time exit", "Expired"):
            continue
        b = bars_by_sym.get(r["symbol"])
        if b is None:
            continue
        sd = pd.Timestamp(r["signal_date"])
        start = int(np.searchsorted(b.dates.values, np.datetime64(sd), side="right"))
        out = simulate(b, start, dict(entry=r["entry"], stop=r["stop"], t1=r["t1"], order_type=r["order_type"]), cfg)
        log.at[k, "result"] = out["result"]
        if "fill_i" in out:
            log.at[k, "fill_date"] = str(b.dates[out["fill_i"]].date()); log.at[k, "fill_price"] = round(out["fill"], 2)
            log.at[k, "r_multiple"] = round(out["r"], 2)
        if "exit_i" in out:
            log.at[k, "exit_date"] = str(b.dates[out["exit_i"]].date()); log.at[k, "exit_price"] = round(out["exit"], 2)
    # 2) add today's trades (skip symbols with a live signal)
    live = set(log.loc[log["result"].isin(["Pending", "Open"]), "symbol"])
    today = [r for r in new_rows if r["symbol"] not in live]
    adds = []
    for r in today:
        if str(r["date"].date()) in set(log.loc[log["symbol"] == r["symbol"], "signal_date"].astype(str)):
            continue
        p, pl = r["pattern"], r["plan"]
        adds.append(dict(signal_date=str(r["date"].date()), symbol=r["symbol"], pattern=p["name"], status=p["status"],
                         order_type=pl["order_type"], entry=pl["entry"], stop=pl["stop"], t1=pl["t1"], t2=pl["t2"],
                         trade_score=r["trade_score"], result="Pending"))
    if adds:
        log = pd.concat([log, pd.DataFrame(adds)], ignore_index=True)
    log.to_csv(path, index=False)
    return log


def log_stats(log):
    done = log[log["result"].isin(["Stop", "Target 1", "Time exit"])].copy()
    if not len(done):
        return None
    done["r_multiple"] = pd.to_numeric(done["r_multiple"], errors="coerce")
    g = done.groupby("pattern").agg(trades=("result", "size"),
                                    win_rate=("result", lambda s: (s == "Target 1").mean() * 100),
                                    avg_r=("r_multiple", "mean")).reset_index()
    tot = dict(pattern="All patterns", trades=len(done), win_rate=(done["result"] == "Target 1").mean() * 100,
               avg_r=done["r_multiple"].mean())
    return pd.concat([g.sort_values("avg_r", ascending=False), pd.DataFrame([tot])], ignore_index=True)

# ======================================================================================
# Backtest
# ======================================================================================
def backtest(data, idx, cfg, test_start, progress_every=20):
    """Replays the scanner day by day from test_start using only data available on each day.
    One trade per stock at a time. Returns a DataFrame of trades."""
    idx_d = indicators(idx) if idx is not None else None
    trades = []
    t0 = time.time()
    for n_, (sym, df) in enumerate(data.items()):
        b = Bars(indicators(df))
        first = int(np.searchsorted(b.dates.values, np.datetime64(pd.Timestamp(test_start))))
        i = max(260, first)
        while i < b.n - 1:
            ok, r63 = True, None
            if idx_d is not None:
                j = idx_d.index.searchsorted(b.dates[i], side="right") - 1
                if j >= 63:
                    ok = bool(idx_d["Close"].iloc[j] > idx_d["ema50"].iloc[j] and idx_d["ema50"].iloc[j] > idx_d["ema50"].iloc[j - 10])
                    r63 = (idx_d["Close"].iloc[j] / idx_d["Close"].iloc[j - 63] - 1) * 100
            row = analyze(sym, b, i, cfg, r63, ok)
            if row and row["plan"]:
                out = simulate(b, i + 1, row["plan"], cfg)
                if out["result"] in ("Pending", "Open"):
                    break
                pl = row["plan"]
                trades.append(dict(signal_date=b.dates[i], symbol=sym, pattern=row["pattern"]["name"],
                                   status=row["pattern"]["status"], market_ok=ok, score=row["score"],
                                   trade_score=row["trade_score"], entry=pl["entry"], stop=pl["stop"], t1=pl["t1"],
                                   risk_pct=pl["risk_pct"], result=out["result"],
                                   fill_date=b.dates[out["fill_i"]] if "fill_i" in out else pd.NaT,
                                   fill=out.get("fill", np.nan),
                                   exit_date=b.dates[out["exit_i"]] if "exit_i" in out else pd.NaT,
                                   exit=out.get("exit", np.nan), r=out.get("r", np.nan),
                                   mfe_pct=out.get("mfe", np.nan) * 100 if "mfe" in out else np.nan,
                                   hold_days=(out["exit_i"] - out["fill_i"] + 1) if "exit_i" in out else np.nan))
                i = (out.get("exit_i") or i + cfg["entry_valid_days"]) + 1
                continue
            i += 1
        if (n_ + 1) % progress_every == 0 or n_ + 1 == len(data):
            print(f"  {n_ + 1}/{len(data)} stocks done  ({time.time() - t0:.0f}s, {len(trades)} signals so far)", flush=True)
    return pd.DataFrame(trades)


def backtest_summary(t):
    f = t[t["result"].isin(["Stop", "Target 1", "Time exit"])]
    if not len(f):
        return None
    rows = []
    for name, x in list(f.groupby("pattern")) + [("All patterns", f)]:
        sig = len(t) if name == "All patterns" else int((t["pattern"] == name).sum())
        rows.append(dict(pattern=name, signals=sig, filled=len(x), win_rate=(x["result"] == "Target 1").mean() * 100,
                         avg_r=x["r"].mean(), total_r=x["r"].sum(), rally10=(x["mfe_pct"] >= 10).mean() * 100))
    g = pd.DataFrame(rows).set_index("pattern")
    g = pd.concat([g.drop("All patterns").sort_values("avg_r", ascending=False), g.loc[["All patterns"]]])
    mk = f.groupby("market_ok")["r"].agg(["count", "mean"])
    return g, mk




# ======================================================================================
# Charts
# ======================================================================================
def chart_png(d, row, n=120):
    import matplotlib.pyplot as plt
    s = d.tail(n)
    x = np.arange(len(s))
    fig, (ax, av) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[3.2, 1], hspace=0.04))
    up = s["Close"] >= s["Open"]
    col = np.where(up, "#1a9a5b", "#d6455a")
    ax.vlines(x, s["Low"], s["High"], color=col, linewidth=0.8)
    ax.bar(x, (s["Close"] - s["Open"]).abs().clip(lower=s["Close"] * 0.0008), bottom=np.minimum(s["Open"], s["Close"]),
           color=col, width=0.65)
    for c_, nm, lc in (("ema20", "20 EMA", "#2f8fd8"), ("ema50", "50 EMA", "#e08a2c"), ("ema200", "200 EMA", "#8a63c7")):
        ax.plot(x, s[c_], color=lc, linewidth=1.1, label=nm)
    pl, p = row["plan"], row["pattern"]
    xr = [len(s) - 25, len(s) + 6]
    for val, lab, lc, ls in ((p["pivot"], "Pivot", "#666666", ":"), (pl["entry"], "Entry", "#1f5fbf", "-"),
                             (pl["stop"], "Stop", "#d6455a", "--"), (pl["t1"], "T1", "#1a9a5b", "--"), (pl["t2"], "T2", "#1a9a5b", ":")):
        if p["status"] == "pullback" and lab == "Pivot":
            continue
        ax.hlines(val, *xr, colors=lc, linestyles=ls, linewidth=1.2)
        if lab == "Pivot" and abs(val - pl["entry"]) / pl["entry"] < 0.02:
            continue  # label would sit on top of the entry label
        ax.text(xr[1] + 0.5, val, f"{lab} {val:,.2f}", va="center", fontsize=7.5, color=lc)
    ax.set_xlim(-1, len(s) + 16)
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=3)
    ax.grid(alpha=0.15); ax.tick_params(labelsize=7.5)
    ax.set_title(f"{row['symbol']}  ·  {p['name']}  ({p['status']})", fontsize=10, loc="left")
    av.bar(x, s["Volume"], color=col, width=0.65, alpha=0.6)
    av.plot(x, s["vol20"], color="#555555", linewidth=0.9)
    av.set_yticks([]); av.grid(alpha=0.15)
    ticks = x[:: max(1, len(x) // 6)]
    av.set_xticks(ticks); av.set_xticklabels([s.index[t].strftime("%d %b") for t in ticks], fontsize=7.5)
    for a in (ax, av):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ======================================================================================
# HTML report
# ======================================================================================
def fnum(v, d=2):
    return "–" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:,.{d}f}"


def rating(score):
    for m, l, k in ((75, "Strong Buy", "sb"), (60, "Buy", "b"), (45, "Hold", "h"), (30, "Sell", "s"), (-1, "Strong Sell", "ss")):
        if score >= m:
            return l, k


CSS = """
:root{--bg:#f4f5f7;--card:#fff;--ink:#16202e;--mut:#5d6878;--line:#dde1e7;--acc:#1f5fbf;--g:#11804a;--gb:#e2f4ea;--y:#8a6a00;--yb:#fbf3d6;--r:#b42335;--rb:#fbe4e7}
@media (prefers-color-scheme:dark){:root{--bg:#0f141b;--card:#171e28;--ink:#e6ebf2;--mut:#95a1b2;--line:#2a3442;--acc:#6ea4f2;--g:#4cc68a;--gb:#133523;--y:#e0c05a;--yb:#3a3112;--r:#f47b8b;--rb:#401a21}img{background:#fff;border-radius:6px}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14.5px/1.5 "Segoe UI",system-ui,sans-serif}
.w{max-width:1200px;margin:0 auto;padding:24px 18px 60px;display:grid;gap:18px}.w>*,.trade>*{min-width:0}
h1{margin:0;font-size:26px}h2{margin:0 0 10px;font-size:18px}h3{margin:0;font-size:16px}
.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.mono,td.n{font-family:Consolas,"SF Mono",monospace;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:13.5px}th{text-align:left;color:var(--mut);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;padding:6px 8px;border-bottom:1px solid var(--line);cursor:pointer;white-space:nowrap}
td{padding:7px 8px;border-bottom:1px solid var(--line);white-space:nowrap}td.n,th.n{text-align:right}
.tw{overflow-x:auto}.pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:600}
.sb,.b{background:var(--gb);color:var(--g)}.h{background:var(--yb);color:var(--y)}.s,.ss{background:var(--rb);color:var(--r)}
.st-breakout{background:var(--gb);color:var(--g)}.st-forming{background:var(--yb);color:var(--y)}.st-pullback{background:#e5eefb;color:#1f5fbf}
.banner{border-left:5px solid var(--g);}.banner.bad{border-left-color:var(--r)}
.trade{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:16px;align-items:start}
.trade img{width:100%;height:auto}
.lv{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0}.lv div{background:var(--bg);border-radius:8px;padding:7px 10px}
.lv .k{font-size:11.5px;color:var(--mut)}.lv .v{font:600 15px Consolas,monospace}
ul.why{margin:6px 0 0;padding-left:18px;font-size:13px}ul.why li{margin:2px 0}
.warn{font-size:12.5px;color:var(--mut)}@media(max-width:800px){.trade{grid-template-columns:1fr}}
"""

SORT_JS = """
document.querySelectorAll('table.sort').forEach(t=>{t.querySelectorAll('th').forEach((th,ci)=>{th.addEventListener('click',()=>{
const tb=t.tBodies[0],rows=[...tb.rows],asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
rows.sort((a,b)=>{const x=a.cells[ci].dataset.v??a.cells[ci].innerText,y=b.cells[ci].dataset.v??b.cells[ci].innerText;
const nx=parseFloat(x),ny=parseFloat(y);const r=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc?r:-r});rows.forEach(r=>tb.appendChild(r))})})});
"""


def build_report(rows, trades, watch, market_txt, market_ok, data_date, warns, failed, stats, bt, cfg, bars_by_sym):
    h = []
    h.append(f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Swing trades {data_date.date()}</title><style>{CSS}</style></head><body><div class='w'>")
    h.append(f"<header><h1>NSE Swing Trades for the session after {data_date.strftime('%d %b %Y')}</h1>"
             f"<p class='mut'>Based on closing prices of {data_date.strftime('%A %d %B %Y')} · {len(rows)} stocks scanned · "
             f"generated {datetime.now().strftime('%d %b %Y %H:%M')} · capital ₹{cfg['capital']:,.0f}, risk {cfg['risk_pct']}% per trade</p></header>")
    h.append(f"<div class='card banner {'bad' if not market_ok else ''}'><strong>Market:</strong> {market_txt}</div>")

    # Trades
    h.append(f"<div class='card'><h2>Trades to place ({len(trades)})</h2>"
             "<p class='mut' style='margin:0 0 10px'>Place these as orders before the open. Buy-stop orders fill only if the price trades up "
             f"to the entry; limit orders fill only on a dip to it. Cancel any order not filled within {cfg['entry_valid_days']} sessions. "
             f"Exit at the stop, at T1 (or trail to T2), or after {cfg['max_hold_days']} sessions.</p>")
    if trades:
        h.append("<div class='tw'><table class='sort'><thead><tr><th>#</th><th>Stock</th><th>Pattern</th><th>Status</th><th class='n'>Close</th>"
                 "<th class='n'>Entry</th><th>Order</th><th class='n'>Stop</th><th class='n'>T1 (2R)</th><th class='n'>T2</th><th class='n'>Risk %</th>"
                 "<th class='n'>Qty</th><th class='n'>Score</th></tr></thead><tbody>")
        for k, r in enumerate(trades, 1):
            p, pl = r["pattern"], r["plan"]
            h.append(f"<tr><td>{k}</td><td><a href='#t-{r['symbol']}'><strong>{r['symbol']}</strong></a></td><td>{p['name']}</td>"
                     f"<td><span class='pill st-{p['status']}'>{p['status'].capitalize()}</span></td><td class='n'>{fnum(r['close'])}</td>"
                     f"<td class='n'><strong>{fnum(pl['entry'])}</strong></td><td>{'Buy stop' if pl['order_type'] == 'stop' else 'Limit'}</td>"
                     f"<td class='n'>{fnum(pl['stop'])}</td><td class='n'>{fnum(pl['t1'])}</td><td class='n'>{fnum(pl['t2'])}</td>"
                     f"<td class='n'>{pl['risk_pct']:.1f}</td><td class='n'>{pl['qty']}</td><td class='n'>{r['trade_score']}</td></tr>")
        h.append("</tbody></table></div>")
    else:
        h.append("<p>No stock passed every filter today. Doing nothing is a valid trade.</p>")
    h.append("</div>")

    # Trade cards with charts
    for r in trades:
        p, pl = r["pattern"], r["plan"]
        img = chart_png(bars_by_sym[r["symbol"]].d, r)
        lab, key = rating(r["score"])
        others = [q["name"] for q in r["patterns"] if q is not p]
        h.append(f"<div class='card' id='t-{r['symbol']}'><div class='trade'><img alt='{r['symbol']} chart' src='data:image/png;base64,{img}'>"
                 f"<div><h3>{r['symbol']} <span class='pill {key}'>{lab} · {r['score']}</span></h3>"
                 f"<p class='mut' style='margin:4px 0'>{p['name']}: {p['note']}</p>"
                 f"<div class='lv'><div><div class='k'>Entry ({'buy stop' if pl['order_type'] == 'stop' else 'limit'})</div><div class='v'>{fnum(pl['entry'])}</div></div>"
                 f"<div><div class='k'>Stop loss</div><div class='v' style='color:var(--r)'>{fnum(pl['stop'])}</div></div>"
                 f"<div><div class='k'>Risk</div><div class='v'>{pl['risk_pct']:.1f}%</div></div>"
                 f"<div><div class='k'>Target 1 (2R)</div><div class='v' style='color:var(--g)'>{fnum(pl['t1'])}</div></div>"
                 f"<div><div class='k'>Target 2</div><div class='v' style='color:var(--g)'>{fnum(pl['t2'])}</div></div>"
                 f"<div><div class='k'>Quantity</div><div class='v'>{pl['qty']}</div></div></div>"
                 f"<p style='margin:0;font-size:13px'><strong>Order:</strong> {pl['order']}</p>"
                 f"<ul class='why'><li>RSI {r['rsi']:.0f} · MACD {'above' if r['macd'] > r['macd_sig'] else 'below'} signal · ADX {r['adx']:.0f} "
                 f"({'+DI' if r['pdi'] > r['mdi'] else '−DI'} leads)</li>"
                 f"<li>{r['rs63']:+.1f}% vs Nifty over 3 months · volume {fnum(r['vol_ratio'], 1)}× average</li>"
                 + (f"<li>Also showing: {', '.join(others)}</li>" if others else "")
                 + "".join(f"<li class='mut'>{n}</li>" for n in r["notes"] if n.startswith("✗"))
                 + "</ul></div></div></div>")

    # Watchlist
    if watch:
        h.append(f"<div class='card'><h2>Watchlist ({len(watch)})</h2><p class='mut' style='margin:0 0 8px'>Patterns detected, but the score, risk or rank "
                 "kept them off today's list. Check again tomorrow.</p><div class='tw'><table class='sort'><thead><tr><th>Stock</th><th>Pattern</th><th>Status</th>"
                 "<th class='n'>Close</th><th class='n'>Pivot</th><th class='n'>Score</th><th class='n'>RSI</th></tr></thead><tbody>")
        for r in watch:
            for p in r["patterns"][:1]:
                h.append(f"<tr><td>{r['symbol']}</td><td>{p['name']}</td><td><span class='pill st-{p['status']}'>{p['status'].capitalize()}</span></td>"
                         f"<td class='n'>{fnum(r['close'])}</td><td class='n'>{fnum(p['pivot'])}</td><td class='n'>{r['score']}</td><td class='n'>{r['rsi']:.0f}</td></tr>")
        h.append("</tbody></table></div></div>")

    # Full scorecard
    h.append("<div class='card'><h2>All stocks scorecard</h2><p class='mut' style='margin:0 0 8px'>Click a column to sort.</p><div class='tw'><table class='sort'><thead><tr>"
             "<th>Stock</th><th class='n'>Close</th><th class='n'>Chg %</th><th class='n'>Score</th><th>Rating</th><th class='n'>RSI</th><th>MACD</th>"
             "<th class='n'>ADX</th><th class='n'>Vol ×</th><th class='n'>vs 20 EMA %</th><th class='n'>RS vs Nifty %</th><th>Patterns</th></tr></thead><tbody>")
    for r in sorted(rows, key=lambda r: -r["score"]):
        lab, key = rating(r["score"])
        h.append(f"<tr><td>{r['symbol']}</td><td class='n'>{fnum(r['close'])}</td><td class='n'>{r['chg']:+.2f}</td><td class='n'>{r['score']}</td>"
                 f"<td data-v='{r['score']}'><span class='pill {key}'>{lab}</span></td><td class='n'>{r['rsi']:.1f}</td>"
                 f"<td>{'Bullish' if r['macd'] > r['macd_sig'] else 'Bearish'}</td><td class='n'>{r['adx']:.1f}</td><td class='n'>{fnum(r['vol_ratio'])}</td>"
                 f"<td class='n'>{r['ext20']:+.1f}</td><td class='n'>{r['rs63']:+.1f}</td><td>{', '.join(p['name'] for p in r['patterns']) or '–'}</td></tr>")
    h.append("</tbody></table></div></div>")

    # Track record
    h.append("<div class='card'><h2>Track record of past signals</h2>")
    if stats is not None:
        h.append("<p class='mut' style='margin:0 0 8px'>From signals_log.csv: every trade this scanner listed, checked against later prices. "
                 "Win = Target 1 hit before the stop. Avg R = average result in units of risk (1R = the stop distance).</p>"
                 "<table><thead><tr><th>Pattern</th><th class='n'>Closed trades</th><th class='n'>Win rate %</th><th class='n'>Avg R</th></tr></thead><tbody>")
        for _, s in stats.iterrows():
            h.append(f"<tr><td>{s['pattern']}</td><td class='n'>{int(s['trades'])}</td><td class='n'>{s['win_rate']:.0f}</td><td class='n'>{s['avg_r']:+.2f}</td></tr>")
        h.append("</tbody></table>")
    else:
        h.append("<p class='mut'>No closed trades yet. Results build up as you run the scanner daily.</p>")
    h.append("</div>")

    if bt is not None:
        g, mk = bt
        h.append("<div class='card'><h2>Backtest: how each pattern did historically</h2><p class='mut' style='margin:0 0 8px'>From the last --backtest run, "
                 "using the same entry, stop and target rules. Rally 10% = share of filled trades that rose 10%+ above entry within the holding period.</p>"
                 "<table><thead><tr><th>Pattern</th><th class='n'>Signals</th><th class='n'>Filled</th><th class='n'>Win rate %</th><th class='n'>Avg R</th>"
                 "<th class='n'>Rally 10% %</th></tr></thead><tbody>")
        for name, s in g.iterrows():
            h.append(f"<tr><td>{name}</td><td class='n'>{int(s['signals'])}</td><td class='n'>{int(s['filled'])}</td><td class='n'>{s['win_rate']:.0f}</td>"
                     f"<td class='n'>{s['avg_r']:+.2f}</td><td class='n'>{s['rally10']:.0f}</td></tr>")
        h.append("</tbody></table>")
        if len(mk) == 2:
            h.append(f"<p class='mut'>Avg R when Nifty was in an uptrend: {mk.loc[True, 'mean']:+.2f} ({int(mk.loc[True, 'count'])} trades); "
                     f"when it wasn't: {mk.loc[False, 'mean']:+.2f} ({int(mk.loc[False, 'count'])} trades).</p>")
        h.append("</div>")

    h.append("<div class='card warn'><strong>Data checks.</strong> Prices from Yahoo Finance (end of day). ")
    h.append(f"{len(failed)} symbol(s) could not be downloaded: {', '.join(failed) or 'none'}. ")
    h.append("<ul>" + "".join(f"<li>{w}</li>" for w in warns[:40]) + "</ul>" if warns else "No data problems found. ")
    h.append("Before placing an order, check the last close and the day's range against your broker or nseindia.com.<br><br>"
             "Not investment advice. Patterns raise the odds of a move; they don't guarantee one. Check results dates and news, and never risk more than you can afford to lose.</div>")
    h.append(f"</div><script>{SORT_JS}</script></body></html>")
    return "".join(h)



# ======================================================================================
# VERSION 2: market-wide context, new score, stricter entries, research backtest
# ======================================================================================
V2 = {
    "patterns": ["Pullback to rising 20 EMA", "Pullback to rising 50 EMA", "Flat base breakout",
                 "52-week high breakout", "Bollinger squeeze breakout", "Double bottom"],
    "confirm_breakout": True,   # breakouts need a daily CLOSE above the pivot (no intraday buy-stops)
    "stop_atr_min": 1.5,        # stop at least this many ATRs below entry
    "stop_atr_max": 3.0,        # and no more than this many
    "max_risk_pct": 8.0,
    "target_r": 2.0,
    "trail_max_days": 40,
}
# Filters used by the v2 strategy (applied to the research candidates; change and re-run the portfolio cell)
V2_RULES = {
    "patterns": ["Pullback to rising 20 EMA", "Pullback to rising 50 EMA", "Flat base breakout", "52-week high breakout"],
    "min_score": 60,            # v2 score (0-100)
    "min_rs_pct": 0.60,         # 6-month return must beat 60% of the universe
    "max_rsi": 70,
    "max_ext20": 6.0,           # % above 20 EMA
    "regime": "breadth",        # "breadth" | "nifty200" | "v1" | "none"
    "min_breadth": 0.50,        # share of universe above its 50 EMA
    "exit": "trail",            # "target" (fixed 2R) or "trail" (breakeven at +1R, exit on close below 20 EMA)
}


def market_context(data, idx):
    """Cross-sectional data per date: relative-strength percentiles and breadth. Uses only past data."""
    closes = pd.DataFrame({s: df["Close"] for s, df in data.items()}).sort_index()
    r126 = pd.DataFrame({s: df["Close"].pct_change(126) for s, df in data.items()}).reindex(closes.index)
    r63 = pd.DataFrame({s: df["Close"].pct_change(63) for s, df in data.items()}).reindex(closes.index)
    above50 = pd.DataFrame({s: (df["Close"] > df["Close"].ewm(span=50, adjust=False).mean()).astype(float)
                                .where(np.arange(len(df)) >= 50) for s, df in data.items()}).reindex(closes.index)
    ctx = dict(rs126=r126.rank(axis=1, pct=True), rs63=r63.rank(axis=1, pct=True),
               breadth=above50.mean(axis=1, skipna=True))
    if idx is not None and len(idx) > 210:
        di = indicators(idx)
        n200 = (di["Close"] > di["ema200"]).reindex(closes.index, method="ffill")
        v1ok = ((di["Close"] > di["ema50"]) & (di["ema50"] > di["ema50"].shift(10))).reindex(closes.index, method="ffill")
        ctx["nifty200"], ctx["nifty_v1"] = n200.fillna(True), v1ok.fillna(True)
    else:
        ctx["nifty200"] = ctx["nifty_v1"] = pd.Series(True, index=closes.index)
    return ctx


def attach_context(b, sym, ctx):
    d = b.dates
    b.rs126 = ctx["rs126"][sym].reindex(d).to_numpy(dtype=float)
    b.rs63 = ctx["rs63"][sym].reindex(d).to_numpy(dtype=float)
    b.breadth = ctx["breadth"].reindex(d).to_numpy(dtype=float)
    b.nifty200 = ctx["nifty200"].reindex(d).fillna(True).to_numpy(dtype=bool)
    b.nifty_v1 = ctx["nifty_v1"].reindex(d).fillna(True).to_numpy(dtype=bool)
    b.atr10 = pd.Series(b.atr).rolling(10).mean().to_numpy()
    b.atr50 = pd.Series(b.atr).rolling(50).mean().to_numpy()
    return b


def features(b, i):
    c = b.close[i]
    up = sum(b.volume[j] for j in range(i - 19, i + 1) if b.close[j] > b.close[j - 1])
    dn = sum(b.volume[j] for j in range(i - 19, i + 1) if b.close[j] < b.close[j - 1])
    hi252 = b.high[max(0, i - 251):i + 1].max()
    return dict(
        rsi=b.rsi[i], ext20=(c / b.ema20[i] - 1) * 100, adx=b.adx[i], di_bull=float(b.pdi[i] > b.mdi[i]),
        macd_bull=float(b.macd[i] > b.macd_sig[i]), vol_ratio=b.volume[i] / b.vol20[i - 1] if b.vol20[i - 1] else np.nan,
        updown_vol=3.0 if dn == 0 else up / dn, atr_pct=b.atr[i] / c * 100,
        atr_contract=b.atr10[i] / b.atr50[i] if b.atr50[i] else np.nan,
        rs_pct126=b.rs126[i], rs_pct63=b.rs63[i],
        ret126=(c / b.close[i - 126] - 1) * 100 if i >= 126 else np.nan,
        ema50_slope=(b.ema50[i] / b.ema50[i - 10] - 1) * 100, ema200_slope=(b.ema200[i] / b.ema200[i - 20] - 1) * 100,
        above_ema200=float(c > b.ema200[i]), ema50_over_200=float(b.ema50[i] > b.ema200[i]),
        dist_52wh=(c / hi252 - 1) * 100, breadth=b.breadth[i], nifty200=float(b.nifty200[i]), nifty_v1=float(b.nifty_v1[i]),
        turnover_cr=b.turnover_cr[i])


def score_v2(fe):
    """0-100. Weights: relative strength 35, trend 25, entry quality 25, volume 15."""
    p = 0.0
    rs = fe["rs_pct126"] if np.isfinite(fe["rs_pct126"]) else 0.5
    rs3 = fe["rs_pct63"] if np.isfinite(fe["rs_pct63"]) else 0.5
    p += 25 * rs + 10 * rs3
    p += 5 * fe["above_ema200"] + 5 * fe["ema50_over_200"] + 5 * (fe["ema50_slope"] > 0) + 5 * (fe["ema200_slope"] > 0) \
        + 5 * (fe["dist_52wh"] > -25)
    e = fe["ext20"]
    p += 10 if -1 <= e <= 5 else 5 if 5 < e <= 8 or -3 <= e < -1 else 0
    r = fe["rsi"]
    p += 10 if 45 <= r <= 65 else 5 if 65 < r <= 70 or 40 <= r < 45 else 0
    p += 5 if np.isfinite(fe["atr_contract"]) and fe["atr_contract"] < 0.9 else 0
    u = fe["updown_vol"]
    p += 10 if u >= 1.3 else 5 if u >= 1.0 else 0
    p += 5 if np.isfinite(fe["vol_ratio"]) and fe["vol_ratio"] >= 1.2 else 0
    return round(p, 1)


def plan_v2(b, i, p):
    c, atr = b.close[i], b.atr[i]
    if p["status"] == "pullback":
        entry = b.high[i] + 0.05 * atr
        order_type = "stop"
        stop = p["base_low"] - 0.2 * atr
        order = f"Buy stop at {entry:.2f}, above today's high, to confirm the bounce."
    else:  # confirmed breakout -> limit near the pivot
        entry = max(p["pivot"], c - 0.25 * atr) if c - p["pivot"] <= 0.5 * atr else p["pivot"] + 0.25 * atr
        order_type = "limit"
        stop = max(p["base_low"], p["pivot"] - 2.0 * atr) - 0.1 * atr
        order = f"Limit buy at {entry:.2f} on a dip/retest after a confirmed close above {p['pivot']:.2f}."
    stop = max(stop, entry - V2["stop_atr_max"] * atr)
    stop = min(stop, entry - V2["stop_atr_min"] * atr)
    risk = entry - stop
    if risk <= 0 or risk / entry * 100 > V2["max_risk_pct"]:
        return None
    return dict(entry=round(entry, 2), stop=round(stop, 2), t1=round(entry + V2["target_r"] * risk, 2),
                t2=round(entry + 3 * risk, 2), risk_pct=round(risk / entry * 100, 2), order_type=order_type, order=order)


def candidates_at(b, i, sym):
    """All v2 setups at bar i (no score/regime filters), each with features and a plan."""
    out = []
    pats = find_patterns(b, i)
    if not pats:
        return out
    fe = None
    for p in pats:
        if p["name"] not in V2["patterns"]:
            continue
        if p["status"] == "forming" and V2["confirm_breakout"]:
            continue
        pl = plan_v2(b, i, p)
        if not pl:
            continue
        if fe is None:
            fe = features(b, i)
            fe["score_v1"] = tech_score(b, i)[0]
            fe["score_v2"] = score_v2(fe)
        out.append(dict(symbol=sym, i=i, signal_date=b.dates[i], pattern=p["name"], status=p["status"],
                        pivot=p["pivot"], note=p["note"], close=b.close[i], **pl, **fe))
    return out


def simulate_v2(b, start, plan, mode, cfg):
    entry, stop0 = plan["entry"], plan["stop"]
    fill_i = fill = None
    for k in range(start, min(start + cfg["entry_valid_days"], b.n)):
        o, h, l = b.open[k], b.high[k], b.low[k]
        if plan["order_type"] == "stop" and h >= entry:
            fill_i, fill = k, max(entry, o); break
        if plan["order_type"] == "limit" and l <= entry:
            fill_i, fill = k, min(entry, o); break
    if fill_i is None:
        return dict(result="Expired" if start + cfg["entry_valid_days"] <= b.n else "Pending")
    risk = fill - stop0 if fill > stop0 else entry - stop0
    stop, mfe = stop0, 0.0
    t1 = fill + V2["target_r"] * risk
    max_days = cfg["max_hold_days"] if mode == "target" else V2["trail_max_days"]
    for k in range(fill_i, min(fill_i + max_days, b.n)):
        o, h, l, c = b.open[k], b.high[k], b.low[k], b.close[k]
        if l <= stop:
            px = min(o, stop) if k > fill_i else stop
            res = "Stop" if stop < fill else "Breakeven/trail stop"
            return dict(result=res, fill_i=fill_i, fill=fill, exit_i=k, exit=px, r=(px - fill) / risk, mfe=max(mfe, h / fill - 1))
        mfe = max(mfe, h / fill - 1)
        if mode == "target":
            if h >= t1:
                px = max(o, t1) if k > fill_i else t1
                return dict(result="Target 1", fill_i=fill_i, fill=fill, exit_i=k, exit=px, r=(px - fill) / risk, mfe=mfe)
        else:
            if h >= fill + risk:
                stop = max(stop, fill)              # breakeven from the next bar
            if k > fill_i and c < b.ema20[k] and c > fill:
                return dict(result="Trail exit", fill_i=fill_i, fill=fill, exit_i=k, exit=c, r=(c - fill) / risk, mfe=mfe)
    if fill_i + max_days <= b.n:
        k = fill_i + max_days - 1
        return dict(result="Time exit", fill_i=fill_i, fill=fill, exit_i=k, exit=b.close[k], r=(b.close[k] - fill) / risk, mfe=mfe)
    return dict(result="Open", fill_i=fill_i, fill=fill, r=(b.close[-1] - fill) / risk, mfe=mfe)


def research_backtest(data, ctx, cfg, test_start, progress_every=20):
    """Every v2 setup on every day of the test window, simulated with both exit styles.
    Filters are applied later in pandas, so you can try different rules without re-running this."""
    rows, t0 = [], time.time()
    for n_, (sym, df) in enumerate(data.items()):
        b = attach_context(Bars(indicators(df)), sym, ctx)
        first = int(np.searchsorted(b.dates.values, np.datetime64(pd.Timestamp(test_start))))
        for i in range(max(260, first), b.n - 1):
            for c in candidates_at(b, i, sym):
                for mode in ("target", "trail"):
                    o = simulate_v2(b, i + 1, c, mode, cfg)
                    c[f"{mode}_result"] = o["result"]
                    c[f"{mode}_r"] = o.get("r", np.nan)
                    c[f"{mode}_mfe_pct"] = o.get("mfe", np.nan) * 100 if "mfe" in o else np.nan
                    c[f"{mode}_fill_date"] = b.dates[o["fill_i"]] if "fill_i" in o else pd.NaT
                    c[f"{mode}_fill"] = o.get("fill", np.nan)
                    c[f"{mode}_exit_date"] = b.dates[o["exit_i"]] if "exit_i" in o else pd.NaT
                    c[f"{mode}_exit"] = o.get("exit", np.nan)
                    c[f"{mode}_days"] = (o["exit_i"] - o["fill_i"] + 1) if "exit_i" in o else np.nan
                rows.append(c)
        if (n_ + 1) % progress_every == 0 or n_ + 1 == len(data):
            print(f"  {n_ + 1}/{len(data)} stocks  ({time.time() - t0:.0f}s, {len(rows)} setups)", flush=True)
    df = pd.DataFrame(rows)
    if len(df):
        df["regime_ok_breadth"] = df["breadth"] >= V2_RULES["min_breadth"]
    return df


def apply_rules(cand, rules):
    """Filter research candidates with a rule set; returns one row per trade-able signal."""
    c = cand[cand["pattern"].isin(rules["patterns"])]
    c = c[(c["score_v2"] >= rules["min_score"]) & (c["rs_pct126"].fillna(0) >= rules["min_rs_pct"]) &
          (c["rsi"] <= rules["max_rsi"]) & (c["ext20"] <= rules["max_ext20"])]
    if rules["regime"] == "breadth":
        c = c[c["breadth"] >= rules["min_breadth"]]
    elif rules["regime"] == "nifty200":
        c = c[c["nifty200"] == 1]
    elif rules["regime"] == "v1":
        c = c[c["nifty_v1"] == 1]
    # one setup per stock per day: keep the highest-priority pattern
    order = {n: k for k, n in enumerate(rules["patterns"])}
    c = c.assign(_o=c["pattern"].map(order)).sort_values(["symbol", "signal_date", "_o"]).drop_duplicates(["symbol", "signal_date"])
    return c.drop(columns="_o")


def to_trades(sig, mode, score_col="score_v2"):
    """Convert research rows to the trade format used by portfolio_sim / Excel export."""
    m = mode
    t = pd.DataFrame(dict(signal_date=sig["signal_date"], symbol=sig["symbol"], pattern=sig["pattern"], status=sig["status"],
                          market_ok=sig["breadth"] >= V2_RULES["min_breadth"], score=sig[score_col], trade_score=sig[score_col],
                          entry=sig["entry"], stop=sig["stop"], t1=sig["t1"], risk_pct=sig["risk_pct"],
                          result=sig[f"{m}_result"], fill_date=sig[f"{m}_fill_date"], fill=sig[f"{m}_fill"],
                          exit_date=sig[f"{m}_exit_date"], exit=sig[f"{m}_exit"], r=sig[f"{m}_r"],
                          mfe_pct=sig[f"{m}_mfe_pct"], hold_days=sig[f"{m}_days"]))
    return t.reset_index(drop=True)


CLOSED = ["Stop", "Target 1", "Time exit", "Trail exit", "Breakeven/trail stop"]


def portfolio_sim(t, capital, risk_pct, max_pos, cost_pct, max_alloc_pct=25):
    f = t[t["result"].isin(CLOSED)].copy()
    if not len(f):
        return pd.DataFrame(), pd.Series(dtype=float)
    f["fill_date"] = pd.to_datetime(f["fill_date"]); f["exit_date"] = pd.to_datetime(f["exit_date"])
    f = f.sort_values(["fill_date", "trade_score"], ascending=[True, False])
    equity, open_pos, taken, curve = capital, [], [], [(f["fill_date"].min(), capital)]

    def realize(upto):
        nonlocal equity, open_pos
        for p in sorted([p for p in open_pos if p["exit_date"] < upto], key=lambda p: p["exit_date"]):
            equity += p["pnl"]; curve.append((p["exit_date"], equity))
        open_pos = [p for p in open_pos if p["exit_date"] >= upto]

    for _, r in f.iterrows():
        realize(r["fill_date"])
        if len(open_pos) >= max_pos or any(p["symbol"] == r["symbol"] for p in open_pos):
            continue
        per_share = r["fill"] - r["stop"]
        if per_share <= 0:
            continue
        qty = math.floor(min(equity * risk_pct / 100 / per_share, equity * max_alloc_pct / 100 / r["fill"]))
        if qty < 1:
            continue
        pnl = qty * (r["exit"] - r["fill"]) - cost_pct / 100 * qty * r["fill"]
        open_pos.append(dict(symbol=r["symbol"], exit_date=r["exit_date"], pnl=pnl))
        taken.append(dict(r, qty=qty, pnl=pnl, r_after_cost=pnl / (qty * per_share)))
    realize(pd.Timestamp.max)
    c = pd.DataFrame(curve, columns=["date", "equity"]).drop_duplicates("date", keep="last").set_index("date")["equity"]
    return pd.DataFrame(taken), c


def perf(taken, curve, capital, start=None, end=None):
    """Stats for trades closed between start and end (equity rebased to the window)."""
    if taken is None or not len(taken):
        return dict(trades=0)
    t = taken.copy(); t["exit_date"] = pd.to_datetime(t["exit_date"])
    if start is not None:
        t = t[t["exit_date"] >= pd.Timestamp(start)]
    if end is not None:
        t = t[t["exit_date"] <= pd.Timestamp(end)]
    if not len(t):
        return dict(trades=0)
    c = curve.copy()
    if start is not None:
        c = c[c.index >= pd.Timestamp(start)]
    if end is not None:
        c = c[c.index <= pd.Timestamp(end)]
    base = curve[curve.index < pd.Timestamp(start)].iloc[-1] if start is not None and (curve.index < pd.Timestamp(start)).any() else capital
    eq = pd.concat([pd.Series([base], index=[c.index[0] - pd.Timedelta(days=1)]), c]) if len(c) else pd.Series([base])
    yrs = max((t["exit_date"].max() - t["exit_date"].min()).days / 365.25, 0.25)
    ret = eq.iloc[-1] / base - 1
    wins = t["result"].isin(["Target 1", "Trail exit"]) & (t["r_after_cost"] > 0)
    return dict(trades=len(t), win_rate=round(wins.mean() * 100, 1), avg_r=round(t["r_after_cost"].mean(), 3),
                total_return_pct=round(ret * 100, 1), cagr_pct=round(((1 + ret) ** (1 / yrs) - 1) * 100, 1),
                max_dd_pct=round(((eq / eq.cummax()) - 1).min() * 100, 1), avg_days=round(t["hold_days"].mean(), 1))


def quintile_table(df, col, rcol, split_date, q=5):
    d = df[[col, rcol, "signal_date"]].dropna()
    if d[col].nunique() < q:
        bins = d[col]
    else:
        bins = pd.qcut(d[col], q, duplicates="drop")
    d = d.assign(bin=bins, period=np.where(d["signal_date"] <= pd.Timestamp(split_date), "in-sample", "out-of-sample"))
    t = d.pivot_table(index="bin", columns="period", values=rcol, aggfunc=["mean", "count"], observed=True)
    t.columns = [f"{a} {b}".replace("mean", "avg R").replace("count", "n") for a, b in t.columns]
    return t.round(3)


def chart_trade(b, sig, mode, n_before=80, n_after=45):
    """Candlestick chart of a historical trade with entry, stop, target and exit marked."""
    import matplotlib.pyplot as plt
    i = int(sig["i"])
    lo, hi = max(0, i - n_before), min(b.n, i + n_after)
    s = b.d.iloc[lo:hi]
    x = np.arange(len(s))
    up = s["Close"] >= s["Open"]; col = np.where(up, "#1a9a5b", "#d6455a")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.vlines(x, s["Low"], s["High"], color=col, linewidth=0.7)
    ax.bar(x, (s["Close"] - s["Open"]).abs().clip(lower=s["Close"] * 0.0008), bottom=np.minimum(s["Open"], s["Close"]), color=col, width=0.6)
    ax.plot(x, s["ema20"], color="#2f8fd8", lw=1, label="20 EMA"); ax.plot(x, s["ema50"], color="#e08a2c", lw=1, label="50 EMA")
    si = i - lo
    ax.axvline(si, color="#999999", ls=":", lw=1)
    for v, lab, c_ in ((sig["entry"], "entry", "#1f5fbf"), (sig["stop"], "stop", "#d6455a"), (sig["t1"], "2R", "#1a9a5b")):
        ax.hlines(v, si, len(s) - 1, colors=c_, linestyles="--", lw=0.9)
    ed = sig.get(f"{mode}_exit_date")
    if pd.notna(ed):
        xi = int(np.searchsorted(s.index.values, np.datetime64(pd.Timestamp(ed))))
        ax.scatter([xi], [sig[f"{mode}_exit"]], marker="v", color="black", zorder=5, s=30)
    fd = sig.get(f"{mode}_fill_date")
    if pd.notna(fd):
        xf = int(np.searchsorted(s.index.values, np.datetime64(pd.Timestamp(fd))))
        ax.scatter([xf], [sig[f"{mode}_fill"]], marker="^", color="#1f5fbf", zorder=5, s=30)
    ax.set_title(f"{sig['symbol']} · {sig['pattern']} · {pd.Timestamp(sig['signal_date']):%d %b %Y} · "
                 f"{sig[f'{mode}_result']} {sig[f'{mode}_r']:+.1f}R", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.grid(alpha=.15); ax.tick_params(labelsize=7)
    t_ = x[::max(1, len(x) // 6)]; ax.set_xticks(t_); ax.set_xticklabels([s.index[k].strftime("%d %b %y") for k in t_])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig


# ======================================================================================
# VERSION 3: strategy library. Each strategy is tested on its own.
# Signals use the CLOSE of day i; orders execute from day i+1. One trade per stock per strategy at a time.
# ======================================================================================
def adjust_corporate_actions(data, gap_down=0.60, gap_up=1.70):
    """Back-adjust unadjusted demergers/splits/bonuses: an overnight open below 60% (or above 170%) of the prior close
    is treated as a corporate action, and all earlier prices are scaled by open/prev_close. Returns (data, events)."""
    events = []
    for s, df in data.items():
        o, c = df["Open"].to_numpy(), df["Close"].to_numpy()
        ratio = o[1:] / c[:-1]
        hits = np.where((ratio < gap_down) | (ratio > gap_up))[0]
        if len(hits):
            df = df.copy()
            for h in hits[::-1]:
                k = h + 1; f = ratio[h]
                for col in ("Open", "High", "Low", "Close"):
                    df.iloc[:k, df.columns.get_loc(col)] *= f
                df.iloc[:k, df.columns.get_loc("Volume")] /= f
                events.append(f"{s}: {df.index[k].date()} overnight change {(f - 1) * 100:+.0f}%, treated as a corporate action and back-adjusted")
            data[s] = df
    return data, events


def extra_indicators(b):
    c, h, l, o = b.close, b.high, b.low, b.open
    s = pd.Series(c)
    b.sma5 = s.rolling(5).mean().to_numpy()
    b.sma200 = s.rolling(200).mean().to_numpy()
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean(); ls = (-d.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    b.rsi2 = (100 - 100 / (1 + g / ls.replace(0, np.nan))).fillna(100).to_numpy()
    mid = s.rolling(20).mean(); sd = s.rolling(20).std()
    b.bb_mid, b.bb_low = mid.to_numpy(), (mid - 2 * sd).to_numpy()
    b.hi20 = pd.Series(h).shift(1).rolling(20).max().to_numpy()     # prior 20-day high (excl. today)
    b.lo10 = pd.Series(l).rolling(10).min().to_numpy()               # incl. today
    b.hi252 = pd.Series(h).shift(1).rolling(252, min_periods=200).max().to_numpy()
    rng = pd.Series(h - l)
    b.nr7 = (rng == rng.rolling(7).min()).to_numpy()
    b.ema10 = s.ewm(span=10, adjust=False).mean().to_numpy()
    b.ema30 = s.ewm(span=30, adjust=False).mean().to_numpy()
    # Supertrend (10, 3)
    atr10 = pd.Series(np.maximum.reduce([h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))])).ewm(alpha=1 / 10, adjust=False).mean().to_numpy()
    hl2 = (h + l) / 2
    ub, lb = hl2 + 3 * atr10, hl2 - 3 * atr10
    st = np.zeros(len(c)); dirn = np.ones(len(c))
    fub, flb = ub.copy(), lb.copy()
    for k in range(1, len(c)):
        fub[k] = ub[k] if (ub[k] < fub[k - 1] or c[k - 1] > fub[k - 1]) else fub[k - 1]
        flb[k] = lb[k] if (lb[k] > flb[k - 1] or c[k - 1] < flb[k - 1]) else flb[k - 1]
        if dirn[k - 1] == 1:
            dirn[k] = -1 if c[k] < flb[k] else 1
        else:
            dirn[k] = 1 if c[k] > fub[k] else -1
        st[k] = flb[k] if dirn[k] == 1 else fub[k]
    b.st_dir, b.st_line = dirn, st
    return b


def _bound_stop(entry, stop, atr, lo=1.5, hi=3.0):
    stop = max(stop, entry - hi * atr)
    return min(stop, entry - lo * atr)


# ---- strategy definitions: signal(b, i) -> dict(entry_type, level, stop_fn, exit, target_r, breakeven, max_days) or None
def s_pullback(which):
    def f(b, i):
        p = p_ema_pullback(b, i)
        if not p or which not in p["name"]:
            return None
        e = b.high[i] + 0.05 * b.atr[i]
        return dict(entry="stop", level=e, stop=_bound_stop(e, p["base_low"] - 0.2 * b.atr[i], b.atr[i]),
                    exit="trail_ema20", breakeven=True, max_days=40)
    return f


def s_rsi2(b, i):
    if b.close[i] > b.sma200[i] and b.rsi2[i] < 10:
        return dict(entry="open", stop_atr=3.0, exit="close_above_sma5", max_days=10)


def s_bb_low(b, i):
    if b.close[i] > b.sma200[i] and b.close[i] < b.bb_low[i]:
        return dict(entry="open", stop_atr=3.0, exit="close_above_bbmid", max_days=15)


def s_3down(b, i):
    if b.close[i] > b.sma200[i] and b.close[i] < b.close[i - 1] < b.close[i - 2] < b.close[i - 3]:
        return dict(entry="open", stop_atr=3.0, exit="close_above_prev_high", max_days=10)


def s_donchian(b, i):
    if b.close[i] > b.hi20[i] and b.close[i] > b.ema200[i] and b.close[i - 1] <= b.hi20[i - 1]:
        return dict(entry="open", stop_low10=True, exit="close_below_lo10", max_days=60)


def s_52wh(b, i):
    if np.isfinite(b.hi252[i]) and b.close[i] > b.hi252[i] and b.close[i - 1] <= b.hi252[i - 1] and \
            b.vol20[i - 1] > 0 and b.volume[i] >= 1.5 * b.vol20[i - 1]:
        return dict(entry="open", stop_atr=2.5, exit="close_below_ema20", max_days=60)


def s_nr7(b, i):
    if b.nr7[i] and b.close[i] > b.ema50[i] > b.ema200[i]:
        e = b.high[i] + 0.05 * b.atr[i]
        return dict(entry="stop", level=e, stop=_bound_stop(e, b.low[i] - 0.1 * b.atr[i], b.atr[i], 1.0, 2.5),
                    exit="target", target_r=2.0, max_days=15)


def s_supertrend(b, i):
    if b.st_dir[i] == 1 and b.st_dir[i - 1] == -1 and b.close[i] > b.ema200[i]:
        return dict(entry="open", stop_line=True, exit="supertrend_flip", max_days=60)


def s_ema_cross(b, i):
    if b.ema10[i] > b.ema30[i] and b.ema10[i - 1] <= b.ema30[i - 1] and b.close[i] > b.ema200[i]:
        return dict(entry="open", stop_atr=2.5, exit="ema10_below_30", max_days=60)


def s_macd(b, i):
    if b.macd_hist[i] > 0 >= b.macd_hist[i - 1] and b.macd[i] < 0 and b.close[i] > b.ema200[i]:
        return dict(entry="open", stop_atr=2.5, exit="macd_cross_down", max_days=40)


def s_oversold_leader(b, i):
    if np.isfinite(b.rs126[i]) and b.rs126[i] >= 0.8 and b.rsi[i] < 35:
        return dict(entry="open", stop_atr=3.0, exit="rsi_above_55", max_days=15)


def s_oversold_leader_wide(b, i):
    """Looser version for more trades: top 30% relative strength, RSI(14) < 40."""
    if np.isfinite(b.rs126[i]) and b.rs126[i] >= 0.7 and b.rsi[i] < 40:
        return dict(entry="open", stop_atr=3.0, exit="rsi_above_55", max_days=15)


def s_pattern(name):
    fn = {"Flat base breakout": p_flat_base, "Double bottom": p_double_bottom, "Cup and handle": p_cup_handle}[name]
    def f(b, i):
        p = fn(b, i)
        if not p or p["status"] != "breakout":
            return None
        pl = plan_v2(b, i, p)
        if not pl:
            return None
        return dict(entry="limit", level=pl["entry"], stop=pl["stop"], exit="trail_ema20", breakeven=True, max_days=40)
    return f


def s_gap_up(b, i):
    if b.open[i] >= b.high[i - 1] * 1.02 and b.close[i] > b.open[i] and b.vol20[i - 1] > 0 and \
            b.volume[i] >= 2 * b.vol20[i - 1] and b.close[i] > b.ema50[i]:
        e = b.close[i]
        return dict(entry="limit", level=e, stop=_bound_stop(e, b.low[i] - 0.1 * b.atr[i], b.atr[i]),
                    exit="trail_ema20", breakeven=True, max_days=30)


STRATEGIES = {
    # name: (function, family, one-line description)
    "Pullback to 20 EMA":       (s_pullback("20 EMA"), "Trend pullback", "Uptrend, dips to rising 20 EMA, bounce candle. Buy stop above bounce high; trail on 20 EMA."),
    "Pullback to 50 EMA":       (s_pullback("50 EMA"), "Trend pullback", "Same, at the 50 EMA."),
    "RSI-2 dip in uptrend":     (s_rsi2, "Mean reversion", "Above 200-day avg and 2-day RSI < 10. Buy next open; sell on close above 5-day avg."),
    "Bollinger lower band":     (s_bb_low, "Mean reversion", "Above 200-day avg, closes below lower Bollinger band. Sell at the middle band."),
    "3 down days in uptrend":   (s_3down, "Mean reversion", "Above 200-day avg, 3 lower closes in a row. Sell on close above previous day's high."),
    "Oversold leader":          (s_oversold_leader, "Mean reversion", "Top 20% relative strength, RSI(14) < 35. Sell when RSI > 55."),
    "Oversold leader (wide)":   (s_oversold_leader_wide, "Mean reversion", "Top 30% relative strength, RSI(14) < 40. Sell when RSI > 55."),
    "20-day breakout":          (s_donchian, "Breakout / trend", "Close above 20-day high, above 200 EMA. Exit on close below 10-day low."),
    "52-week high + volume":    (s_52wh, "Breakout / trend", "First close above 52-week high on 1.5x volume. Exit on close below 20 EMA."),
    "Flat base breakout":       (s_pattern("Flat base breakout"), "Chart pattern", "Confirmed close above a tight base; limit buy on retest; trail."),
    "Double bottom":            (s_pattern("Double bottom"), "Chart pattern", "Confirmed close above the neckline; limit buy on retest; trail."),
    "Cup and handle":           (s_pattern("Cup and handle"), "Chart pattern", "Confirmed close above handle high; limit buy on retest; trail."),
    "NR7 breakout":             (s_nr7, "Volatility", "Narrowest range of 7 days in an uptrend. Buy stop above its high, 2R target."),
    "Gap-up on volume":         (s_gap_up, "Momentum", "2%+ gap up, closes strong on 2x volume. Limit at close; trail."),
    "Supertrend (10,3) flip":   (s_supertrend, "Indicator", "Supertrend turns up above 200 EMA. Exit when it turns down."),
    "EMA 10/30 crossover":      (s_ema_cross, "Indicator", "10 EMA crosses above 30 EMA above 200 EMA. Exit on cross back."),
    "MACD cross below zero":    (s_macd, "Indicator", "MACD crosses its signal while below zero, above 200 EMA. Exit on cross down."),
}


def sim_strategy(b, i, sp, cfg):
    """Execute a strategy signal from bar i+1. Returns dict or None."""
    atr = b.atr[i]
    start = i + 1
    if start >= b.n:
        return dict(result="Pending")
    fill_i = fill = None
    if sp["entry"] == "open":
        fill_i, fill = start, b.open[start]
    else:
        lvl = sp["level"]
        for k in range(start, min(start + cfg["entry_valid_days"], b.n)):
            if sp["entry"] == "stop" and b.high[k] >= lvl:
                fill_i, fill = k, max(lvl, b.open[k]); break
            if sp["entry"] == "limit" and b.low[k] <= lvl:
                fill_i, fill = k, min(lvl, b.open[k]); break
        if fill_i is None:
            return dict(result="Expired" if start + cfg["entry_valid_days"] <= b.n else "Pending")
    # initial stop
    if "stop" in sp:
        if fill <= sp["stop"]:
            return dict(result="Expired")          # gapped through the stop before filling: skip the trade
        stop = min(sp["stop"], fill - 1.0 * atr)   # never less than 1 ATR of room after the actual fill
    elif sp.get("stop_low10"):
        stop = _bound_stop(fill, b.lo10[i] - 0.1 * atr, atr, 1.5, 3.0)
    elif sp.get("stop_line"):
        stop = _bound_stop(fill, b.st_line[i], atr, 1.5, 4.0)
    else:
        stop = fill - sp["stop_atr"] * atr
    risk = fill - stop
    if risk <= 0:
        return None
    t = fill + sp.get("target_r", 0) * risk if sp["exit"] == "target" else None
    mfe = 0.0
    ex = sp["exit"]
    for k in range(fill_i, min(fill_i + sp["max_days"], b.n)):
        o, h, l, c = b.open[k], b.high[k], b.low[k], b.close[k]
        if l <= stop:
            px = min(o, stop) if k > fill_i else stop
            return dict(result="Stop" if stop < fill else "Breakeven stop", fill_i=fill_i, fill=fill, exit_i=k, exit=px,
                        r=(px - fill) / risk, mfe=max(mfe, h / fill - 1), stop=stop if stop < fill else fill - risk)
        mfe = max(mfe, h / fill - 1)
        if t is not None and h >= t:
            px = max(o, t) if k > fill_i else t
            return dict(result="Target", fill_i=fill_i, fill=fill, exit_i=k, exit=px, r=(px - fill) / risk, mfe=mfe, stop=fill - risk)
        if sp.get("breakeven") and h >= fill + risk:
            stop = max(stop, fill)
        done = False
        if k > fill_i or sp["entry"] == "open":
            if ex == "trail_ema20":
                done = c < b.ema20[k] and c > fill
            elif ex == "close_above_sma5":
                done = c > b.sma5[k]
            elif ex == "close_above_bbmid":
                done = c >= b.bb_mid[k]
            elif ex == "close_above_prev_high":
                done = c > b.high[k - 1]
            elif ex == "close_below_lo10":
                done = k > fill_i and c < b.lo10[k - 1]
            elif ex == "close_below_ema20":
                done = c < b.ema20[k]
            elif ex == "supertrend_flip":
                done = b.st_dir[k] == -1
            elif ex == "ema10_below_30":
                done = b.ema10[k] < b.ema30[k]
            elif ex == "macd_cross_down":
                done = b.macd_hist[k] < 0
            elif ex == "rsi_above_55":
                done = b.rsi[k] > 55
        if done:
            return dict(result="Exit rule", fill_i=fill_i, fill=fill, exit_i=k, exit=c, r=(c - fill) / risk, mfe=mfe, stop=fill - risk)
    if fill_i + sp["max_days"] <= b.n:
        k = fill_i + sp["max_days"] - 1
        return dict(result="Time exit", fill_i=fill_i, fill=fill, exit_i=k, exit=b.close[k], r=(b.close[k] - fill) / risk, mfe=mfe, stop=fill - risk)
    return dict(result="Open", fill_i=fill_i, fill=fill, r=(b.close[-1] - fill) / risk, mfe=mfe, stop=fill - risk)


def strategy_backtest(data, ctx, cfg, test_start, names=None, progress_every=40):
    names = names or list(STRATEGIES)
    rows, t0 = [], time.time()
    for n_, (sym, df) in enumerate(data.items()):
        b = extra_indicators(attach_context(Bars(indicators(df)), sym, ctx))
        first = max(260, int(np.searchsorted(b.dates.values, np.datetime64(pd.Timestamp(test_start)))))
        liquid = (b.close >= cfg["min_price"]) & ~(b.turnover_cr < cfg["min_turnover_cr"])
        for nm in names:
            fn = STRATEGIES[nm][0]
            i = first
            while i < b.n - 1:
                if not liquid[i]:
                    i += 1; continue
                sp = fn(b, i)
                if not sp:
                    i += 1; continue
                o = sim_strategy(b, i, sp, cfg)
                if o is None:
                    i += 1; continue
                if o["result"] in ("Pending", "Open"):
                    break
                if o["result"] == "Expired":
                    i += 1; continue
                fe = features(b, i)
                rows.append(dict(strategy=nm, symbol=sym, signal_date=b.dates[i], fill_date=b.dates[o["fill_i"]],
                                 exit_date=b.dates[o["exit_i"]], fill=o["fill"], exit=o["exit"], stop=o["stop"],
                                 risk_pct=(o["fill"] - o["stop"]) / o["fill"] * 100, result=o["result"], r=o["r"],
                                 mfe_pct=o["mfe"] * 100, hold_days=o["exit_i"] - o["fill_i"] + 1,
                                 score_v1=tech_score(b, i)[0], score_v2=score_v2(fe), **{k: fe[k] for k in
                                 ("rs_pct126", "rsi", "ext20", "adx", "vol_ratio", "atr_pct", "breadth", "nifty200", "dist_52wh")}))
                i = o["exit_i"] + 1
        if (n_ + 1) % progress_every == 0 or n_ + 1 == len(data):
            print(f"  {n_ + 1}/{len(data)} stocks ({time.time() - t0:.0f}s, {len(rows)} trades)", flush=True)
    t = pd.DataFrame(rows)
    if len(t):
        t["r_net"] = t["r"] - (COST_PCT / 100) / (t["risk_pct"] / 100)
        t["win"] = t["r_net"] > 0
    return t


def strategy_stats(t, split_date):
    def agg(x):
        w, l_ = x.loc[x["r_net"] > 0, "r_net"], x.loc[x["r_net"] <= 0, "r_net"]
        return pd.Series(dict(trades=len(x), win_pct=x["win"].mean() * 100, avg_win_R=w.mean(), avg_loss_R=l_.mean(),
                              expectancy_R=x["r_net"].mean(), profit_factor=w.sum() / -l_.sum() if len(l_) and l_.sum() < 0 else np.nan,
                              avg_days=x["hold_days"].mean()))
    full = t.groupby("strategy").apply(agg)
    sd = pd.Timestamp(split_date)
    ins = t[t["signal_date"] <= sd].groupby("strategy").apply(agg)[["trades", "win_pct", "expectancy_R"]]
    oos = t[t["signal_date"] > sd].groupby("strategy").apply(agg)[["trades", "win_pct", "expectancy_R"]]
    ins.columns = [f"IS {c}" for c in ins.columns]; oos.columns = [f"OOS {c}" for c in oos.columns]
    s = full.join(ins).join(oos)
    s["family"] = [STRATEGIES[k][1] for k in s.index]
    s["robust"] = np.where((s["IS expectancy_R"] > 0.05) & (s["OOS expectancy_R"] > 0.05), "YES (both periods)",
                  np.where((s["IS expectancy_R"] > 0) & (s["OOS expectancy_R"] > 0), "weak", "no"))
    return s.sort_values("OOS expectancy_R", ascending=False)


def momentum_rotation(data, ctx, idx, capital, cost_pct, top_n=10, keep_rank=20, lookback=126, min_breadth=0.4, test_start=None):
    """Weekly: hold the top_n stocks by 6-month return (above 200 EMA). Keep a holding while it ranks within keep_rank.
    Trades at next day's open. Goes to cash when breadth < min_breadth."""
    opens = pd.DataFrame({s: d["Open"] for s, d in data.items()}).sort_index()
    closes = pd.DataFrame({s: d["Close"] for s, d in data.items()}).reindex(opens.index)
    ema200 = closes.ewm(span=200, adjust=False, ignore_na=True).mean()
    ret = closes.pct_change(lookback)
    turn = pd.DataFrame({s: (d["Close"] * d["Volume"]).rolling(20).mean() / 1e7 for s, d in data.items()}).reindex(opens.index)
    dates = opens.index[opens.index >= pd.Timestamp(test_start)] if test_start is not None else opens.index[260:]
    week_last = pd.Series(dates, index=dates).groupby([dates.isocalendar().year.values, dates.isocalendar().week.values]).max().values
    rebal = set(pd.DatetimeIndex(week_last))
    cash, pos, eq, trades, entry = capital, {}, [], [], {}
    pending = None
    for d in dates:
        if pending is not None:          # execute at today's open
            tgt = pending
            for s in list(pos):
                if s not in tgt:
                    px = opens.at[d, s]
                    if np.isfinite(px):
                        cash += pos[s] * px * (1 - cost_pct / 200)
                        trades.append(dict(symbol=s, entry_date=entry[s][0], exit_date=d, entry=entry[s][1], exit=px,
                                           ret_pct=(px / entry[s][1] - 1) * 100 - cost_pct))
                        del pos[s]
            val = cash + sum(pos[s] * closes.at[d, s] for s in pos if np.isfinite(closes.at[d, s]))
            slot = val / top_n
            for s in tgt:
                if s not in pos and len(pos) < top_n:
                    px = opens.at[d, s]
                    if np.isfinite(px) and px > 0 and cash > slot * 0.5:
                        q = math.floor(min(slot, cash) / (px * (1 + cost_pct / 200)))
                        if q > 0:
                            cash -= q * px * (1 + cost_pct / 200); pos[s] = q; entry[s] = (d, px)
            pending = None
        val = cash + sum(pos[s] * closes.at[d, s] for s in pos if np.isfinite(closes.at[d, s]))
        eq.append((d, val))
        if d in rebal:
            br = ctx["breadth"].get(d, 1.0)
            if br < min_breadth:
                pending = set()
            else:
                r = ret.loc[d][(closes.loc[d] > ema200.loc[d]) & (turn.loc[d] >= 5)].dropna().sort_values(ascending=False)
                ranks = {s: k + 1 for k, s in enumerate(r.index)}
                keep = {s for s in pos if ranks.get(s, 999) <= keep_rank}
                new = [s for s in r.index if s not in keep][: max(0, top_n - len(keep))]
                pending = keep | set(new)
    curve = pd.Series(dict(eq))
    return curve, pd.DataFrame(trades)


def curve_stats(curve, start=None, end=None):
    c = curve
    if start is not None:
        c = c[c.index > pd.Timestamp(start)]
    if end is not None:
        c = c[c.index <= pd.Timestamp(end)]
    if len(c) < 5:
        return dict()
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    return dict(total_return_pct=round((c.iloc[-1] / c.iloc[0] - 1) * 100, 1),
                cagr_pct=round(((c.iloc[-1] / c.iloc[0]) ** (1 / max(yrs, .25)) - 1) * 100, 1),
                max_dd_pct=round(((c / c.cummax()) - 1).min() * 100, 1))


# ======================================================================================
# PAPER TRADING ENGINE (v5)
# Decisions (which signals were taken, how many shares) are written once to the ledger and never changed.
# Fills and exits are computed from market data with the same simulator used in the backtests.
# ======================================================================================
LEDGER_COLS = ["signal_date", "symbol", "strategy", "qty", "equity_at_signal", "score_v2", "breadth", "est_entry", "est_stop", "logged_at"]


LOCKED_SPEC = {
    "Oversold leader":        dict(entry="open", stop_atr=3.0, exit="rsi_above_55", max_days=15),
    "Oversold leader (wide)": dict(entry="open", stop_atr=3.0, exit="rsi_above_55", max_days=15),
    "20-day breakout":        dict(entry="open", stop_low10=True, exit="close_below_lo10", max_days=60),
    "Supertrend (10,3) flip": dict(entry="open", stop_line=True, exit="supertrend_flip", max_days=60),
    "EMA 10/30 crossover":    dict(entry="open", stop_atr=2.5, exit="ema10_below_30", max_days=60),
}


def load_ledger_file(path):
    if path and os.path.exists(path):
        meta = pd.read_excel(path, sheet_name="Meta", dtype=str).set_index("key")["value"].to_dict()
        meta = {k: (None if (v is None or str(v) in ("None", "nan", "NaT", "")) else v) for k, v in meta.items()}
        led = pd.read_excel(path, sheet_name="Ledger")
        if len(led):
            led["signal_date"] = pd.to_datetime(led["signal_date"])
        return meta, led
    return None, pd.DataFrame(columns=LEDGER_COLS)


def _bars_for(data, ctx, cache, s):
    if s not in cache:
        cache[s] = extra_indicators(attach_context(Bars(indicators(data[s])), s, ctx))
    return cache[s]


def evaluate_trade(b, row, cfg):
    """Outcome of one ledger entry using all data available now."""
    i = int(np.searchsorted(b.dates.values, np.datetime64(pd.Timestamp(row["signal_date"]))))
    if i >= b.n or b.dates[i] != pd.Timestamp(row["signal_date"]):
        return dict(status="Data missing")
    sp = LOCKED_SPEC.get(row["strategy"])     # exit rules are fixed by strategy; the entry decision is never re-checked
    if sp is None:
        sp = STRATEGIES[row["strategy"]][0](b, i)
        if not sp:
            return dict(status="Signal vanished (data revised)")
    o = sim_strategy(b, i, sp, cfg)
    if o is None:
        return dict(status="Skipped (invalid stop)")
    res = o["result"]
    if res == "Pending":
        return dict(status="Order pending")
    if res == "Expired":
        return dict(status="Not filled")
    out = dict(fill_date=b.dates[o["fill_i"]], fill=o["fill"], stop=o["stop"], fill_i=o["fill_i"])
    if res == "Open":
        out.update(status="Open", last=b.close[-1], days=b.n - o["fill_i"])
    else:
        out.update(status="Closed", result=res, exit_date=b.dates[o["exit_i"]], exit=o["exit"], days=o["exit_i"] - o["fill_i"] + 1)
    return out


def book(ledger, data, ctx, cfg, cost_pct, cache):
    """Evaluate every ledger entry; returns DataFrame with P&L."""
    rows = []
    for _, r in ledger.iterrows():
        if r["symbol"] not in data:
            rows.append(dict(r, status="No data")); continue
        b = _bars_for(data, ctx, cache, r["symbol"])
        e = evaluate_trade(b, r, cfg)
        d = dict(r); d.update(e)
        q = r["qty"]
        if e["status"] == "Closed":
            d["pnl"] = q * (e["exit"] - e["fill"]) - cost_pct / 100 * q * e["fill"]
            d["ret_pct"] = (e["exit"] / e["fill"] - 1) * 100 - cost_pct
            d["r_mult"] = d["pnl"] / (q * (e["fill"] - e["stop"])) if e["fill"] > e["stop"] else np.nan
        elif e["status"] == "Open":
            d["pnl"] = q * (e["last"] - e["fill"]) - cost_pct / 200 * q * e["fill"]
            d["ret_pct"] = (e["last"] / e["fill"] - 1) * 100
        rows.append(d)
    df = pd.DataFrame(rows)
    for c in ("status", "fill_date", "fill", "stop", "exit_date", "exit", "result", "last", "days", "pnl", "ret_pct", "r_mult"):
        if c not in df.columns:
            df[c] = np.nan
    for c in ("fill_date", "exit_date"):
        df[c] = pd.to_datetime(df[c])
    return df


def state_on(book_df, day):
    """Realized P&L and occupied slots as of the close of `day` (no look-ahead)."""
    day = pd.Timestamp(day)
    realized, occupied, invested = 0.0, set(), 0.0
    for _, r in book_df.iterrows():
        st = r.get("status")
        if st == "Closed" and pd.Timestamp(r["exit_date"]) <= day:
            realized += r["pnl"]
        elif st in ("Closed", "Open") and pd.Timestamp(r["fill_date"]) <= day:
            occupied.add(r["symbol"]); invested += r["qty"] * r["fill"]
        elif st in ("Order pending", "Open", "Closed") and pd.Timestamp(r["signal_date"]) <= day:
            occupied.add(r["symbol"]); invested += r["qty"] * r["est_entry"]   # order placed but fills after `day`
    return realized, occupied, invested


def signals_on(data, ctx, cache, i_day, version, cfg):
    """All v5 signals at the close of trading day i_day (a Timestamp)."""
    v = version
    br = ctx["breadth"].get(i_day, np.nan)
    active = v["always"] + (v["trend"] if br >= v["breadth"] else [])
    out = []
    for s in data:
        b = _bars_for(data, ctx, cache, s)
        i = int(np.searchsorted(b.dates.values, np.datetime64(i_day)))
        if i >= b.n or b.dates[i] != i_day or i < 265:
            continue
        if b.close[i] < cfg["min_price"] or b.turnover_cr[i] < cfg["min_turnover_cr"]:
            continue
        for nm in active:
            sp = STRATEGIES[nm][0](b, i)
            if not sp:
                continue
            atr = b.atr[i]; entry = b.close[i]
            if sp.get("stop_low10"):
                stop = min(max(b.lo10[i] - 0.1 * atr, entry - 3 * atr), entry - 1.5 * atr)
            elif sp.get("stop_line"):
                stop = min(max(b.st_line[i], entry - 4 * atr), entry - 1.5 * atr)
            else:
                stop = entry - sp["stop_atr"] * atr
            out.append(dict(signal_date=i_day, symbol=s, strategy=nm, est_entry=round(entry, 2), est_stop=round(stop, 2),
                            score_v2=score_v2(features(b, i)), breadth=round(float(br), 3)))
    df = pd.DataFrame(out)
    if len(df):
        df = df.sort_values("score_v2", ascending=False).drop_duplicates("symbol")
    return df


def run_paper_day(ledger, meta, data, ctx, idx, cfg, version, capital, cost_pct):
    """Process every trading day after meta['last_day'] up to the latest data. Returns updated ledger and meta."""
    cache = {}
    days = idx.index[idx.index <= max(d.index[-1] for d in data.values())]
    last = pd.Timestamp(meta["last_day"]) if meta.get("last_day") else None
    todo = [d for d in days if last is None or d > last]
    if last is None:
        todo = todo[-1:]          # first run: start trading from the latest close only
    new_rows = []
    for d in todo:
        led = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else ledger
        bk = book(led, {s: data[s] for s in data}, ctx, cfg, cost_pct, cache) if len(led) else pd.DataFrame()
        realized, occupied, invested = state_on(bk, d) if len(bk) else (0.0, set(), 0.0)
        equity = capital + realized
        cash = equity - invested
        free = version["max_pos"] - len(occupied)
        if free <= 0:
            continue
        sig = signals_on(data, ctx, cache, d, version, cfg)
        if not len(sig):
            continue
        sig = sig[~sig["symbol"].isin(occupied)].head(free)
        for r in sig.itertuples():
            risk_ps = r.est_entry - r.est_stop
            # size: risk-based, but never more than an equal slot (equity / max positions) or the cash left (no margin)
            qty = int(min(equity * version["risk"] / 100 / risk_ps, equity / version["max_pos"] / r.est_entry,
                          max(0.0, cash) / (r.est_entry * 1.02))) if risk_ps > 0 else 0
            if qty < 1:
                continue
            cash -= qty * r.est_entry
            new_rows.append(dict(signal_date=d, symbol=r.symbol, strategy=r.strategy, qty=qty, equity_at_signal=round(equity, 2),
                                 score_v2=r.score_v2, breadth=r.breadth, est_entry=r.est_entry, est_stop=r.est_stop,
                                 logged_at=datetime.now().strftime("%Y-%m-%d %H:%M")))
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
    meta = dict(meta); meta["last_day"] = str(todo[-1].date()) if todo else meta.get("last_day")
    return ledger, meta, todo


def equity_curve(bk, data, capital, start, end):
    days = pd.bdate_range(start, end)
    eq = pd.Series(capital, index=days, dtype=float)
    for _, r in bk[bk["status"].isin(["Open", "Closed"])].iterrows():
        px = data[r["symbol"]]["Close"].reindex(days).ffill()
        fd = pd.Timestamp(r["fill_date"]); ed = pd.Timestamp(r["exit_date"]) if r["status"] == "Closed" else None
        q, f = r["qty"], r["fill"]
        hold = (days >= fd) & ((days < ed) if ed is not None else True)
        eq[hold] += q * (px[hold] - f)
        if ed is not None:
            eq[days >= ed] += r["pnl"]
    return eq


CLOSED += ['Target', 'Exit rule', 'Breakeven stop']
COST_PCT = 0.25



# ======================================================================================
# BOT: runs once per day on GitHub Actions (or anywhere with Python + internet)
# ======================================================================================
V5_TREND = ["20-day breakout", "Supertrend (10,3) flip", "EMA 10/30 crossover"]
VERSIONS = {
    "v5 core":        dict(always=["Oversold leader"], trend=V5_TREND, breadth=0.70, risk=1.5, max_pos=6),
    "v5 more trades": dict(always=["Oversold leader", "Oversold leader (wide)"], trend=V5_TREND, breadth=0.70, risk=1.0, max_pos=10),
}
EXPECTED = {"Oversold leader": (55, 65), "Oversold leader (wide)": (55, 65), "20-day breakout": (35, 46),
            "Supertrend (10,3) flip": (35, 50), "EMA 10/30 crossover": (30, 45)}
MAX_DAYS = {"Oversold leader": 15, "Oversold leader (wide)": 15, "20-day breakout": 60, "Supertrend (10,3) flip": 60, "EMA 10/30 crossover": 60}


def save_ledger(path, meta, ledger):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"key": list(meta.keys()), "value": [str(v) for v in meta.values()]}).to_excel(xw, sheet_name="Meta", index=False)
        ledger[LEDGER_COLS].to_excel(xw, sheet_name="Ledger", index=False)


def exit_hint(b, strategy):
    i = b.n - 1
    return {"Oversold leader": f"RSI {b.rsi[i]:.0f} → sells on a close above 55",
            "Oversold leader (wide)": f"RSI {b.rsi[i]:.0f} → sells on a close above 55",
            "20-day breakout": f"sells on a close below {b.lo10[i]:.2f} (10-day low)",
            "Supertrend (10,3) flip": f"Supertrend {b.st_line[i]:.2f} ({'up' if b.st_dir[i] == 1 else 'down: exits'})",
            "EMA 10/30 crossover": f"EMA10 {b.ema10[i]:.2f} vs EMA30 {b.ema30[i]:.2f}"}.get(strategy, "")


def md_table(df):
    if df is None or not len(df):
        return "_None._\n"
    cols = list(df.columns)
    out = "| " + " | ".join(map(str, cols)) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join("" if (isinstance(v, float) and v != v) else (f"{v:,.2f}" if isinstance(v, float) else str(v)) for v in r.values) + " |\n"
    return out


def equity_png(eq, nifty, capital, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(eq.index, eq / capital * 100, color="#1f5fbf", lw=1.7, label="Paper account")
    if len(nifty) > 1:
        ax.plot(nifty.index, nifty / nifty.iloc[0] * 100, color="#333", ls="--", lw=1.1, label="Nifty 50")
    ax.axhline(100, color="#ccc", lw=.8); ax.legend(frameon=False); ax.grid(alpha=.2)
    ax.set_title("Paper account vs Nifty (start = 100)", loc="left", fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def chart_file(b, row, path):
    png = base64.b64decode(chart_png(b.d, row))
    with open(path, "wb") as fh:
        fh.write(png)


def notify_telegram(text):
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return False
    import urllib.request, urllib.parse
    data = urllib.parse.urlencode({"chat_id": chat, "text": text[:4000], "disable_web_page_preview": "true"}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=20)
        return True
    except Exception as e:
        print("Telegram failed:", e)
        return False


PALETTE = ["#0f6e56", "#2f7fd0", "#d9822b", "#8a63c7", "#c2364a", "#3a9e9e", "#b08a00", "#6b7f2a", "#9c4f86", "#5a6b75"]
RESULT_LABEL = {"Stop": "Stop hit", "Exit rule": "Exit signal", "Time exit": "Time limit", "Target": "Target", "Breakeven stop": "Breakeven stop"}


def _f(v, d=2):
    try:
        v = float(v)
        return None if not np.isfinite(v) else round(v, d)
    except Exception:
        return None


def build_dashboard(root, meta, version, capital, cost, data, ctx, idx, cache, bk, eq, data_date, health_df):
    docs = os.path.join(root, "docs")
    start = pd.Timestamp(meta["start_day"])
    nifty = idx["Close"].reindex(eq.index).ffill()
    op = bk[bk["status"] == "Open"] if len(bk) else bk
    cl = bk[bk["status"] == "Closed"] if len(bk) else bk
    orders = bk[(bk["signal_date"] == data_date) & (bk["status"] == "Order pending")] if len(bk) else bk
    mv = float((op["qty"] * op["last"]).sum()) if len(op) else 0.0
    invested = float((op["qty"] * op["fill"]).sum()) if len(op) else 0.0
    equity = float(eq.iloc[-1]); prev = float(eq.iloc[-2]) if len(eq) > 1 else equity
    realized = float(cl["pnl"].sum()) if len(cl) else 0.0
    kpi = dict(equity=_f(equity, 0), day_change=_f(equity - prev, 0), day_change_pct=_f((equity / prev - 1) * 100),
               ret_pct=_f((equity / capital - 1) * 100), nifty_ret=_f((nifty.iloc[-1] / nifty.iloc[0] - 1) * 100) if nifty.notna().sum() > 1 else 0.0,
               cash=_f(equity - mv, 0), invested=_f(invested, 0), unrealized=_f(mv - invested, 0), realized=_f(realized, 0),
               open_count=int(len(op)), closed_count=int(len(cl)), win_rate=_f((cl["pnl"] > 0).mean() * 100, 1) if len(cl) else None,
               max_dd=_f(((eq / eq.cummax()) - 1).min() * 100))
    alloc = []
    for k, r in enumerate(op.itertuples() if len(op) else []):
        alloc.append(dict(label=r.symbol, pct=round(r.qty * r.last / equity * 100, 2), color=PALETTE[k % len(PALETTE)]))
    alloc.append(dict(label="Cash", pct=round(max(0.0, (equity - mv) / equity * 100), 2), color="#c8ccc6"))
    open_rows = []
    for r in (op.itertuples() if len(op) else []):
        b = cache.get(r.symbol) or _bars_for(data, ctx, cache, r.symbol)
        d = data[r.symbol]["Close"].tail(45)
        open_rows.append(dict(symbol=r.symbol, strategy=r.strategy, filled=str(pd.Timestamp(r.fill_date).date()), qty=int(r.qty),
                              fill=_f(r.fill), last=_f(r.last), stop=_f(r.stop), ret=_f(r.ret_pct), pnl=_f(r.pnl, 0),
                              value=_f(r.qty * r.last, 0), room=_f((r.last - r.stop) / r.last * 100), days=int(r.days),
                              max_days=MAX_DAYS.get(r.strategy, 15), days_left=int(MAX_DAYS.get(r.strategy, 15) - r.days),
                              watch=exit_hint(b, r.strategy), spark=[[str(i.date()), _f(v)] for i, v in d.items()]))
    order_rows = [dict(symbol=r.symbol, strategy=r.strategy, qty=int(r.qty), entry=_f(r.est_entry), stop=_f(r.est_stop),
                       stop_pct=_f((r.est_entry - r.est_stop) / r.est_entry * 100, 1), exit_rule=STRATEGIES[r.strategy][2].split(". ")[-1])
                  for r in (orders.itertuples() if len(orders) else [])]
    closed_rows = [dict(symbol=r.symbol, strategy=r.strategy, filled=str(pd.Timestamp(r.fill_date).date()), exited=str(pd.Timestamp(r.exit_date).date()),
                        days=int(r.days), fill=_f(r.fill), exit=_f(r.exit), ret=_f(r.ret_pct), r=_f(r.r_mult), pnl=_f(r.pnl, 0),
                        result=RESULT_LABEL.get(r.result, r.result))
                   for r in (cl.sort_values("exit_date", ascending=False).itertuples() if len(cl) else [])]
    health = []
    if health_df is not None and len(health_df):
        for r in health_df.itertuples():
            health.append(dict(strategy=r.Strategy, trades=int(r._2), win=float(r._3), expected=r._4, avg_r=_f(r._5), pnl=_f(r._6, 0), status=r.Health))
    me = eq.resample("ME").last(); base = pd.concat([pd.Series([capital], index=[start - pd.Timedelta(days=1)]), me])
    months = [dict(label=i.strftime("%b %y"), ret=_f(v * 100)) for i, v in base.pct_change().dropna().items()]
    br = float(ctx["breadth"].iloc[-1])
    payload = dict(version=meta["version"], start=str(start.date()), data_date=str(data_date.date()),
                   updated=datetime.now().strftime("%d %b %Y %H:%M UTC"), max_pos=version["max_pos"], cost_pct=cost, kpi=kpi,
                   market=dict(breadth=round(br, 3), threshold=version["breadth"], trend_on=bool(br >= version["breadth"]), universe=len(data)),
                   alloc=alloc, orders=order_rows, open=open_rows, closed=closed_rows, health=health, months=months,
                   equity=[[str(i.date()), _f(v, 0), _f(n)] for i, v, n in zip(eq.index, eq.values, nifty.values)])
    js = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    with open(os.path.join(root, "dashboard_template.html"), encoding="utf-8") as fh:
        html = fh.read().replace("/*DATA*/", js)
    with open(os.path.join(docs, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    with open(os.path.join(docs, "data.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "config.json")) as fh:
        conf = json.load(fh)
    state_dir, docs = os.path.join(root, "state"), os.path.join(root, "docs")
    os.makedirs(state_dir, exist_ok=True); os.makedirs(os.path.join(docs, "charts"), exist_ok=True)
    ledger_path = os.path.join(state_dir, "paper_ledger.xlsx")
    meta, ledger = load_ledger_file(ledger_path)
    if meta is None:
        meta = dict(start_day=None, last_day=None, capital=conf["capital"], version=conf["version"], cost_pct=conf["cost_pct"])
        if conf.get("start_from"):
            meta["last_day"] = str((pd.Timestamp(conf["start_from"]) - pd.Timedelta(days=1)).date())
        print("New paper account:", meta)
    version = VERSIONS[meta["version"]]; capital = float(meta["capital"]); cost = float(meta["cost_pct"])
    cfg = dict(CONFIG, capital=capital)

    symbols = read_universe_file(os.path.join(root, "stocks.txt"))
    now = pd.Timestamp(datetime.now())
    first = pd.Timestamp(meta["last_day"]) if meta.get("last_day") else now
    raw, idx, failed = download(symbols, (min(first, now) - pd.Timedelta(days=560)).strftime("%Y-%m-%d"),
                                (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), cfg["index"])
    if idx is None or not len(raw):
        print("No market data downloaded. Nothing changed."); return 1
    raw, ca_events = adjust_corporate_actions(raw)
    data, warns, data_date = quality_check(raw, idx)
    data = {s: d for s, d in data.items() if len(d) > 300}
    ctx = market_context(data, idx)
    prev_last = meta.get("last_day")
    ledger, meta, processed = run_paper_day(ledger, meta, data, ctx, idx, cfg, version, capital, cost)
    if not meta.get("start_day"):
        meta["start_day"] = str(processed[0].date()) if processed else str(data_date.date())
    if not processed:
        print(f"No new trading day since {prev_last} (holiday or data not updated yet). Nothing changed.")
        return 0

    cache = {}
    bk = book(ledger, data, ctx, cfg, cost, cache) if len(ledger) else pd.DataFrame(columns=LEDGER_COLS + ["status"])
    save_ledger(ledger_path, meta, ledger)

    # ---------- tables
    br = ctx["breadth"].iloc[-1]; trend_on = br >= version["breadth"]
    orders = bk[(bk["signal_date"] == data_date) & (bk["status"] == "Order pending")] if len(bk) else bk
    op = bk[bk["status"] == "Open"].copy() if len(bk) else bk
    cl = bk[bk["status"] == "Closed"].copy() if len(bk) else bk
    new_exits = cl[pd.to_datetime(cl["exit_date"]).isin(processed)] if len(cl) else cl
    new_fills = bk[(bk["status"].isin(["Open", "Closed"])) & pd.to_datetime(bk["fill_date"]).isin(processed)] if len(bk) else bk
    start = pd.Timestamp(meta["start_day"])
    eq = equity_curve(bk, data, capital, start, data_date) if len(bk) else pd.Series([capital, capital], index=[start, data_date])
    nifty = idx["Close"][start:]
    ret = (eq.iloc[-1] / capital - 1) * 100
    nret = (nifty.iloc[-1] / nifty.iloc[0] - 1) * 100 if len(nifty) > 1 else 0.0

    t_orders = pd.DataFrame()
    if len(orders):
        t_orders = pd.DataFrame({"Stock": orders["symbol"], "Strategy": orders["strategy"], "Qty": orders["qty"].astype(int),
                                 "Est. entry": orders["est_entry"].round(2), "Est. stop": orders["est_stop"].round(2),
                                 "Stop %": ((orders["est_entry"] - orders["est_stop"]) / orders["est_entry"] * 100).round(1)})
    t_open = pd.DataFrame()
    if len(op):
        t_open = pd.DataFrame({"Stock": op["symbol"], "Strategy": op["strategy"], "Filled": pd.to_datetime(op["fill_date"]).dt.strftime("%d %b"),
                               "Qty": op["qty"].astype(int), "Fill": op["fill"].round(2), "Last": op["last"].round(2), "Stop": op["stop"].round(2),
                               "Return %": op["ret_pct"].round(2), "P&L ₹": op["pnl"].round(0),
                               "Days left": [MAX_DAYS.get(s, 15) - int(d) for s, d in zip(op["strategy"], op["days"])],
                               "Exit watch": [exit_hint(cache[s], st) for s, st in zip(op["symbol"], op["strategy"])]})
    t_closed = pd.DataFrame()
    if len(cl):
        c2 = cl.sort_values("exit_date", ascending=False).head(25)
        t_closed = pd.DataFrame({"Stock": c2["symbol"], "Strategy": c2["strategy"], "Filled": pd.to_datetime(c2["fill_date"]).dt.strftime("%d %b %y"),
                                 "Exited": pd.to_datetime(c2["exit_date"]).dt.strftime("%d %b %y"), "Exit type": c2["result"],
                                 "Return %": c2["ret_pct"].round(2), "R": c2["r_mult"].round(2), "P&L ₹": c2["pnl"].round(0)})
    t_health = pd.DataFrame()
    if len(cl):
        h = cl.groupby("strategy").agg(trades=("pnl", "size"), win=("pnl", lambda s: (s > 0).mean() * 100), avg_R=("r_mult", "mean"), pnl=("pnl", "sum"))
        def verdict(r):
            lo, hi = EXPECTED.get(r.name, (0, 100))
            if r["trades"] < 20:
                return f"too early ({int(r['trades'])}/20 trades)"
            return "in line" if lo - 5 <= r["win"] <= hi + 5 else ("better" if r["win"] > hi else "WORSE: review")
        t_health = pd.DataFrame({"Strategy": h.index, "Closed trades": h["trades"].astype(int), "Win %": h["win"].round(0),
                                 "Expected win %": [f"{EXPECTED.get(s, (0, 0))[0]}–{EXPECTED.get(s, (0, 0))[1]}" for s in h.index],
                                 "Avg R": h["avg_R"].round(2), "P&L ₹": h["pnl"].round(0), "Health": h.apply(verdict, axis=1).values})

    # ---------- charts
    equity_png(eq, nifty, capital, os.path.join(docs, "equity.png"))
    for f in os.listdir(os.path.join(docs, "charts")):
        os.remove(os.path.join(docs, "charts", f))
    chart_links = []
    for _, r in pd.concat([orders, op]).iterrows() if (len(orders) or len(op)) else []:
        entry = r["fill"] if r["status"] == "Open" else r["est_entry"]
        stop = r["stop"] if r["status"] == "Open" else r["est_stop"]
        row = dict(symbol=r["symbol"], pattern=dict(name=r["strategy"], status="pullback", pivot=entry),
                   plan=dict(entry=entry, stop=stop, t1=entry + 2 * (entry - stop), t2=entry + 3 * (entry - stop)))
        fn = f"{r['symbol']}.png"
        try:
            chart_file(cache.get(r["symbol"]) or _bars_for(data, ctx, cache, r["symbol"]), row, os.path.join(docs, "charts", fn))
            chart_links.append((r["symbol"], fn))
        except Exception as e:
            print("chart failed", r["symbol"], e)

    # ---------- REPORT.md (renders on GitHub, even in private repos)
    md = [f"# v5 Paper Trading: {data_date:%A %d %b %Y}\n",
          f"_Updated automatically {datetime.now():%d %b %Y %H:%M} UTC · version **{meta['version']}** · started {start:%d %b %Y}_\n",
          f"**Equity ₹{eq.iloc[-1]:,.0f}** ({ret:+.2f}%) · Nifty since start {nret:+.2f}% · open positions {len(op)} · closed trades {len(cl)}"
          + (f" · win rate {(cl['pnl'] > 0).mean() * 100:.0f}%" if len(cl) else "") + f" · max drawdown {((eq / eq.cummax()) - 1).min() * 100:.1f}%\n",
          f"Market breadth {br * 100:.0f}% → trend strategies **{'ON' if trend_on else 'OFF'}**\n",
          "![equity](equity.png)\n",
          "## Orders for the next session (buy at the open)\n", md_table(t_orders),
          "After the fill, the stop sits the same % below the fill price.\n" if len(t_orders) else "",
          "## Open positions\n", md_table(t_open),
          "## Health check vs backtest\n", md_table(t_health),
          "## Recent closed trades\n", md_table(t_closed)]
    if chart_links:
        md.append("## Charts\n" + "\n".join(f"**{s}**\n\n![{s}](charts/{f})\n" for s, f in chart_links))
    if warns or ca_events or failed:
        md.append("## Data notes\n" + "\n".join(f"- {w}" for w in (ca_events + warns)[:20]) + (f"\n- Not downloaded: {', '.join(failed)}" if failed else "") + "\n")
    md.append("\n_Paper trading only. Not investment advice._\n")
    with open(os.path.join(docs, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    import shutil
    build_dashboard(root, meta, version, capital, cost, data, ctx, idx, cache, bk, eq, data_date, t_health)
    os.makedirs(os.path.join(state_dir, "history"), exist_ok=True)
    pd.concat([bk.assign(as_of=str(data_date.date()))]).to_csv(os.path.join(state_dir, "history", f"book_{data_date.date()}.csv"), index=False)

    # ---------- phone notification
    lines = [f"v5 paper · {data_date:%d %b %Y}", f"Equity ₹{eq.iloc[-1]:,.0f} ({ret:+.2f}%) · Nifty {nret:+.2f}%",
             f"Breadth {br * 100:.0f}% · trend {'ON' if trend_on else 'OFF'}"]
    if len(new_fills):
        lines.append("Filled: " + ", ".join(f"{r.symbol} @{r.fill:.2f}" for r in new_fills.itertuples()))
    if len(new_exits):
        lines.append("Closed: " + ", ".join(f"{r.symbol} {r.ret_pct:+.1f}%" for r in new_exits.itertuples()))
    if len(t_orders):
        lines.append("Tomorrow buy at open: " + ", ".join(f"{r.Stock} x{r.Qty}" for r in t_orders.itertuples()))
    else:
        lines.append("No new orders.")
    if len(op):
        lines.append("Open: " + ", ".join(f"{r.symbol} {r.ret_pct:+.1f}%" for r in op.itertuples()))
    if conf.get("dashboard_url"):
        lines.append(conf["dashboard_url"])
    msg = "\n".join(lines)
    print(msg)
    print("Telegram:", "sent" if notify_telegram(msg) else "not configured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
