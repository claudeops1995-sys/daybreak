"""
DAYBREAK — Trade-of-the-Day engine
----------------------------------
Two-stage scan of a liquid US equity universe (S&P 500 by default):

  Stage 1  Daily-bar history for the full universe -> liquidity/price filters,
           coarse momentum + mean-reversion proxies -> ~50 candidates.
  Stage 2  Fresh 1-minute (pre/post included) quotes for candidates ->
           gap %, relative volume pace, final composite scores.

Output: one champion trade card (stock sizing + call-option alternative,
both capped at MAX_NOTIONAL) plus a ranked watchlist and diagnostics.

Data source: Yahoo Finance via yfinance (free; quotes may be briefly
delayed). All output is algorithmic screen material for the operator's
own professional review — not investment advice.
"""

from __future__ import annotations

import io
import math
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ET = ZoneInfo("America/New_York")

CONFIG = {
    "max_notional": 5000.0,     # hard cap per position, stock or option
    "min_price": 5.0,
    "max_price": 1500.0,
    "min_dollar_vol": 30e6,     # 20-day average daily dollar volume
    "history_period": "1y",
    "stage2_per_style": 25,     # candidates carried per style into stage 2
    "watchlist_n": 5,
    "momentum_stop_atr": 0.50,  # intraday stop as fraction of daily ATR
    "momentum_tgt_atr": 1.00,
    "meanrev_stop_atr": 0.60,
    "meanrev_tgt_atr": 0.80,
    "time_exit": "15:45 ET",
}

# Used only if the S&P 500 constituent fetch fails.
FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD",
    "MU", "INTC", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "ADI", "MRVL",
    "PLTR", "CRM", "ORCL", "ADBE", "NOW", "SNOW", "NET", "DDOG", "CRWD",
    "PANW", "ZS", "FTNT", "SHOP", "PYPL", "COIN", "HOOD", "SOFI", "NFLX",
    "DIS", "CMCSA", "T", "VZ", "TMUS", "BAC", "JPM", "WFC", "C", "GS",
    "MS", "SCHW", "BLK", "V", "MA", "AXP", "COF", "XOM", "CVX", "COP",
    "OXY", "SLB", "HAL", "FCX", "NEM", "CLF", "NUE", "BA", "GE", "RTX",
    "LMT", "CAT", "DE", "HON", "UPS", "FDX", "DAL", "UAL", "AAL", "LUV",
    "CCL", "RCL", "NCLH", "MAR", "ABNB", "UBER", "DASH", "WMT", "COST",
    "TGT", "HD", "LOW", "NKE", "LULU", "SBUX", "MCD", "CMG", "KO", "PEP",
    "PG", "JNJ", "PFE", "MRK", "LLY", "UNH", "ABBV", "BMY", "AMGN",
    "GILD", "MRNA", "CVS", "ISRG", "SPGI", "GEV", "VST", "SMCI", "ARM",
]


# ----------------------------------------------------------------- clock ---

def now_et() -> datetime:
    return datetime.now(ET)


def market_phase(ts: datetime | None = None) -> str:
    """One of: weekend, overnight, premarket, open, afterhours."""
    ts = ts or now_et()
    if ts.weekday() >= 5:
        return "weekend"
    pre = ts.replace(hour=4, minute=0, second=0, microsecond=0)
    opn = ts.replace(hour=9, minute=30, second=0, microsecond=0)
    cls = ts.replace(hour=16, minute=0, second=0, microsecond=0)
    post = ts.replace(hour=20, minute=0, second=0, microsecond=0)
    if ts < pre:
        return "overnight"
    if ts < opn:
        return "premarket"
    if ts < cls:
        return "open"
    if ts < post:
        return "afterhours"
    return "overnight"


def _fmt_asof(ts: datetime) -> str:
    """'Wed Jul 01, 2026 · 9:07 PM ET' — no platform-specific %-I/%#I."""
    hour12 = ts.hour % 12 or 12
    return f"{ts:%a %b %d, %Y} · {hour12}:{ts:%M %p} ET"


PHASE_LABEL = {
    "open": "Market open — live scan",
    "premarket": "Pre-market — scanning for the open",
    "afterhours": "After hours — preview for next session",
    "overnight": "Overnight — preview for next session",
    "weekend": "Weekend — preview for next session",
}


# -------------------------------------------------------------- universe ---

def get_universe() -> tuple[list[str], str]:
    try:
        html = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (daybreak-screener)"},
            timeout=12,
        ).text
        tbl = pd.read_html(io.StringIO(html))[0]
        syms = (
            tbl["Symbol"].astype(str)
            .str.replace(".", "-", regex=False)
            .str.upper()
            .tolist()
        )
        if len(syms) >= 400:
            return sorted(set(syms)), "S&P 500 constituents"
    except Exception:
        pass
    return FALLBACK_UNIVERSE, "built-in liquid list (S&P fetch unavailable)"


# ------------------------------------------------------------- indicators ---

def _rsi_last(close: pd.Series, n: int) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    val = (100 - 100 / (1 + rs)).iloc[-1]
    return float(val) if pd.notna(val) else float("nan")


def _atr_last(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> float:
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1])


def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not sd or math.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / sd).fillna(0.0)


# --------------------------------------------------------- option pricing ---

RISK_FREE = 0.045  # coarse short-rate; near-dated calls barely care about it


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, sigma: float,
                  r: float = RISK_FREE) -> float:
    """Black–Scholes value of a European call. T in years, sigma annualized.

    Degenerate inputs (expired, zero vol, non-positive price/strike) collapse
    to intrinsic value so the payoff curve stays well-defined at every node.
    """
    S, K, T, sigma = float(S), float(K), float(T), float(sigma)
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    srt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / srt
    d2 = d1 - srt
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


# ----------------------------------------------------------------- stage 1 ---

def build_features(universe: list[str], progress=None) -> pd.DataFrame:
    if progress:
        progress(f"Downloading daily history for {len(universe)} tickers…")
    raw = yf.download(
        universe,
        period=CONFIG["history_period"],
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    today = now_et().date()
    rows = []
    tickers = raw.columns.get_level_values(0).unique() if isinstance(
        raw.columns, pd.MultiIndex) else universe

    for t in tickers:
        try:
            sub = raw[t].dropna()
        except Exception:
            continue
        if len(sub) < 210:
            continue
        c, h, l, o, v = (sub["Close"], sub["High"], sub["Low"],
                         sub["Open"], sub["Volume"])
        has_today = sub.index[-1].date() == today
        price = float(c.iloc[-1])
        prev_close = float(c.iloc[-2]) if has_today else price
        prev_low = float(l.iloc[-2]) if has_today else float(l.iloc[-1])

        # Vendor split-adjustment glitches around split dates can show a
        # phantom 2-4x "move" between adjusted history and raw quotes.
        # No S&P name legitimately moves ±45% in a day, so quarantine it.
        ratio = price / prev_close if prev_close else 1.0
        split_suspect = bool(has_today and (ratio > 1.8 or ratio < 0.55))

        dv20 = float((c * v).rolling(20).mean().iloc[-1])
        if not (CONFIG["min_price"] <= price <= CONFIG["max_price"]):
            continue
        if dv20 < CONFIG["min_dollar_vol"]:
            continue

        sma20 = float(c.rolling(20).mean().iloc[-1])
        sma50 = float(c.rolling(50).mean().iloc[-1])
        sma200 = float(c.rolling(200).mean().iloc[-1])
        std20 = float(c.rolling(20).std().iloc[-1])
        atr = _atr_last(h, l, c)
        rows.append({
            "symbol": t,
            "split_suspect": split_suspect,
            "price": price,
            "prev_close": prev_close,
            "prev_low": prev_low,
            "today_open": float(o.iloc[-1]) if has_today else np.nan,
            "today_vol": float(v.iloc[-1]) if has_today else np.nan,
            "avg_vol20": float(v.rolling(20).mean().iloc[-1]),
            "dollar_vol20": dv20,
            "atr": atr,
            "atr_pct": atr / price if price else np.nan,
            "rsi2": _rsi_last(c, 2),
            "rsi14": _rsi_last(c, 14),
            "ret3": price / float(c.iloc[-4]) - 1 if len(c) >= 4 else np.nan,
            "bb_z": (price - sma20) / std20 if std20 else np.nan,
            "near_high": price / float(h.rolling(20).max().iloc[-1]),
            "above20": price > sma20,
            "above200": price > sma200,
            "trend_up": sma20 > sma50,
        })
    df = pd.DataFrame(rows)
    # An empty frame has no "symbol" column to index on — return it as-is
    # so run_scan can degrade to its error card instead of a KeyError.
    return df.set_index("symbol") if len(df) else df


def shortlist(feat: pd.DataFrame) -> pd.DataFrame:
    f = feat.copy()
    gap_proxy = f["price"] / f["prev_close"] - 1
    f["mom_proxy"] = (
        _z(f["near_high"]) + _z(f["atr_pct"]) + 0.5 * _z(gap_proxy)
        + f["trend_up"].astype(float) * 0.5
    )
    f["mr_proxy"] = _z(-f["rsi2"]) + _z(-f["ret3"]) + _z(-f["bb_z"])
    f.loc[~f["above200"], "mr_proxy"] = -np.inf

    n = CONFIG["stage2_per_style"]
    mom = f.nlargest(n, "mom_proxy").index
    mr = f[f["mr_proxy"] > -np.inf].nlargest(n, "mr_proxy").index
    return f.loc[mom.union(mr)]


# ----------------------------------------------------------------- stage 2 ---

def live_snapshot(cands: pd.DataFrame, progress=None) -> pd.DataFrame:
    syms = cands.index.tolist()
    if progress:
        progress(f"Pulling live quotes for {len(syms)} candidates…")
    live = yf.download(
        syms, period="1d", interval="1m", prepost=True,
        group_by="ticker", auto_adjust=True, threads=True, progress=False,
    )
    out = cands.copy()
    out["live"] = np.nan
    out["quote_time"] = pd.NaT
    for t in syms:
        try:
            closes = live[t]["Close"].dropna()
            if len(closes):
                out.loc[t, "live"] = float(closes.iloc[-1])
                out.loc[t, "quote_time"] = closes.index[-1]
        except Exception:
            continue
    out["live"] = out["live"].fillna(out["price"])
    # If the 1-minute quote disagrees with the daily series by >25%, the two
    # series are on different split-adjustment bases — trust the daily close.
    mismatch = (out["live"] / out["price"] - 1).abs() > 0.25
    out.loc[mismatch, "live"] = out.loc[mismatch, "price"]

    phase = market_phase()
    ts = now_et()
    if phase == "open":
        elapsed = (ts - ts.replace(hour=9, minute=30, second=0)).seconds / 60
        frac = float(np.clip(elapsed / 390.0, 0.08, 1.0))
        out["rvol"] = out["today_vol"] / (out["avg_vol20"] * frac)
    else:
        out["rvol"] = out["today_vol"] / out["avg_vol20"]

    out["gap_pct"] = np.where(
        out["today_open"].notna(),
        out["today_open"] / out["prev_close"] - 1,
        out["live"] / out["prev_close"] - 1,
    )
    out["day_pct"] = out["live"] / out["prev_close"] - 1
    return out


def score(cands: pd.DataFrame) -> pd.DataFrame:
    f = cands.copy()
    f["mom_score"] = (
        0.35 * _z(f["gap_pct"]) + 0.25 * _z(f["rvol"])
        + 0.20 * _z(f["atr_pct"]) + 0.20 * _z(f["near_high"])
        - (~f["above20"]).astype(float) * 0.75
    )
    f["mr_score"] = (
        0.40 * _z(-f["rsi2"]) + 0.25 * _z(-f["ret3"])
        + 0.20 * _z(-f["bb_z"]) + 0.15 * _z(f["atr_pct"])
    )
    f.loc[~f["above200"] | (f["ret3"] >= 0), "mr_score"] = -np.inf

    f["style"] = np.where(f["mom_score"] >= f["mr_score"],
                          "momentum", "mean-reversion")
    f["score"] = f[["mom_score", "mr_score"]].max(axis=1)
    return f.sort_values("score", ascending=False)


# -------------------------------------------------------------- trade card ---

def _round_px(x: float) -> float:
    return round(float(x), 2)


def build_reasons(r: pd.Series) -> list[str]:
    out = []
    if r["style"] == "momentum":
        out.append(f"Gap {r['gap_pct']:+.1%} vs prior close")
        if pd.notna(r["rvol"]):
            out.append(f"{r['rvol']:.1f}× normal volume pace")
        out.append(f"Within {max(0.0, 1 - r['near_high']):.1%} of its 20-day high")
        out.append(f"Daily ATR {r['atr_pct']:.1%} — real intraday range to work with")
    else:
        out.append(f"RSI(2) at {r['rsi2']:.0f} — short-term washout")
        out.append(f"{r['ret3']:+.1%} over 3 sessions, still above its 200-day")
        out.append(f"{abs(r['bb_z']):.1f}σ below its 20-day mean")
        out.append(f"Daily ATR {r['atr_pct']:.1%} gives the bounce room to pay")
    return out


def build_plan(r: pd.Series) -> dict:
    ref = float(r["live"])
    atr = float(r["atr"])
    if r["style"] == "momentum":
        stop = ref - CONFIG["momentum_stop_atr"] * atr
        tgt = ref + CONFIG["momentum_tgt_atr"] * atr
        note = "Enter on strength — price holding above the opening range."
    else:
        stop = min(float(r["prev_low"]), ref - CONFIG["meanrev_stop_atr"] * atr)
        tgt = ref + CONFIG["meanrev_tgt_atr"] * atr
        note = "Enter the flush — scale in near the reference, not chasing green."
    ref, stop, tgt = _round_px(ref), _round_px(stop), _round_px(tgt)
    shares = int(CONFIG["max_notional"] // ref) if ref > 0 else 0
    risk = round(shares * (ref - stop), 0)
    rr = (tgt - ref) / (ref - stop) if ref > stop else float("nan")
    return {
        "entry": ref, "stop": stop, "target": tgt,
        "shares": shares, "notional": round(shares * ref, 0),
        "risk_dollars": risk, "reward_risk": round(rr, 1),
        "entry_note": note, "time_exit": CONFIG["time_exit"],
    }


def pick_option(symbol: str, ref: float) -> dict | None:
    try:
        tk = yf.Ticker(symbol)
        exps = tk.options
        today = now_et().date()
        usable = [e for e in exps
                  if datetime.strptime(e, "%Y-%m-%d").date() >= today]
        if not usable:
            return None
        exp = usable[0]
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        calls = tk.option_chain(exp).calls.copy()
        if calls.empty:
            return None
        mid = np.where((calls["bid"] > 0) & (calls["ask"] > 0),
                       (calls["bid"] + calls["ask"]) / 2, calls["lastPrice"])
        calls["mid"] = mid
        calls = calls[(calls["mid"] > 0.05)
                      & (calls["mid"] * 100 <= CONFIG["max_notional"])]
        if calls.empty:
            return {"unavailable": "No contract fits under the notional cap."}
        itm = calls[calls["strike"] <= ref]
        row = (itm.sort_values("strike").iloc[-1] if not itm.empty
               else calls.sort_values("strike").iloc[0])
        m = float(row["mid"])
        spread = float(row["ask"] - row["bid"]) if row["ask"] > 0 else np.nan
        spread_pct = spread / m if m and not math.isnan(spread) else np.nan
        contracts = int(CONFIG["max_notional"] // (m * 100))
        flags = []
        if pd.notna(row.get("openInterest")) and row["openInterest"] < 200:
            flags.append("thin open interest")
        if pd.notna(spread_pct) and spread_pct > 0.12:
            flags.append(f"wide spread ({spread_pct:.0%} of mid)")
        if dte == 0:
            flags.append("0DTE — decay is brutal, this is a scalp vehicle")
        return {
            "contract": str(row["contractSymbol"]),
            "expiry": exp, "dte": dte, "strike": float(row["strike"]),
            "mid": round(m, 2), "contracts": contracts,
            "cost": round(contracts * m * 100, 0),
            "breakeven": round(float(row["strike"]) + m, 2),
            "open_interest": int(row["openInterest"])
            if pd.notna(row.get("openInterest")) else None,
            "iv": round(float(row["impliedVolatility"]), 3)
            if pd.notna(row.get("impliedVolatility")) else None,
            "flags": flags,
        }
    except Exception:
        return None


# -------------------------------------------------------------------- run ---

def run_scan(progress=None) -> dict:
    t0 = time.time()
    phase = market_phase()
    universe, source = get_universe()

    feat = build_features(universe, progress)
    if feat.empty:
        return {"error": "No usable daily data came back from Yahoo.",
                "diag": {"universe": len(universe), "filtered": 0}}
    quarantined = feat.index[feat["split_suspect"]].tolist()
    feat = feat[~feat["split_suspect"]]
    cands = shortlist(feat)
    if cands.empty:
        return {"error": "No candidates survived the filters.",
                "diag": {"universe": len(universe), "filtered": len(feat)}}
    snap = live_snapshot(cands, progress)
    ranked = score(snap)
    ranked = ranked[np.isfinite(ranked["score"])]
    if ranked.empty:
        return {"error": "No candidates survived the filters.", "diag": {
            "universe": len(universe), "filtered": len(feat)}}

    top = ranked.iloc[0]
    sym = str(top.name)
    if progress:
        progress(f"Building trade card for {sym}…")

    try:
        name = yf.Ticker(sym).info.get("shortName") or sym
    except Exception:
        name = sym

    plan = build_plan(top)
    option = pick_option(sym, plan["entry"])

    card = {
        "symbol": sym, "name": name, "style": str(top["style"]),
        "score": round(float(top["score"]), 2),
        "live": plan["entry"], "prev_close": float(top["prev_close"]),
        "atr": float(top["atr"]),
        "day_pct": float(top["day_pct"]), "gap_pct": float(top["gap_pct"]),
        "rvol": float(top["rvol"]) if pd.notna(top["rvol"]) else None,
        "atr_pct": float(top["atr_pct"]),
        "reasons": build_reasons(top),
        "plan": plan, "option": option,
        "phase": phase, "phase_label": PHASE_LABEL[phase],
        "asof": _fmt_asof(now_et()),
    }

    wl_cols = ["style", "score", "live", "prev_close", "day_pct", "gap_pct",
               "rvol", "atr_pct", "rsi2", "atr", "prev_low"]
    watchlist = ranked.head(CONFIG["watchlist_n"])[wl_cols].copy()
    watchlist.index.name = "symbol"

    # Every watchlist row gets the same ATR-based plan the champion gets, so the
    # detail view can show entry/stop/target for any symbol without a re-scan.
    plans = {str(s): build_plan(ranked.loc[s]) for s in watchlist.index}

    diag = {
        "universe": len(universe), "source": source,
        "passed_filters": len(feat), "stage2": len(cands),
        "quarantined": quarantined,
        "phase": phase, "elapsed_s": round(time.time() - t0, 1),
    }
    return {"card": card, "watchlist": watchlist, "plans": plans, "diag": diag}


if __name__ == "__main__":
    res = run_scan(progress=print)
    import json
    print(json.dumps(res["card"], indent=2, default=str))
    print(res["watchlist"].round(2).to_string())
    print(res["diag"])
