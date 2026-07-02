"""
DAYBREAK data layer — every external fetch routes through here.

Priority: better free tiers when keys exist (Alpaca IEX real-time quotes
and news, Finnhub earnings calendar and company news), always degrading
to yfinance so the app runs fine with ZERO keys configured.

Keys come from the environment (GitHub Actions repo secrets) or from
Streamlit Cloud app secrets — NEVER from code:
    ALPACA_KEY_ID, ALPACA_SECRET   quotes/bars + news
    FINNHUB_KEY                    earnings calendar + news fallback
    NTFY_TOPIC                     morning push alerts (ntfy.sh)
    ANTHROPIC_API_KEY (optional)   one-line "why it's moving" summaries
    POLYGON_KEY (stubbed)          wired but inactive — future
                                   historical + options upgrade slot

No Streamlit UI here; `_secret` may lazily READ st.secrets when running
inside the dashboard, but this module renders nothing.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

ET = ZoneInfo("America/New_York")
ALPACA_DATA = "https://data.alpaca.markets"
FINNHUB = "https://finnhub.io/api/v1"
UA = {"User-Agent": "daybreak-screener/1.0"}


# ----------------------------------------------------------------- secrets ---

def _secret(name: str) -> str | None:
    v = os.environ.get(name)
    if v and v.strip():
        return v.strip()
    try:  # dashboard only; absent/misconfigured secrets never raise
        import streamlit as st
        v = st.secrets.get(name)
        return str(v).strip() if v else None
    except Exception:
        return None


def alpaca_keys() -> tuple[str, str] | None:
    k, s = _secret("ALPACA_KEY_ID"), _secret("ALPACA_SECRET")
    return (k, s) if k and s else None


def finnhub_key() -> str | None:
    return _secret("FINNHUB_KEY")


def anthropic_key() -> str | None:
    return _secret("ANTHROPIC_API_KEY")


def ntfy_topic() -> str | None:
    return _secret("NTFY_TOPIC")


def polygon_key() -> str | None:
    return _secret("POLYGON_KEY")


def polygon_enabled() -> bool:
    # Wired but inactive — slot for the historical + options data upgrade.
    return False


# ------------------------------------------------------------------- http ---

def _get(url: str, *, headers: dict | None = None, params: dict | None = None,
         attempts: int = 3, timeout: int = 8):
    """Polite GET with exponential backoff. Raises after the last attempt;
    callers wrap in try/except and degrade."""
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, headers={**UA, **(headers or {})},
                             params=params, timeout=timeout)
            if r.status_code == 429:
                raise RuntimeError("rate limited")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if i < attempts - 1:
                time.sleep(1.2 * (2 ** i))
    raise last  # type: ignore[misc]


def _alpaca_headers() -> dict | None:
    ak = alpaca_keys()
    if not ak:
        return None
    return {"APCA-API-KEY-ID": ak[0], "APCA-API-SECRET-KEY": ak[1]}


# ------------------------------------------------------------ quotes/bars ---

def latest_prices(symbols: list[str]) -> dict[str, tuple[float, datetime]]:
    """Alpaca IEX latest trades for a symbol list — real-time overlay for
    the scanner. {} when keys are absent or the call fails (caller keeps
    its yfinance prices)."""
    hdr = _alpaca_headers()
    if not hdr or not symbols:
        return {}
    out: dict[str, tuple[float, datetime]] = {}
    try:
        for i in range(0, len(symbols), 100):
            batch = symbols[i:i + 100]
            j = _get(f"{ALPACA_DATA}/v2/stocks/trades/latest",
                     headers=hdr,
                     params={"symbols": ",".join(batch), "feed": "iex"})
            for sym, tr in (j.get("trades") or {}).items():
                try:
                    px = float(tr["p"])
                    ts = datetime.fromisoformat(
                        tr["t"].replace("Z", "+00:00")).astimezone(ET)
                    if px > 0:
                        out[sym] = (px, ts)
                except Exception:
                    continue
    except Exception:
        return {}
    return out


def alpaca_intraday(symbol: str) -> pd.DataFrame | None:
    """Today's 5-minute IEX bars (4:00 ET onward), ET-indexed OHLCV.
    None when keys are absent, the call fails, or there are no bars yet —
    callers fall back to yfinance."""
    hdr = _alpaca_headers()
    if not hdr:
        return None
    try:
        start = datetime.now(ET).replace(hour=4, minute=0, second=0,
                                         microsecond=0)
        j = _get(f"{ALPACA_DATA}/v2/stocks/bars", headers=hdr, params={
            "symbols": symbol, "timeframe": "5Min", "feed": "iex",
            "start": start.isoformat(), "limit": 10000,
        })
        rows = (j.get("bars") or {}).get(symbol) or []
        if not rows:
            return None
        df = pd.DataFrame([{
            "ts": datetime.fromisoformat(b["t"].replace("Z", "+00:00")),
            "Open": b["o"], "High": b["h"], "Low": b["l"],
            "Close": b["c"], "Volume": b["v"],
        } for b in rows]).set_index("ts")
        df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
        return df.astype(float)
    except Exception:
        return None


# ---------------------------------------------------------------- earnings ---

def finnhub_earnings(symbols: list[str], start: date,
                     end: date) -> dict[str, dict] | None:
    """One ranged calendar call. A symbol with no row in a successful
    response is genuinely 'clear' for the window (stronger than the
    per-symbol yfinance fallback). None = call unavailable/failed."""
    key = finnhub_key()
    if not key:
        return None
    try:
        j = _get(f"{FINNHUB}/calendar/earnings", params={
            "from": start.isoformat(), "to": end.isoformat(), "token": key})
        rows = j.get("earningsCalendar") or []
        want = {s.upper() for s in symbols}
        hits: dict[str, str] = {}
        for r in rows:
            sym = str(r.get("symbol", "")).upper()
            if sym in want and r.get("date"):
                d = str(r["date"])
                if sym not in hits or d < hits[sym]:
                    hits[sym] = d
        return {s: ({"status": "imminent", "date": hits[s]} if s in hits
                    else {"status": "clear", "date": None})
                for s in want}
    except Exception:
        return None


# -------------------------------------------------------------------- news ---

# Recognizable outlets only; everything else is dropped.
NEWS_WHITELIST = (
    "reuters", "bloomberg", "cnbc", "benzinga", "associated press",
    "wsj", "wall street journal", "barron", "marketwatch", "dow jones",
    "business wire", "businesswire", "pr newswire", "prnewswire",
    "globenewswire",
)


def _source_ok(src: str) -> bool:
    s = (src or "").strip().lower()
    return s == "ap" or any(tok in s for tok in NEWS_WHITELIST)


def _norm_item(headline, source, url, ts) -> dict | None:
    if not headline or not url or ts is None:
        return None
    return {"headline": str(headline).strip(), "source": str(source).strip(),
            "url": str(url), "ts": ts}


def _alpaca_news(symbol: str, since: datetime) -> list[dict]:
    hdr = _alpaca_headers()
    if not hdr:
        return []
    j = _get(f"{ALPACA_DATA}/v1beta1/news", headers=hdr, params={
        "symbols": symbol, "start": since.isoformat(), "limit": 50,
        "sort": "desc"})
    out = []
    for n in j.get("news") or []:
        ts = datetime.fromisoformat(
            str(n.get("created_at")).replace("Z", "+00:00")).astimezone(ET)
        it = _norm_item(n.get("headline"), n.get("source"), n.get("url"), ts)
        if it:
            out.append(it)
    return out


def _finnhub_news(symbol: str, since: datetime) -> list[dict]:
    key = finnhub_key()
    if not key:
        return []
    j = _get(f"{FINNHUB}/company-news", params={
        "symbol": symbol, "from": since.date().isoformat(),
        "to": datetime.now(ET).date().isoformat(), "token": key})
    out = []
    for n in j or []:
        try:
            ts = datetime.fromtimestamp(int(n.get("datetime", 0)),
                                        tz=ET)
        except Exception:
            continue
        it = _norm_item(n.get("headline"), n.get("source"), n.get("url"), ts)
        if it:
            out.append(it)
    return out


def _yf_news(symbol: str) -> list[dict]:
    out = []
    for n in (yf.Ticker(symbol).news or [])[:20]:
        try:
            c = n.get("content") or {}
            if c:  # newer yfinance shape
                headline = c.get("title")
                src = ((c.get("provider") or {}).get("displayName") or "")
                url = ((c.get("canonicalUrl") or {}).get("url")
                       or (c.get("clickThroughUrl") or {}).get("url"))
                ts_raw = c.get("pubDate")
                ts = (datetime.fromisoformat(
                    str(ts_raw).replace("Z", "+00:00")).astimezone(ET)
                    if ts_raw else None)
            else:  # legacy shape
                headline = n.get("title")
                src = n.get("publisher") or ""
                url = n.get("link")
                ts = (datetime.fromtimestamp(int(n["providerPublishTime"]),
                                             tz=ET)
                      if n.get("providerPublishTime") else None)
            it = _norm_item(headline, src, url, ts)
            if it:
                out.append(it)
        except Exception:
            continue
    return out


def news(symbol: str, limit: int = 3, hours: int = 48) -> list[dict]:
    """Whitelisted headlines from the last `hours`, newest first.
    Alpaca → Finnhub → yfinance; [] on total failure."""
    since = datetime.now(ET) - timedelta(hours=hours)
    items: list[dict] = []
    for fetch in (_alpaca_news, _finnhub_news):
        try:
            items = fetch(symbol, since)
            if items:
                break
        except Exception:
            continue
    if not items:
        try:
            items = _yf_news(symbol)
        except Exception:
            items = []
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x["ts"], reverse=True):
        if it["ts"] < since or not _source_ok(it["source"]):
            continue
        key = it["headline"].lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------- sentiment ---

def stocktwits(symbol: str) -> dict:
    """Bull/bear ratio from the latest Stocktwits stream (no key needed).
    {} on any failure — sentiment is display-only, never gates."""
    try:
        j = _get("https://api.stocktwits.com/api/2/streams/symbol/"
                 f"{symbol}.json", attempts=2)
        msgs = j.get("messages") or []
        bull = bear = 0
        for m in msgs:
            s = (((m.get("entities") or {}).get("sentiment") or {})
                 .get("basic") or "")
            if s == "Bullish":
                bull += 1
            elif s == "Bearish":
                bear += 1
        tagged = bull + bear
        return {
            "st_msgs": len(msgs),
            "st_bull_pct": round(100.0 * bull / tagged, 0) if tagged else None,
        }
    except Exception:
        return {}


def wsb_map() -> dict[str, dict]:
    """ApeWisdom WSB top-100: {SYM: {rank, mentions}}. {} on failure."""
    try:
        j = _get("https://apewisdom.io/api/v1.0/filter/wallstreetbets/page/1",
                 attempts=2)
        out = {}
        for r in j.get("results") or []:
            sym = str(r.get("ticker", "")).upper()
            if sym:
                out[sym] = {"rank": int(r.get("rank", 0)),
                            "mentions": int(r.get("mentions", 0))}
        return out
    except Exception:
        return {}


# ------------------------------------------------------------------ claude ---

def claude_why(symbol: str, gap_pct: float | None,
               headlines: list[dict]) -> str | None:
    """One-line 'why it's moving' via the Claude API. None when the key is
    absent or anything fails — the card simply has no summary line."""
    key = anthropic_key()
    if not key or not headlines:
        return None
    try:
        heads = "\n".join(f"- {h['headline']} ({h['source']})"
                          for h in headlines[:3])
        gap_txt = f"{gap_pct:+.1%}" if gap_pct is not None else "n/a"
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 80,
                "system": ("You are a terse markets-desk assistant. Answer "
                           "with ONE short plain-English sentence saying why "
                           "the stock is likely moving today, based only on "
                           "the given headlines. No advice, no hedging "
                           "boilerplate, no preamble."),
                "messages": [{"role": "user", "content":
                              f"{symbol} gap today: {gap_txt}. Recent "
                              f"headlines:\n{heads}"}],
            }, timeout=20)
        r.raise_for_status()
        txt = r.json()["content"][0]["text"].strip()
        return txt.splitlines()[0][:200] if txt else None
    except Exception:
        return None


# -------------------------------------------------------------------- ntfy ---

def ntfy_send(title: str, message: str, priority: str = "default",
              tags: str = "sunrise") -> bool:
    """Push via ntfy.sh. False (silently) when no topic configured."""
    topic = ntfy_topic()
    if not topic:
        return False
    # HTTP headers are latin-1 only — an em-dash in the title would make
    # requests raise (and the alert silently vanish). Body stays UTF-8.
    safe_title = (title.replace("—", "-").replace("–", "-")
                  .replace("·", "|").encode("ascii", "ignore")
                  .decode().strip() or "DAYBREAK")
    try:
        r = requests.post(
            f"https://ntfy.sh/{topic}", data=message.encode("utf-8"),
            headers={"Title": safe_title, "Priority": priority,
                     "Tags": tags},
            timeout=8)
        return r.status_code < 300
    except Exception:
        return False
