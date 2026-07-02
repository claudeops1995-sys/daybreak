"""
DAYBREAK — Trade-of-the-Day dashboard (Streamlit)
Run locally:   streamlit run app.py
Deploy free:   share.streamlit.io  ->  point at this repo, main file app.py
"""

import json
import math
from datetime import time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from engine import (DEFAULT_SETTINGS, PHASE_LABEL, STYLES, bs_call_greeks,
                    build_output, earnings_candidates, earnings_guard,
                    fetch_features, market_tape, option_exit_value,
                    pick_option, scan_market, session_frac)

st.set_page_config(
    page_title="DAYBREAK — Trade of the Day",
    page_icon="🌅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ theme ---

AMBER = "#FFB454"   # momentum — dawn amber
BLUE = "#5CC8FF"    # mean-reversion — pre-dawn blue
RED = "#E5484D"
INK = "#0B0F14"
PANEL = "#121922"
LINE = "#1E2935"
TEXT = "#E8EEF4"
MUTED = "#8A97A5"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

.stApp { background: #0B0F14; }
.block-container { padding-top: 1.2rem; max-width: 720px; }
h1, h2, h3, p, span, div { color: #E8EEF4; }

.db-wordmark { font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: 1.5rem; letter-spacing: .35em; }
.db-wordmark .sun { color: #FFB454; }
.db-sub { color: #8A97A5; font-size: .8rem; margin-top: -2px; }

.db-pill { display:inline-block; font-family:'IBM Plex Mono',monospace;
  font-size:.68rem; letter-spacing:.08em; color:#8A97A5;
  border:1px solid #1E2935; border-radius:999px; padding:3px 10px; }

/* ---- component vocabulary (spacing scale 4/8/12/16/24) ---------------- */
.card { background:#121922; border:1px solid #1E2935; border-radius:14px;
  padding:16px 16px 12px; margin:12px 0 8px; }
.card-rule { height:3px; border-radius:3px; margin:-16px -16px 12px;
  background:linear-gradient(90deg,#FFB454,#5CC8FF); }
.card-head { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.chip { font-family:'IBM Plex Mono',monospace; font-size:.66rem;
  letter-spacing:.14em; padding:4px 9px; border-radius:5px; font-weight:600;
  display:inline-block; }
.chip-solid { background:var(--c,#8A97A5); color:#0B0F14; }
.chip-outline { background:transparent; border:1px solid var(--c,#8A97A5);
  color:var(--c,#8A97A5); padding:3px 8px; }
.chip-muted { background:#1E2935; color:#8A97A5; }
.chip-warn { background:#E5484D; color:#0B0F14; }
.sym { font-family:'Space Grotesk',sans-serif; font-weight:700;
  font-size:2.6rem; line-height:1.05; margin:8px 0 0; }
.sym-sm { font-size:2rem; }
.sub { color:#8A97A5; font-size:.9rem; margin-bottom:8px; }
.px-line { font-family:'IBM Plex Mono',monospace; font-size:1.2rem;
  font-weight:600; }
.up { color:#FFB454; } .dn { color:#5CC8FF; }
.eyebrow { font-family:'IBM Plex Mono',monospace; font-size:.62rem;
  letter-spacing:.14em; color:#8A97A5; margin:16px 0 4px; }
.nm { color:#C7D2DC; opacity:.55; font-size:.86rem; margin:4px 0; }
.nm b { opacity:1; }
.nm-gate { font-family:'IBM Plex Mono',monospace; font-size:.74rem;
  color:#8A97A5; }
.notice { border:1px dashed #1E2935; border-radius:9px; padding:8px 12px;
  color:#8A97A5; font-size:.76rem; font-family:'IBM Plex Mono',monospace;
  margin:8px 0; }
.horizon { height:44px; margin:8px -16px 4px;
  background:radial-gradient(120% 130% at 50% 115%,
  rgba(255,180,84,.18), rgba(92,200,255,.06) 55%, transparent 78%); }

.lvls { display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px;
  margin:12px 0 8px; }
.lvl { background:#0B0F14; border:1px solid #1E2935; border-radius:9px;
  padding:9px 10px; }
.lvl .lab { font-size:.62rem; letter-spacing:.14em; color:#8A97A5;
  font-family:'IBM Plex Mono',monospace; }
.lvl .val { font-family:'IBM Plex Mono',monospace; font-size:1.05rem;
  font-weight:600; margin-top:2px; }
.val-stop { color:#E5484D; }

.meta { font-family:'IBM Plex Mono',monospace; font-size:.78rem;
  color:#8A97A5; margin:2px 0 12px; }
.meta b { color:#E8EEF4; font-weight:600; }

.why { border-top:1px dashed #1E2935; padding-top:12px; margin-top:6px; }
.why .lab { font-size:.62rem; letter-spacing:.14em; color:#8A97A5;
  font-family:'IBM Plex Mono',monospace; }
.why ul { margin:6px 0 0 1.05rem; padding:0; }
.why li { color:#C7D2DC; font-size:.86rem; margin:3px 0; }

.opt { background:#0B0F14; border:1px solid #1E2935; border-radius:9px;
  padding:10px 12px; margin-top:12px; }
.opt .lab { font-size:.62rem; letter-spacing:.14em; color:#8A97A5;
  font-family:'IBM Plex Mono',monospace; }
.opt .line { font-family:'IBM Plex Mono',monospace; font-size:.84rem;
  margin-top:4px; }
.opt .flags { color:#FFB454; font-size:.74rem; margin-top:3px;
  font-family:'IBM Plex Mono',monospace; }

table.wl { width:100%; border-collapse:collapse; margin-top:4px; }
table.wl th { font-family:'IBM Plex Mono',monospace; font-size:.62rem;
  letter-spacing:.12em; color:#8A97A5; text-align:right; padding:6px 4px;
  border-bottom:1px solid #1E2935; }
table.wl th:first-child { text-align:left; }
table.wl td { font-family:'IBM Plex Mono',monospace; font-size:.82rem;
  text-align:right; padding:7px 4px; border-bottom:1px solid #141d27; }
table.wl td:first-child { text-align:left; font-weight:600; }
.dot { display:inline-block; width:7px; height:7px; border-radius:50%;
  margin-right:6px; }

.foot { color:#5d6976; font-size:.7rem; margin-top:18px; line-height:1.5; }

/* ---- tap targets: watchlist rows & controls --------------------------- */
.stButton>button {
  width:100%; min-height:44px; text-align:left;
  font-family:'IBM Plex Mono',monospace; font-size:.8rem;
  background:#121922; color:#E8EEF4; border:1px solid #1E2935;
  border-radius:9px; padding:8px 12px; white-space:pre;
  transition:border-color .12s ease, background .08s ease;
}
.stButton>button:hover { border-color:#8A97A5; color:#E8EEF4; }
.stButton>button:active { background:#1E2935; }   /* pressed state */
.stButton>button[kind="primary"],
.stButton>button[kind="primary"]:hover {
  background:#0B0F14; border:1.5px solid #FFB454; color:#E8EEF4; }

/* ---- touch sizing for settings controls ------------------------------- */
.stNumberInput input, .stTextInput input { min-height:44px; }
.stCheckbox { padding:6px 0; }

/* ---- 390px canvas: step the dense mono surfaces down one size --------- */
@media (max-width:430px) {
  table.wl { font-size:.72rem; }
  table.wl th, table.wl td { padding:6px 3px; }
  .stButton>button { font-size:.72rem; padding:8px 8px; }
}

/* ---- micro-motion: one entrance, CSS-only, reduced-motion aware ------- */
@keyframes db-rise { from { opacity:0; transform:translateY(6px); }
  to { opacity:1; transform:none; } }
.card, .notice { animation: db-rise .28s ease-out both; }
@media (prefers-reduced-motion: reduce) {
  .card, .notice { animation:none; }
  .stButton>button { transition:none; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------- data ---

@st.cache_data(ttl=2700, show_spinner=False)
def cached_features() -> dict:
    # Stage 1: ~500-ticker daily history — the slow half. 45-minute cache;
    # daily bars barely move intraday, and rescans inside the window skip
    # straight to the 10-minute stage-2 refresh (seconds, not a minute).
    return fetch_features()


@st.cache_data(ttl=600, show_spinner=False)
def cached_scan() -> dict:
    # Stage 2 on top of cached stage 1 — settings-independent;
    # build_output() derives plans/cards per rerun so Settings changes
    # are instant.
    return scan_market(prefetched=cached_features())


@st.cache_data(ttl=600, show_spinner=False)
def name_for(symbol: str) -> str:
    try:
        return yf.Ticker(symbol).info.get("shortName") or symbol
    except Exception:
        return symbol


@st.cache_data(ttl=600, show_spinner=False)
def intraday(symbol: str) -> pd.DataFrame:
    df = yf.download([symbol], period="1d", interval="5m", prepost=True,
                     group_by="ticker", auto_adjust=True, progress=False)
    return df[symbol].dropna()


@st.cache_data(ttl=600, show_spinner=False)
def daily_history(symbol: str) -> pd.DataFrame:
    # 6 months so the 50-day SMA is fully warmed before we slice to ~3 months.
    df = yf.download([symbol], period="6mo", interval="1d",
                     group_by="ticker", auto_adjust=True, progress=False)
    return df[symbol].dropna()


@st.cache_data(ttl=600, show_spinner=False)
def option_for(symbol: str, ref: float) -> dict | None:
    return pick_option(symbol, ref)


@st.cache_data(ttl=600, show_spinner=False)
def cached_earnings(symbols: tuple[str, ...]) -> dict:
    try:
        return earnings_guard(list(symbols))
    except Exception:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def cached_tape() -> dict:
    try:
        return market_tape()
    except Exception:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def load_journal() -> list[dict]:
    """Read journal/ (committed by the GitHub Actions workflows; each
    commit redeploys the app, so the files are always local)."""
    out = []
    base = Path(__file__).parent / "journal"
    if not base.exists():
        return out
    for day in sorted(p for p in base.iterdir() if p.is_dir()):
        off = day / "official.json"      # dryrun-* files are ignored
        if not off.exists():
            continue
        try:
            rec = json.loads(off.read_text(encoding="utf-8"))
            outc = day / "outcomes.json"
            rec["outcomes"] = (json.loads(outc.read_text(encoding="utf-8"))
                               if outc.exists() else None)
            out.append(rec)
        except Exception:
            continue  # one corrupt day must not hide the rest
    return out


def option_block_html(symbol: str, o: dict | None,
                      plan: dict | None = None,
                      atr: float | None = None) -> str:
    """Render an option play as HTML — shared by champion card and details."""
    if o and "contract" in o:
        flags = " · ".join(o["flags"]) if o["flags"] else ""
        risk_line = f'<div class="line">max loss ${o["cost"]:,.0f} (full premium)'
        greeks_line = ""
        if plan is not None:
            fb_iv = ((atr / plan["entry"]) * math.sqrt(252)
                     if atr and plan["entry"] else None)
            try:
                stop_pnl = option_exit_value(o, plan["stop"], fb_iv) - o["cost"]
                risk_line += f' · at stock stop {stop_pnl:+,.0f}'
            except Exception:
                pass
            try:
                iv = o.get("iv") or fb_iv or 0.5
                t_now = max(int(o.get("dte", 0)) - session_frac(), 0.0) / 365.0
                g = bs_call_greeks(plan["entry"], o["strike"], t_now, iv)
                pos_theta = g["theta_day"] * 100 * o["contracts"]
                lam = (g["delta"] * 100 * o["contracts"] * plan["entry"]
                       / o["cost"]) if o["cost"] else float("nan")
                greeks_line = (f'<div class="line">Δ {g["delta"]:.2f} · '
                               f'θ {pos_theta:+,.0f}$/day · '
                               f'λ {lam:.1f}× delta-adj leverage</div>')
            except Exception:
                pass
        risk_line += "</div>"
        return (
            f'<div class="opt"><div class="lab">OPTION ALTERNATIVE</div>'
            f'<div class="line">{o["contracts"]}× {symbol} '
            f'${o["strike"]:g}C {o["expiry"]} @ ${o["mid"]:.2f} '
            f'≈ ${o["cost"]:,.0f}</div>'
            f'<div class="line">breakeven ${o["breakeven"]:.2f} · '
            f'{o["dte"]} DTE · OI '
            + (f'{o["open_interest"]:,}' if o["open_interest"] is not None else "—")
            + "</div>" + greeks_line + risk_line
            + (f'<div class="flags">⚠ {flags}</div>' if flags else "")
            + "</div>"
        )
    msg = (o or {}).get(
        "unavailable",
        "No liquid contract fits — chain unavailable or over the cap.")
    return (f'<div class="opt"><div class="lab">OPTION ALTERNATIVE</div>'
            f'<div class="line">{msg}</div></div>')


# ---------------------------------------------------------------- charts ---
# One chart family: shared layout template, shared candle spec, mono axis
# type, no toolbar, right-side axis. VWAP is neutral (#C7D2DC) so it never
# collides with a style accent.

VWAP_C = "#C7D2DC"


def chart_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=INK, plot_bgcolor=INK,
        font=dict(family="IBM Plex Mono, monospace", color=MUTED, size=11),
        showlegend=False,
    )
    fig.update_xaxes(rangeslider_visible=False, gridcolor=LINE)
    fig.update_yaxes(gridcolor=LINE, side="right")
    return fig


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def candles(df: pd.DataFrame, accent: str) -> go.Candlestick:
    return go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], showlegend=False,
        increasing_line_color=accent, decreasing_line_color="#55606c",
        increasing_fillcolor=accent, decreasing_fillcolor="#3a4552",
    )


def render_daily(symbol: str, accent: str) -> None:
    """3-month daily candlestick with 20/50 SMAs (annotated, no legend)."""
    try:
        df = daily_history(symbol)
        if len(df) < 30:
            notice("daily history too short to chart")
            return
        c = df["Close"]
        sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
        view = df.iloc[-63:]
        fig = go.Figure(candles(view, accent))
        for s, col, dash, lab in ((sma20, TEXT, None, "20"),
                                  (sma50, MUTED, "dot", "50")):
            sv = s.iloc[-63:]
            fig.add_trace(go.Scatter(
                x=view.index, y=sv, mode="lines",
                line=dict(color=col, width=1, dash=dash), showlegend=False))
            fig.add_annotation(x=view.index[-1], y=float(sv.iloc[-1]),
                               text=lab, showarrow=False, xshift=14,
                               font=dict(size=9, color=col))
        chart_layout(fig, 300)
        fig.update_layout(margin=dict(l=8, r=26, t=8, b=8))  # SMA tags
        show_chart(fig)
    except Exception:
        notice("daily chart unavailable — data fetch failed")


def render_intraday(symbol: str, plan: dict, accent: str,
                    prev_close: float, live: float) -> None:
    """Today's 5-min chart: entry/stop/target, VWAP, prior close, volume bars."""
    try:
        bars = intraday(symbol)
        if not len(bars):
            notice("no intraday bars yet for this session")
            return
        if abs(float(bars["Close"].iloc[-1]) / live - 1) >= 0.25:
            # 1-min vs daily on different split bases — skip, don't lie
            notice("intraday feed on a different adjustment basis — skipped")
            return
        o, h, l = bars["Open"], bars["High"], bars["Low"]
        c, v = bars["Close"], bars["Volume"]
        # Session VWAP: regular hours only — pre/post prints would skew it.
        # Before the open there are no RTH bars yet; fall back to all bars.
        rth = ((bars.index.time >= dt_time(9, 30))
               & (bars.index.time < dt_time(16, 0)))
        vb = bars[rth] if rth.any() else bars
        vtyp = (vb["High"] + vb["Low"] + vb["Close"]) / 3.0
        vwap = ((vtyp * vb["Volume"]).cumsum()
                / vb["Volume"].cumsum().replace(0, np.nan))
        vol_colors = [accent if cc >= oo else "#3a4552"
                      for oo, cc in zip(o, c)]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.76, 0.24], vertical_spacing=0.04)
        fig.add_trace(candles(bars, accent), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=vwap.index, y=vwap, mode="lines", name="VWAP", showlegend=False,
            line=dict(color=VWAP_C, width=1.2),
            hovertemplate="VWAP %{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Bar(
            x=bars.index, y=v, marker_color=vol_colors, marker_line_width=0,
            showlegend=False), row=2, col=1)

        fig.add_hline(y=prev_close, line_color=MUTED, line_width=1,
                      line_dash="dot", row=1, col=1, annotation_text="prior close",
                      annotation_font_color=MUTED, annotation_font_size=9)
        for y, col, lab in ((plan["entry"], TEXT, "entry"),
                            (plan["stop"], RED, "stop"),
                            (plan["target"], accent, "target")):
            fig.add_hline(y=y, line_color=col, line_width=1, line_dash="dot",
                          row=1, col=1, annotation_text=lab,
                          annotation_font_color=col, annotation_font_size=10)

        chart_layout(fig, 340)
        fig.update_layout(bargap=0.1)
        fig.update_yaxes(showgrid=False, row=2, col=1)
        show_chart(fig)
    except Exception:
        notice("intraday chart unavailable — data fetch failed")


def render_payoff(plan: dict, option: dict | None, atr: float,
                  accent: str) -> None:
    """Same-day P&L across -2 ATR..+2 ATR for the stock and the option."""
    try:
        entry, stop, target = plan["entry"], plan["stop"], plan["target"]
        shares = plan["shares"]
        if not atr or atr <= 0 or shares <= 0:
            notice("payoff unavailable for this plan")
            return
        xs = np.linspace(entry - 2 * atr, entry + 2 * atr, 61)
        # Long stock, truncated where the stop or target would close the trade.
        stock = shares * (np.clip(xs, stop, target) - entry)

        has_opt = bool(option and "strike" in option)
        if has_opt:
            cost = float(option["cost"])
            fb_iv = (atr / entry) * math.sqrt(252) if entry else None

            def opt_val(s: float) -> float:
                # Shared 15:45-exit valuation (T = DTE − session fraction).
                return option_exit_value(option, float(s), fb_iv)

            opt = np.array([opt_val(s) for s in xs]) - cost

        fig = go.Figure()
        fig.add_hline(y=0, line_color=LINE, line_width=1)
        for xline, lab in ((stop, "stop"), (entry, "entry"), (target, "target")):
            fig.add_vline(x=xline, line_color=MUTED, line_width=1, line_dash="dot",
                          annotation_text=lab, annotation_font_size=9,
                          annotation_font_color=MUTED)
        # shaded profit / loss regions under the stock curve
        fig.add_trace(go.Scatter(
            x=xs, y=np.maximum(stock, 0), mode="lines", fill="tozeroy",
            fillcolor="rgba(255,180,84,.10)", showlegend=False,
            line=dict(width=0), hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=xs, y=np.minimum(stock, 0), mode="lines", fill="tozeroy",
            fillcolor="rgba(229,72,77,.10)", showlegend=False,
            line=dict(width=0), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=stock, mode="lines", name="Stock",
                                 line=dict(color=accent, width=2)))
        if has_opt:
            fig.add_trace(go.Scatter(x=xs, y=opt, mode="lines", name="Option",
                                     line=dict(color=VWAP_C, width=1.6,
                                               dash="dash")))
        chart_layout(fig, 300)
        fig.update_layout(showlegend=True,
                          legend=dict(orientation="h", y=1.03, x=0,
                                      font=dict(size=9),
                                      bgcolor="rgba(0,0,0,0)"))
        fig.update_xaxes(tickprefix="$")
        fig.update_yaxes(tickprefix="$")
        show_chart(fig)

        pts = [("STOP", stop), ("ENTRY", entry),
               ("+1 ATR", entry + atr), ("TARGET", target)]
        body = ""
        for lab, s in pts:
            s_pnl = shares * (min(max(s, stop), target) - entry)
            if has_opt:
                o_txt = f"{opt_val(s) - cost:+,.0f}"
            else:
                o_txt = "—"
            body += (f'<tr><td>{lab}</td><td>${s:,.2f}</td>'
                     f'<td>{s_pnl:+,.0f}</td><td>{o_txt}</td></tr>')
        st.markdown(
            '<table class="wl"><tr><th>SCENARIO</th><th>PRICE</th>'
            '<th>STOCK P&amp;L</th><th>OPTION P&amp;L</th></tr>'
            + body + '</table>', unsafe_allow_html=True)
    except Exception:
        notice("payoff view unavailable — computation failed")


def section(label: str) -> None:
    st.markdown(f'<div class="eyebrow">{label}</div>',
                unsafe_allow_html=True)


def notice(msg: str) -> None:
    """Compact inline degraded-state marker — a failed fetch shows this,
    never a blank section and never an exception page."""
    st.markdown(f'<div class="notice">◌ {msg}</div>', unsafe_allow_html=True)


# ------------------------------------------------- component vocabulary ---
# Every card on the page is assembled from these helpers — one visual
# language for champion cards, detail tickets, option blocks, no-trade.

def chip(label: str, color: str = MUTED, variant: str = "solid") -> str:
    """Status chips: solid (TRIGGERED/style), outline (STALKING/LATE
    ENTRY), muted (NO TRADE), warn (EARNINGS)."""
    cls = {"solid": "chip chip-solid", "outline": "chip chip-outline",
           "muted": "chip chip-muted", "warn": "chip chip-warn"}[variant]
    style = (f' style="--c:{color}"' if variant in ("solid", "outline")
             else "")
    return f'<span class="{cls}"{style}>{label}</span>'


def card_head(*chips_html: str) -> str:
    return f'<div class="card-head">{"".join(chips_html)}</div>'


def plan_chips(style: str, plan: dict, earnings: dict | None,
               accent: str) -> str:
    parts = [chip(style.upper(), accent, "solid")]
    if plan.get("status", "triggered") == "triggered":
        parts.append(chip("TRIGGERED", accent, "solid"))
    else:
        parts.append(chip("STALKING", accent, "outline"))
    if plan.get("scale_note"):
        parts.append(chip("LATE ENTRY", AMBER, "outline"))
    e = earnings or {}
    if e.get("status") == "imminent":
        parts.append(chip(f'EARNINGS {e.get("date") or ""}'.strip(),
                          RED, "warn"))
    return card_head(*parts)


def levels_html(p: dict, accent: str) -> str:
    return (f'<div class="lvls">'
            f'<div class="lvl"><div class="lab">ENTRY</div>'
            f'<div class="val">${p["entry"]:,.2f}</div></div>'
            f'<div class="lvl"><div class="lab">STOP</div>'
            f'<div class="val val-stop">${p["stop"]:,.2f}</div></div>'
            f'<div class="lvl"><div class="lab">TARGET</div>'
            f'<div class="val" style="color:{accent}">${p["target"]:,.2f}'
            f'</div></div></div>')


def plan_meta_html(p: dict) -> str:
    return (f'<div class="meta"><b>{p["shares"]} shares</b> ≈ '
            f'${p["notional"]:,.0f} · risk ${p["risk_dollars"]:,.0f} at stop '
            f'· {p["reward_risk"]}R · flat by <b>{p["time_exit"]}</b></div>')


def quote_stale_txt(qt) -> str:
    """' · quote 09:44 ET (2m old)' — red when the print is >15m stale."""
    try:
        if qt is None or pd.isna(qt):
            return ""
        age = max(0.0, (pd.Timestamp.now(tz=qt.tz) - qt).total_seconds() / 60)
        col = RED if age > 15 else MUTED
        return (f' · <span style="color:{col}">quote {qt:%H:%M} ET '
                f'({age:.0f}m old)</span>')
    except Exception:
        return ""


def render_detail(symbol: str) -> None:
    """Full drill-down panel for any watchlist symbol."""
    # A rescan can change the watchlist while detail_sym still holds the old
    # selection — bail quietly rather than KeyError on a vanished symbol.
    plan = plans.get(symbol)
    if plan is None or symbol not in wl.index:
        return
    r = wl.loc[symbol]
    style = str(r["style"])
    mom = style == "momentum"
    acc = AMBER if mom else BLUE
    live, prev_close = float(r["live"]), float(r["prev_close"])
    day_pct = float(r["day_pct"])
    chg_cls = "up" if day_pct >= 0 else "dn"
    rvol_txt = f'{r["rvol"]:.1f}×' if pd.notna(r["rvol"]) else "—"

    failed = gates.get(symbol, [])
    gate_html = (f'<div class="meta" style="color:{AMBER}">⚠ gates: '
                 f'{", ".join(failed)}</div>' if failed else "")
    if plan.get("scale_note"):
        gate_html += (f'<div class="meta" style="color:{AMBER}">⚠ '
                      f'{plan["scale_note"]}</div>')

    # Lazy: the chain is only fetched once a name is selected (cached 10 min).
    try:
        option = option_for(symbol, plan["entry"])
    except Exception:
        option = None
    opt_html = option_block_html(symbol, option, plan, float(r["atr"]))

    st.markdown(
        '<div class="card"><div class="card-rule"></div>'
        + plan_chips(style, plan, earn_map.get(symbol), acc)
        + f'<div class="sym sym-sm">{symbol}</div>'
        + f'<div class="px-line">${live:,.2f} '
        + f'<span class="{chg_cls}">{day_pct:+.1%} today</span></div>'
        + levels_html(plan, acc)
        + plan_meta_html(plan)
        + f'<div class="meta">score <b>{float(r["score"]):.2f}</b> '
        + f'· gap {float(r["gap_pct"]):+.1%} · rvol {rvol_txt} '
        + f'· ATR {float(r["atr_pct"]):.1%} · RSI2 {float(r["rsi2"]):.0f}'
        + f'{quote_stale_txt(r.get("quote_time"))}</div>'
        + f'<div class="meta">{plan["entry_note"]}</div>'
        + gate_html + opt_html + '</div>',
        unsafe_allow_html=True)

    section("3-MONTH DAILY · 20/50 SMA")
    render_daily(symbol, acc)
    section("TODAY · 5-MIN")
    render_intraday(symbol, plan, acc, prev_close, live)
    section("PROJECTED SAME-DAY PAYOFF")
    render_payoff(plan, option, float(r["atr"]), acc)


# --------------------------------------------------------------- settings ---
# Widgets render further down (Settings expander); values are read here from
# session_state so the whole page derives from them on every rerun.

def _sget(key: str, default):
    return st.session_state.get(key, default)


SETTINGS = {
    **DEFAULT_SETTINGS,
    "risk_sizing": bool(_sget("set_risk_sizing",
                              DEFAULT_SETTINGS["risk_sizing"])),
    "risk_budget": float(_sget("set_risk_budget",
                               DEFAULT_SETTINGS["risk_budget"])),
    # Gate inputs are entered in human units (%, ×, R) and converted here.
    "mom_gap_min": float(_sget("set_mom_gap", 1.5)) / 100.0,
    "mom_rvol_min": float(_sget("set_mom_rvol", 1.5)),
    "mom_rr_min": float(_sget("set_mom_rr", 1.5)),
    "mr_rsi2_max": float(_sget("set_mr_rsi2", 10.0)),
    "mr_ret3_max": float(_sget("set_mr_ret3", -3.0)) / 100.0,
    "min_rr": float(_sget("set_min_rr", 1.2)),
    "earnings_guard": bool(_sget("set_earn_guard",
                                 DEFAULT_SETTINGS["earnings_guard"])),
}

# ----------------------------------------------------------------- header ---

left, right = st.columns([3, 1])
with left:
    st.markdown(
        '<div class="db-wordmark">DAYBRE<span class="sun">A</span>K</div>'
        '<div class="db-sub">one trade a day · sized under $5,000</div>',
        unsafe_allow_html=True,
    )
with right:
    if st.button("↻ Rescan", width="stretch"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Scanning the S&P 500 — daily history for ~500 names, then "
                "live quotes. First load takes about a minute; the next 10 "
                "minutes are served from cache."):
    try:
        res = cached_scan()
    except Exception:
        # A Yahoo timeout/rate-limit must degrade to the error card,
        # never to a raw Streamlit exception page.
        res = {"error": "Scan failed — data source unreachable or rate-limited."}

try:
    earn = cached_earnings(tuple(earnings_candidates(res)))
except Exception:
    earn = {}
res = build_output(res, SETTINGS, earn)  # cheap derivation — reruns instant

if "error" in res:
    st.error(res["error"] + " Tap Rescan to try again.")
    st.stop()

card, wl, diag = res["card"], res["watchlist"], res["diag"]
plans = res.get("plans", {})
gates = res.get("gates", {})
earn_map = res.get("earnings", {})

# --------------------------------------------------------------------- IA ---
# Three tabs; TODAY's vertical rhythm is tape → dual champions → watchlist
# → detail. The glance strip answers "tape / two trades / do they qualify"
# before any scrolling.

tape = cached_tape()
risk_off = ((tape.get("SPY") or {}).get("day_pct") or 0) < -0.01

tab_today, tab_journal, tab_settings = st.tabs(
    ["TODAY", "JOURNAL", "SETTINGS"])

with tab_today:
    st.markdown(
        f'<span class="db-pill">{PHASE_LABEL[res["phase"]]}</span>&nbsp;'
        f'<span class="db-pill">{res["asof"]}</span>',
        unsafe_allow_html=True,
    )

    # tape strip
    chips = []
    for k in ("SPY", "QQQ", "VIX"):
        t = tape.get(k)
        if not t:
            continue
        if k == "VIX":
            chips.append(f'<span class="db-pill">VIX {t["last"]:.1f}</span>')
        elif t.get("day_pct") is not None:
            c = AMBER if t["day_pct"] >= 0 else BLUE
            chips.append(f'<span class="db-pill">{k} <span style="color:{c}">'
                         f'{t["day_pct"]:+.1%}</span></span>')
    if chips:
        st.markdown('<div style="margin-top:6px">' + "&nbsp;".join(chips)
                    + "</div>", unsafe_allow_html=True)

    # glance strip — the two trades in one line
    glance = []
    for _style in STYLES:
        _sc = res["style_cards"].get(_style) or {}
        dot = AMBER if _style == "momentum" else BLUE
        if _sc.get("no_trade"):
            glance.append(f'<span class="db-pill" style="opacity:.6">'
                          f'<span class="dot" style="background:{dot}"></span>'
                          f'no trade</span>')
        else:
            _p = _sc.get("plan", {})
            glance.append(f'<span class="db-pill">'
                          f'<span class="dot" style="background:{dot}"></span>'
                          f'{_sc["symbol"]} · {_p.get("status", "—")} · '
                          f'{_p.get("reward_risk", "—")}R</span>')
    if glance:
        st.markdown('<div style="margin-top:6px">' + "&nbsp;".join(glance)
                    + "</div>", unsafe_allow_html=True)

    q = diag.get("quarantined") or []
    if q:
        notice(f'{len(q)} name{"s" if len(q) > 1 else ""} quarantined for '
               f'split-adjustment mismatch: {", ".join(q)} — details in '
               f'Settings › diagnostics')

# ------------------------------------------------------------ trade ticket ---

def render_champion(card: dict) -> None:
    mom = card["style"] == "momentum"
    accent = AMBER if mom else BLUE
    chg_cls = "up" if card["day_pct"] >= 0 else "dn"
    p = card["plan"]
    rvol_txt = f'{card["rvol"]:.1f}×' if card["rvol"] is not None else "—"
    reasons = "".join(f"<li>{r}</li>" for r in card["reasons"])
    cautions = (f'<div class="meta" style="color:{AMBER}">⚠ '
                f'{p["scale_note"]}</div>' if p.get("scale_note") else "")
    if mom and risk_off:
        spy_pct = (tape.get("SPY") or {}).get("day_pct")
        cautions += (f'<div class="meta" style="color:{AMBER}">⚠ tape red '
                     f'(SPY {spy_pct:+.1%}) — momentum longs are fighting '
                     f'the market today.</div>')

    try:
        o = option_for(card["symbol"], p["entry"])
    except Exception:
        o = None
    opt_html = option_block_html(card["symbol"], o, p, card["atr"])

    st.markdown(
        '<div class="card"><div class="card-rule"></div>'
        + plan_chips(card["style"], p, card.get("earnings"), accent)
        + f'<div class="sym">{card["symbol"]}</div>'
        + f'<div class="sub">{card["name"]}</div>'
        + f'<div class="px-line">${card["live"]:,.2f} '
        + f'<span class="{chg_cls}">{card["day_pct"]:+.1%} today</span></div>'
        + levels_html(p, accent)
        + plan_meta_html(p)
        + f'<div class="meta">gap {card["gap_pct"]:+.1%} · rvol {rvol_txt} '
        + f'· ATR {card["atr_pct"]:.1%}'
        + f'{quote_stale_txt(card.get("quote_time"))}</div>'
        + cautions
        + f'<div class="why"><div class="lab">WHY THIS TRADE</div>'
        + f'<ul>{reasons}</ul>'
        + f'<div class="meta" style="margin-top:8px">{p["entry_note"]}</div>'
        + f'</div>{opt_html}</div>',
        unsafe_allow_html=True)
    # Charts and payoff live in the detail flow — the champion is the
    # default detail selection, so they appear right below the watchlist.


def render_style_no_trade(sc: dict) -> None:
    """The skip is a designed moment, not an error: dawn resting on the
    horizon, the statement in display type, near misses greyed below with
    the exact gate each one failed."""
    style = sc["style"]
    accent = AMBER if style == "momentum" else BLUE
    rows = "".join(
        f'<div class="nm"><b>{m["symbol"]}</b> · score {m["score"]:.2f} '
        f'· <span class="nm-gate">{", ".join(m["failed"])}</span></div>'
        for m in sc.get("near_misses", []))
    st.markdown(
        '<div class="card"><div class="card-rule"></div>'
        + card_head(chip(style.upper(), accent, "outline"),
                    chip("NO TRADE", variant="muted"))
        + '<div class="sym">Nothing qualified</div>'
        + f'<div class="sub">No {style} setup cleared the gates. '
        + 'Skipping is a position.</div>'
        + '<div class="horizon"></div>'
        + '<div class="eyebrow">NEAR MISSES · FAILED GATE</div>'
        + rows + '</div>',
        unsafe_allow_html=True)


# Two equal cards — the top pick of EACH style, momentum first.
with tab_today:
    for _style in STYLES:
        _sc = res["style_cards"].get(_style)
        if _sc is None:
            continue
        if _sc.get("no_trade"):
            render_style_no_trade(_sc)
        else:
            _sc["name"] = name_for(_sc["symbol"])
            render_champion(_sc)

with tab_today:
    section("RANKED WATCHLIST")
    syms = [str(s) for s in wl.index]

    # Sticky selection that tolerates a rescan changing the list; defaults
    # to the overall champion so its charts/payoff show without a tap.
    sel = st.session_state.get("detail_sym")
    if sel not in syms:
        sel = (str(card["symbol"])
               if card is not None and str(card["symbol"]) in syms
               else (syms[0] if syms else None))

    smin = float(wl["score"].min()) if len(wl) else 0.0
    rng = (float(wl["score"].max()) - smin) or 1.0
    for sym in syms:
        r = wl.loc[sym]
        blocks = int(round(5 * (float(r["score"]) - smin) / rng))
        bar = "▰" * blocks + "▱" * (5 - blocks)   # score as a bar, not a number
        tag = "MOM" if r["style"] == "momentum" else "MRV"
        marks = (" ✕" if gates.get(sym) else "") + (
            " E!" if (earn_map.get(sym) or {}).get("status") == "imminent"
            else "")
        label = (f"{tag} {sym:<5} {bar} {float(r['score']):.2f} "
                 f"{float(r['live']):>8,.2f} {float(r['day_pct']):+.1%}"
                 f"{marks}")
        if st.button(label, key=f"wl_{sym}",
                     type=("primary" if sym == sel else "secondary")):
            st.session_state["detail_sym"] = sym
            st.rerun()
    st.markdown(
        '<div class="meta" style="margin-top:8px">MOM momentum · '
        'MRV mean-reversion · ▰ score within today\'s list · '
        '✕ failed a gate · E! earnings ≤1 day · amber border = selected'
        '</div>', unsafe_allow_html=True)

    # detail directly under the list — no page jump, chevrons to walk it.
    if sel:
        i = syms.index(sel)
        c_prev, c_next, c_pad = st.columns([1, 1, 2])
        with c_prev:
            if st.button("‹ prev", key="det_prev", disabled=(i == 0)):
                st.session_state["detail_sym"] = syms[i - 1]
                st.rerun()
        with c_next:
            if st.button("next ›", key="det_next",
                         disabled=(i == len(syms) - 1)):
                st.session_state["detail_sym"] = syms[i + 1]
                st.rerun()
        render_detail(sel)

# ----------------------------------------------------------------- journal ---

def render_journal() -> None:
    try:
        days = load_journal()
    except Exception:
        days = []
    if not days:
        st.markdown(
            '<div class="meta">No journal entries yet. The morning workflow '
            'freezes the official cards at ~9:45 ET each weekday and the '
            'nightly workflow scores them after the close — the first row '
            'appears after the next full trading day.</div>',
            unsafe_allow_html=True)
        return

    rows, stats = [], {}
    for rec in days:
        oc = (rec.get("outcomes") or {}).get("styles", {})
        spy = ((rec.get("tape") or {}).get("SPY") or {}).get("day_pct")
        if spy is None:
            regime = "—"
        else:
            col = RED if spy < -0.01 else MUTED
            regime = f'<span style="color:{col}">{spy:+.1%}</span>'
        for style, sc in rec.get("style_cards", {}).items():
            if sc.get("no_trade"):
                rows.append(f'<tr style="opacity:.45"><td>{rec["date"][5:]}</td>'
                            f'<td>{style[:4]}</td><td>no trade</td>'
                            f'<td>—</td><td>—</td><td>—</td><td>—</td>'
                            f'<td>{regime}</td></tr>')
                continue
            o = oc.get(style, {})
            m, f = o.get("model", {}), o.get("fill", {})
            mr, fr = m.get("realized_r"), f.get("realized_r")
            opt_pnl = (o.get("option") or {}).get("pnl")
            chg = (rec.get("changed_from_prelim", {}).get(style, {})
                   or {}).get("changed")
            sym = sc["symbol"] + (" *" if chg else "")
            if mr is not None:
                stats.setdefault(style, []).append((mr, fr))
            rows.append(
                f'<tr><td>{rec["date"][5:]}</td><td>{style[:4]}</td>'
                f'<td>{sym}</td>'
                f'<td>{m.get("exit_reason", "—")}</td>'
                f'<td>{(f"{mr:+.2f}" if mr is not None else "—")}</td>'
                f'<td>{(f"{fr:+.2f}" if fr is not None else "—")}</td>'
                f'<td>{(f"{opt_pnl:+,.0f}" if opt_pnl is not None else "—")}'
                f'</td><td>{regime}</td></tr>')

    st.markdown(
        '<table class="wl"><tr><th>DATE</th><th>STYLE</th><th>SYMBOL</th>'
        '<th>EXIT</th><th>MODEL R</th><th>FILL R</th><th>OPT P&amp;L</th>'
        '<th>SPY</th></tr>'
        + "".join(rows) + "</table>",
        unsafe_allow_html=True)

    lines = []
    for style, pairs in stats.items():
        mrs = [p[0] for p in pairs]
        frs = [p[1] for p in pairs if p[1] is not None]
        wins = sum(1 for r in mrs if r > 0)
        line = (f"<b>{style}</b>: {len(mrs)} trades · "
                f"hit {wins}/{len(mrs)} · expectancy "
                f"{sum(mrs) / len(mrs):+.2f}R")
        if frs:
            slip = (sum(frs) / len(frs)) - (sum(mrs) / len(mrs))
            line += f" · fill slippage {slip:+.2f}R"
        lines.append(line)
    if lines:
        st.markdown(f'<div class="meta" style="margin-top:8px">'
                    f'{"<br>".join(lines)}<br>* champion changed between '
                    f'9:35 prelim and 9:45 official</div>',
                    unsafe_allow_html=True)


with tab_journal:
    section("FROZEN DECISION POINTS · OUTCOMES")
    render_journal()

# ---------------------------------------------------------------- settings ---

with tab_settings:
    st.checkbox(
        "Risk-budget sizing (shares = risk $ ÷ stop distance, "
        "still capped at $5,000 notional)",
        value=DEFAULT_SETTINGS["risk_sizing"], key="set_risk_sizing",
    )
    st.number_input(
        "Risk budget per trade ($)", min_value=10.0, max_value=1000.0,
        value=DEFAULT_SETTINGS["risk_budget"], step=5.0, key="set_risk_budget",
    )
    st.checkbox(
        "Earnings guard — exclude names reporting within 1 trading day "
        "from champion slots (unknown calendar + gap>8% also excluded)",
        value=DEFAULT_SETTINGS["earnings_guard"], key="set_earn_guard",
    )
    st.markdown('<div class="meta" style="margin:10px 0 0">NO-TRADE GATES'
                '</div>', unsafe_allow_html=True)
    st.number_input("Momentum: min gap (%)", min_value=0.0, max_value=10.0,
                    value=1.5, step=0.1, key="set_mom_gap")
    st.number_input("Momentum: min rvol (×)", min_value=0.0, max_value=10.0,
                    value=1.5, step=0.1, key="set_mom_rvol")
    st.number_input("Momentum: min R:R", min_value=0.5, max_value=5.0,
                    value=1.5, step=0.1, key="set_mom_rr")
    st.number_input("Mean-rev: max RSI2", min_value=1.0, max_value=50.0,
                    value=10.0, step=1.0, key="set_mr_rsi2")
    st.number_input("Mean-rev: max 3-day return (%)", min_value=-20.0,
                    max_value=0.0, value=-3.0, step=0.5, key="set_mr_ret3")
    st.number_input("Nomination floor R:R", min_value=0.5, max_value=3.0,
                    value=1.2, step=0.1, key="set_min_rr")

    with st.expander("Scan diagnostics"):
        q = diag["quarantined"]
        st.markdown(
            f"- Universe: **{diag['universe']}** ({diag['source']})\n"
            f"- Passed liquidity/price filters: **{diag['passed_filters']}**\n"
            f"- Carried to live scoring: **{diag['stage2']}**\n"
            f"- Quarantined for split-adjustment mismatch: "
            f"**{', '.join(q) if q else 'none'}**\n"
            f"- Scan time: **{diag['elapsed_s']}s** · phase: {diag['phase']}"
        )

st.markdown(
    '<div class="foot">Algorithmic screen output for the operator\'s own '
    'professional review — not investment advice or a recommendation. '
    'Data: Yahoo Finance via yfinance; quotes may be briefly delayed. '
    'Options involve substantial risk of rapid loss; same-day exits assumed. '
    'Data refreshes on every load (10-min cache) — tap Rescan to force.</div>',
    unsafe_allow_html=True,
)
