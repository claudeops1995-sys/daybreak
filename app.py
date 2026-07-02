"""
DAYBREAK — Trade-of-the-Day dashboard (Streamlit)
Run locally:   streamlit run app.py
Deploy free:   share.streamlit.io  ->  point at this repo, main file app.py
"""

import math
from datetime import time as dt_time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from engine import (DEFAULT_SETTINGS, PHASE_LABEL, bs_call_price,
                    build_output, option_exit_value, pick_option, scan_market)

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

.ticket { background:#121922; border:1px solid #1E2935; border-radius:14px;
  padding:20px 20px 16px; margin:14px 0 6px; }
.ticket-rule { height:3px; border-radius:3px; margin:-20px -20px 16px;
  background:linear-gradient(90deg,#FFB454,#5CC8FF); }
.badge { font-family:'IBM Plex Mono',monospace; font-size:.66rem;
  letter-spacing:.14em; padding:4px 9px; border-radius:5px; font-weight:600; }
.badge-mom { color:#0B0F14; background:#FFB454; }
.badge-mr  { color:#0B0F14; background:#5CC8FF; }
.tk-sym { font-family:'Space Grotesk',sans-serif; font-weight:700;
  font-size:3rem; line-height:1; margin:10px 0 0; }
.tk-name { color:#8A97A5; font-size:.9rem; margin-bottom:10px; }
.tk-px { font-family:'IBM Plex Mono',monospace; font-size:1.25rem;
  font-weight:600; }
.tk-chg-up { color:#FFB454; } .tk-chg-dn { color:#5CC8FF; }

.lvls { display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px;
  margin:16px 0 10px; }
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
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------- data ---

@st.cache_data(ttl=600, show_spinner=False)
def cached_scan() -> dict:
    # Expensive, settings-independent half only — build_output() derives
    # plans/cards from this per rerun so Settings changes are instant.
    return scan_market()


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


def option_block_html(symbol: str, o: dict | None,
                      plan: dict | None = None,
                      atr: float | None = None) -> str:
    """Render an option play as HTML — shared by champion card and details."""
    if o and "contract" in o:
        flags = " · ".join(o["flags"]) if o["flags"] else ""
        risk_line = f'<div class="line">max loss ${o["cost"]:,.0f} (full premium)'
        if plan is not None:
            try:
                fb_iv = ((atr / plan["entry"]) * math.sqrt(252)
                         if atr and plan["entry"] else None)
                stop_pnl = option_exit_value(o, plan["stop"], fb_iv) - o["cost"]
                risk_line += f' · at stock stop {stop_pnl:+,.0f}'
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
            + "</div>" + risk_line
            + (f'<div class="flags">⚠ {flags}</div>' if flags else "")
            + "</div>"
        )
    msg = (o or {}).get(
        "unavailable",
        "No liquid contract fits — chain unavailable or over the cap.")
    return (f'<div class="opt"><div class="lab">OPTION ALTERNATIVE</div>'
            f'<div class="line">{msg}</div></div>')


# ---------------------------------------------------------------- charts ---

def render_daily(symbol: str, accent: str) -> None:
    """3-month daily candlestick with 20/50 SMAs."""
    try:
        df = daily_history(symbol)
        if len(df) < 30:
            return
        c = df["Close"]
        sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
        view = df.iloc[-63:]
        fig = go.Figure(go.Candlestick(
            x=view.index, open=view["Open"], high=view["High"],
            low=view["Low"], close=view["Close"], showlegend=False,
            increasing_line_color=accent, decreasing_line_color="#55606c",
            increasing_fillcolor=accent, decreasing_fillcolor="#3a4552",
        ))
        fig.add_trace(go.Scatter(
            x=view.index, y=sma20.iloc[-63:], mode="lines", name="SMA20",
            line=dict(color=TEXT, width=1)))
        fig.add_trace(go.Scatter(
            x=view.index, y=sma50.iloc[-63:], mode="lines", name="SMA50",
            line=dict(color=MUTED, width=1, dash="dot")))
        fig.update_layout(
            height=300, margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor=INK, plot_bgcolor=INK, font=dict(color=MUTED, size=11),
            xaxis=dict(rangeslider_visible=False, gridcolor=LINE),
            yaxis=dict(gridcolor=LINE, side="right"),
            showlegend=True, legend=dict(orientation="h", y=1.03, x=0,
                                         font=dict(size=9),
                                         bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})
    except Exception:
        pass  # a missing chart never blocks the page


def render_intraday(symbol: str, plan: dict, accent: str,
                    prev_close: float, live: float) -> None:
    """Today's 5-min chart: entry/stop/target, VWAP, prior close, volume bars."""
    try:
        bars = intraday(symbol)
        if not len(bars):
            return
        if abs(float(bars["Close"].iloc[-1]) / live - 1) >= 0.25:
            return  # 1-min vs daily on different split bases — skip, don't lie
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
        fig.add_trace(go.Candlestick(
            x=bars.index, open=o, high=h, low=l, close=c, showlegend=False,
            increasing_line_color=accent, decreasing_line_color="#55606c",
            increasing_fillcolor=accent, decreasing_fillcolor="#3a4552",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=vwap.index, y=vwap, mode="lines", name="VWAP", showlegend=False,
            line=dict(color=BLUE, width=1.2),
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

        fig.update_layout(
            height=390, margin=dict(l=8, r=8, t=8, b=8), bargap=0.1,
            paper_bgcolor=INK, plot_bgcolor=INK, font=dict(color=MUTED, size=11),
            showlegend=False,
        )
        fig.update_xaxes(rangeslider_visible=False, gridcolor=LINE)
        fig.update_yaxes(gridcolor=LINE, side="right", row=1, col=1)
        fig.update_yaxes(gridcolor=LINE, side="right", showgrid=False, row=2, col=1)
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})
    except Exception:
        pass


def render_payoff(plan: dict, option: dict | None, atr: float,
                  accent: str) -> None:
    """Same-day P&L across -2 ATR..+2 ATR for the stock and the option."""
    try:
        entry, stop, target = plan["entry"], plan["stop"], plan["target"]
        shares = plan["shares"]
        if not atr or atr <= 0 or shares <= 0:
            return
        xs = np.linspace(entry - 2 * atr, entry + 2 * atr, 61)
        # Long stock, truncated where the stop or target would close the trade.
        stock = shares * (np.clip(xs, stop, target) - entry)

        has_opt = bool(option and "strike" in option)
        if has_opt:
            K = float(option["strike"])
            contracts = int(option["contracts"])
            cost = float(option["cost"])
            iv = option.get("iv") or (atr / entry) * math.sqrt(252)
            T = max(int(option.get("dte", 0)) - 1, 0) / 365.0  # ~1 day of theta

            def opt_val(s: float) -> float:
                return contracts * 100 * bs_call_price(s, K, T, iv)

            opt = np.array([opt_val(s) for s in xs]) - cost

        fig = go.Figure()
        fig.add_hline(y=0, line_color=LINE, line_width=1)
        for xline, lab in ((stop, "stop"), (entry, "entry"), (target, "target")):
            fig.add_vline(x=xline, line_color=MUTED, line_width=1, line_dash="dot",
                          annotation_text=lab, annotation_font_size=9,
                          annotation_font_color=MUTED)
        fig.add_trace(go.Scatter(x=xs, y=stock, mode="lines", name="Stock",
                                 line=dict(color=accent, width=2)))
        if has_opt:
            fig.add_trace(go.Scatter(x=xs, y=opt, mode="lines", name="Option",
                                     line=dict(color=BLUE, width=2)))
        fig.update_layout(
            height=300, margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor=INK, plot_bgcolor=INK, font=dict(color=MUTED, size=11),
            xaxis=dict(gridcolor=LINE, tickprefix="$"),
            yaxis=dict(gridcolor=LINE, side="right", tickprefix="$"),
            showlegend=True, legend=dict(orientation="h", y=1.03, x=0,
                                         font=dict(size=9),
                                         bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})

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
        pass


def section(label: str) -> None:
    st.markdown(f'<div class="meta" style="margin:16px 0 2px;'
                f'letter-spacing:.12em">{label}</div>', unsafe_allow_html=True)


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
    badge_cls = "badge-mom" if mom else "badge-mr"
    live, prev_close = float(r["live"]), float(r["prev_close"])
    day_pct = float(r["day_pct"])
    chg_cls = "tk-chg-up" if day_pct >= 0 else "tk-chg-dn"
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

    st.markdown(f"""
<div class="ticket">
  <div class="ticket-rule"></div>
  <span class="badge {badge_cls}">{style.upper()}</span>
  <div class="tk-sym" style="font-size:2.2rem">{symbol}</div>
  <div class="tk-px">${live:,.2f}
    <span class="{chg_cls}">{day_pct:+.1%} today</span></div>
  <div class="lvls">
    <div class="lvl"><div class="lab">ENTRY</div>
      <div class="val">${plan["entry"]:,.2f}</div></div>
    <div class="lvl"><div class="lab">STOP</div>
      <div class="val val-stop">${plan["stop"]:,.2f}</div></div>
    <div class="lvl"><div class="lab">TARGET</div>
      <div class="val" style="color:{acc}">${plan["target"]:,.2f}</div></div>
  </div>
  <div class="meta"><b>{plan["shares"]} shares</b> ≈ ${plan["notional"]:,.0f}
     · risk ${plan["risk_dollars"]:,.0f} at stop · {plan["reward_risk"]}R
     · flat by <b>{plan["time_exit"]}</b></div>
  <div class="meta">{style} · score <b>{float(r["score"]):.2f}</b>
     · gap {float(r["gap_pct"]):+.1%} · rvol {rvol_txt}
     · ATR {float(r["atr_pct"]):.1%} · RSI2 {float(r["rsi2"]):.0f}</div>
  {gate_html}
  {opt_html}
</div>
""", unsafe_allow_html=True)

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

with st.spinner("Scanning the S&P 500 — first load takes about a minute…"):
    try:
        res = cached_scan()
    except Exception:
        # A Yahoo timeout/rate-limit must degrade to the error card,
        # never to a raw Streamlit exception page.
        res = {"error": "Scan failed — data source unreachable or rate-limited."}

res = build_output(res, SETTINGS)  # cheap derivation — reruns are instant

if "error" in res:
    st.error(res["error"] + " Tap Rescan to try again.")
    st.stop()

card, wl, diag = res["card"], res["watchlist"], res["diag"]
plans = res.get("plans", {})
gates = res.get("gates", {})

st.markdown(
    f'<span class="db-pill">{PHASE_LABEL[res["phase"]]}</span>&nbsp;'
    f'<span class="db-pill">{res["asof"]}</span>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ trade ticket ---

def render_champion(card: dict) -> None:
    mom = card["style"] == "momentum"
    badge_cls = "badge-mom" if mom else "badge-mr"
    accent = AMBER if mom else BLUE
    chg_cls = "tk-chg-up" if card["day_pct"] >= 0 else "tk-chg-dn"
    p = card["plan"]
    rvol_txt = f'{card["rvol"]:.1f}×' if card["rvol"] is not None else "—"
    reasons = "".join(f"<li>{r}</li>" for r in card["reasons"])
    scale_html = (f'<div class="meta" style="color:{AMBER}">⚠ '
                  f'{p["scale_note"]}</div>' if p.get("scale_note") else "")

    try:
        o = option_for(card["symbol"], p["entry"])
    except Exception:
        o = None
    opt_html = option_block_html(card["symbol"], o, p, card["atr"])

    st.markdown(f"""
<div class="ticket">
  <div class="ticket-rule"></div>
  <span class="badge {badge_cls}">{card["style"].upper()}</span>
  <div class="tk-sym">{card["symbol"]}</div>
  <div class="tk-name">{card["name"]}</div>
  <div class="tk-px">${card["live"]:,.2f}
    <span class="{chg_cls}">{card["day_pct"]:+.1%} today</span></div>
  <div class="lvls">
    <div class="lvl"><div class="lab">ENTRY</div>
      <div class="val">${p["entry"]:,.2f}</div></div>
    <div class="lvl"><div class="lab">STOP</div>
      <div class="val val-stop">${p["stop"]:,.2f}</div></div>
    <div class="lvl"><div class="lab">TARGET</div>
      <div class="val" style="color:{accent}">${p["target"]:,.2f}</div></div>
  </div>
  <div class="meta"><b>{p["shares"]} shares</b> ≈ ${p["notional"]:,.0f}
     · risk ${p["risk_dollars"]:,.0f} at stop · {p["reward_risk"]}R
     · flat by <b>{p["time_exit"]}</b></div>
  <div class="meta">gap {card["gap_pct"]:+.1%} · rvol {rvol_txt}
     · ATR {card["atr_pct"]:.1%}</div>
  {scale_html}
  <div class="why"><div class="lab">WHY THIS TRADE</div>
    <ul>{reasons}</ul>
    <div class="meta" style="margin-top:8px">{p["entry_note"]}</div>
  </div>
  {opt_html}
</div>
""", unsafe_allow_html=True)

    section("TODAY · 5-MIN")
    render_intraday(card["symbol"], p, accent, card["prev_close"], card["live"])
    section("PROJECTED SAME-DAY PAYOFF")
    render_payoff(p, o, card["atr"], accent)


def render_no_trade(style_cards: dict) -> None:
    """Explicit skip — near misses listed with the gate each one failed."""
    misses = [m for sc in style_cards.values()
              for m in sc.get("near_misses", [])]
    misses.sort(key=lambda m: -m["score"])
    rows = "".join(
        f'<li><b>{m["symbol"]}</b> ({m["style"]} · score {m["score"]:.2f}) '
        f'— {", ".join(m["failed"])}</li>' for m in misses)
    st.markdown(f"""
<div class="ticket">
  <div class="ticket-rule"></div>
  <span class="badge" style="background:#1E2935;color:#8A97A5">NO TRADE</span>
  <div class="tk-sym" style="font-size:2rem">No trade today</div>
  <div class="tk-name">Nothing cleared the gates — skipping is a position.</div>
  <div class="why"><div class="lab">NEAR MISSES · FAILED GATES</div>
    <ul>{rows}</ul></div>
</div>
""", unsafe_allow_html=True)


if card is not None:
    card["name"] = name_for(card["symbol"])
    render_champion(card)
else:
    render_no_trade(res["style_cards"])

# --------------------------------------------------------------- watchlist ---

st.markdown("##### Ranked watchlist")
rows = []
for sym, r in wl.iterrows():
    dot = AMBER if r["style"] == "momentum" else BLUE
    failed = gates.get(str(sym), [])
    dim = ' style="opacity:.45"' if failed else ""
    rows.append(
        f'<tr{dim}><td><span class="dot" style="background:{dot}"></span>{sym}</td>'
        f'<td>{r["score"]:.2f}</td><td>${r["live"]:,.2f}</td>'
        f'<td>{r["day_pct"]:+.1%}</td>'
        f'<td>{(f"{r.rvol:.1f}×" if pd.notna(r.rvol) else "—")}</td>'
        f'<td>{r["atr_pct"]:.1%}</td><td>{r["rsi2"]:.0f}</td></tr>'
    )
st.markdown(
    '<table class="wl"><tr><th>SYMBOL</th><th>SCORE</th><th>LAST</th>'
    '<th>DAY</th><th>RVOL</th><th>ATR</th><th>RSI2</th></tr>'
    + "".join(rows) + "</table>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="meta" style="margin-top:8px">'
    f'<span class="dot" style="background:{AMBER}"></span>momentum&nbsp;&nbsp;'
    f'<span class="dot" style="background:{BLUE}"></span>mean-reversion'
    f'&nbsp;&nbsp;· dimmed = failed a no-trade gate</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ detail view ---

st.markdown("##### Inspect a symbol")
choice = st.selectbox(
    "watchlist detail", ["—"] + list(wl.index), index=0,
    key="detail_sym", label_visibility="collapsed",
)
if choice and choice != "—":
    render_detail(choice)

# ---------------------------------------------------------------- settings ---

with st.expander("Settings"):
    st.checkbox(
        "Risk-budget sizing (shares = risk $ ÷ stop distance, "
        "still capped at $5,000 notional)",
        value=DEFAULT_SETTINGS["risk_sizing"], key="set_risk_sizing",
    )
    st.number_input(
        "Risk budget per trade ($)", min_value=10.0, max_value=1000.0,
        value=DEFAULT_SETTINGS["risk_budget"], step=5.0, key="set_risk_budget",
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

# ------------------------------------------------------------- diagnostics ---

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
