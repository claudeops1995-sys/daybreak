"""
DAYBREAK — Trade-of-the-Day dashboard (Streamlit)
Run locally:   streamlit run app.py
Deploy free:   share.streamlit.io  ->  point at this repo, main file app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from engine import run_scan

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
    return run_scan()


@st.cache_data(ttl=600, show_spinner=False)
def intraday(symbol: str) -> pd.DataFrame:
    df = yf.download([symbol], period="1d", interval="5m", prepost=True,
                     group_by="ticker", auto_adjust=True, progress=False)
    return df[symbol].dropna()


# ----------------------------------------------------------------- header ---

left, right = st.columns([3, 1])
with left:
    st.markdown(
        '<div class="db-wordmark">DAYBRE<span class="sun">A</span>K</div>'
        '<div class="db-sub">one trade a day · sized under $5,000</div>',
        unsafe_allow_html=True,
    )
with right:
    if st.button("↻ Rescan", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Scanning the S&P 500 — first load takes about a minute…"):
    res = cached_scan()

if "error" in res:
    st.error(res["error"] + " Tap Rescan to try again.")
    st.stop()

card, wl, diag = res["card"], res["watchlist"], res["diag"]

st.markdown(
    f'<span class="db-pill">{card["phase_label"]}</span>&nbsp;'
    f'<span class="db-pill">{card["asof"]}</span>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ trade ticket ---

mom = card["style"] == "momentum"
badge_cls = "badge-mom" if mom else "badge-mr"
accent = AMBER if mom else BLUE
chg_cls = "tk-chg-up" if card["day_pct"] >= 0 else "tk-chg-dn"
p = card["plan"]
rvol_txt = f'{card["rvol"]:.1f}×' if card["rvol"] else "—"
reasons = "".join(f"<li>{r}</li>" for r in card["reasons"])

o = card.get("option")
if o and "contract" in o:
    flags = " · ".join(o["flags"]) if o["flags"] else ""
    opt_html = (
        f'<div class="opt"><div class="lab">OPTION ALTERNATIVE</div>'
        f'<div class="line">{o["contracts"]}× {card["symbol"]} '
        f'${o["strike"]:g}C {o["expiry"]} @ ${o["mid"]:.2f} '
        f'≈ ${o["cost"]:,.0f}</div>'
        f'<div class="line">breakeven ${o["breakeven"]:.2f} · '
        f'{o["dte"]} DTE · OI '
        + (f'{o["open_interest"]:,}' if o["open_interest"] is not None else "—")
        + "</div>"
        + (f'<div class="flags">⚠ {flags}</div>' if flags else "")
        + "</div>"
    )
elif o and "unavailable" in o:
    opt_html = (f'<div class="opt"><div class="lab">OPTION ALTERNATIVE</div>'
                f'<div class="line">{o["unavailable"]}</div></div>')
else:
    opt_html = ""

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
     · risk ${p["risk_dollars"]:,.0f} · {p["reward_risk"]}R
     · flat by <b>{p["time_exit"]}</b></div>
  <div class="meta">gap {card["gap_pct"]:+.1%} · rvol {rvol_txt}
     · ATR {card["atr_pct"]:.1%}</div>
  <div class="why"><div class="lab">WHY THIS TRADE</div>
    <ul>{reasons}</ul>
    <div class="meta" style="margin-top:8px">{p["entry_note"]}</div>
  </div>
  {opt_html}
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------- chart ---

try:
    bars = intraday(card["symbol"])
    if len(bars) and abs(float(bars["Close"].iloc[-1]) / card["live"] - 1) < 0.25:
        fig = go.Figure(go.Candlestick(
            x=bars.index, open=bars["Open"], high=bars["High"],
            low=bars["Low"], close=bars["Close"],
            increasing_line_color=accent, decreasing_line_color="#55606c",
            increasing_fillcolor=accent, decreasing_fillcolor="#3a4552",
        ))
        for y, c, lab in ((p["entry"], TEXT, "entry"),
                          (p["stop"], RED, "stop"),
                          (p["target"], accent, "target")):
            fig.add_hline(y=y, line_color=c, line_width=1, line_dash="dot",
                          annotation_text=lab, annotation_font_color=c,
                          annotation_font_size=10)
        fig.update_layout(
            height=330, margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor=INK, plot_bgcolor=INK,
            font=dict(color=MUTED, size=11),
            xaxis=dict(rangeslider_visible=False, gridcolor=LINE),
            yaxis=dict(gridcolor=LINE, side="right"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
except Exception:
    pass  # a missing chart never blocks the card

# --------------------------------------------------------------- watchlist ---

st.markdown("##### Ranked watchlist")
rows = []
for sym, r in wl.iterrows():
    dot = AMBER if r["style"] == "momentum" else BLUE
    rows.append(
        f'<tr><td><span class="dot" style="background:{dot}"></span>{sym}</td>'
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
    f'<span class="dot" style="background:{BLUE}"></span>mean-reversion</div>',
    unsafe_allow_html=True,
)

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
