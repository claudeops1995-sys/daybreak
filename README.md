# DAYBREAK — Trade of the Day

One champion day-trade idea each session, scanned from the full S&P 500,
sized under $5,000 (stock and option alternative), on a dashboard that
loads fresh data every time you open it on your phone.

**Screen output for the operator's own professional review — not
investment advice.** You make every entry, exit, and suitability call.

---

## Deploy free in ~10 minutes

1. **GitHub** — create a repo (private is fine, e.g. `daybreak`) and upload
   these four files: `app.py`, `engine.py`, `requirements.txt`, `README.md`.
2. **Streamlit Community Cloud** — go to `share.streamlit.io`, sign in with
   GitHub, click **New app**, pick your repo, branch `main`, main file
   `app.py`, hit **Deploy**. No credit card.
3. **iPhone** — open your app's URL in Safari → Share → **Add to Home
   Screen**. It now behaves like a native app over cell service.

Note the first load of the day takes ~60–90 s (full S&P 500 history pull);
after that it's cached for 10 minutes. The free tier sleeps after inactivity
and wakes automatically when you open it (~30 s).

## Daily use

- **Pre-market (before 9:30 ET):** open the app — it scans for the open
  using latest quotes including pre-market prints.
- **During the session:** rescan any time; relative volume is pace-adjusted
  for time of day.
- **Evenings/weekends:** it shows a preview for the next session.
- **Flat by 15:45 ET** is baked into every card — these are same-day ideas.

## How the pick is chosen

**Stage 1 — universe filter.** Full S&P 500 (fetched live; falls back to a
built-in liquid list). Requires price $5–$1,500 and 20-day average dollar
volume ≥ $30M. One year of daily bars per name.

**Stage 2 — live scoring.** Top ~50 candidates get fresh 1-minute quotes
(pre/post included), then two composite scores:

- **Momentum:** gap vs. prior close, relative volume pace, ATR%, proximity
  to 20-day high; must be above the 20-day MA.
- **Mean-reversion:** RSI(2) washout, 3-session pullback, Bollinger
  deviation, ATR%; must be above the 200-day MA (dips in uptrends only).

Highest score across both styles wins the card. Entry is the live
reference; stops and targets are ATR-scaled (0.5×/1.0× momentum,
0.6×/0.8× mean-reversion). Share count = $5,000 ÷ entry. The option
alternative is the nearest-expiry slightly-ITM call that fits under the
cap, with open-interest, spread-width, and 0DTE warnings.

**Data-integrity quarantine.** Around split dates, vendors sometimes serve
adjusted history against unadjusted quotes (a phantom 2–4× "move"). Any
name whose last two closes differ by >1.8× is benched for the day and
listed in Scan diagnostics. Live quotes that disagree with the daily
series by >25% are also discarded.

## Tuning

Everything lives in `CONFIG` at the top of `engine.py`: notional cap,
liquidity floor, stop/target ATR multiples, candidates per style, time
exit. Edit, push to GitHub, and Streamlit redeploys automatically.

## Data source & upgrade path

Yahoo Finance via `yfinance` — free, no API key; quotes can be briefly
delayed. When you want exchange-grade real-time, swap the quote layer for
Alpaca's free IEX websocket or Polygon (~$30/mo); the engine isolates
quotes in `live_snapshot()` so it's a contained change.

## Roadmap ideas

Short setups with put alternatives · earnings-calendar awareness ·
sector-relative scoring · push notification when the morning card is
ready · logging each card to a journal for hit-rate tracking.
