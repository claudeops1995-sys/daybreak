# DAYBREAK — Trade-of-the-Day

Streamlit dashboard (`app.py`) + scan engine (`engine.py`). Deployed on
Streamlit Cloud from `main` (auto-redeploys on push); repo
`claudeops1995-sys/daybreak`, main file `app.py`.

## Commands

```bash
python -m py_compile app.py engine.py   # must pass before commit
python app.py                            # bare-mode smoke test (exit 0, ~1 min, hits Yahoo)
streamlit run app.py                     # local dev
```

Local Windows dev uses `.venv\Scripts\python.exe` (gitignored).

## Architecture

- `engine.py` — pure data layer, no Streamlit imports. `run_scan()` returns
  `{card, watchlist, plans, diag}` or `{"error", "diag"}` — it must never
  raise for data-shaped reasons (empty frames degrade to the error dict).
- `app.py` — all UI. `cached_scan()` wraps `run_scan()`; the call site is
  try/except-guarded so no exception ever reaches the user as a raw page.

## Conventions (hold these on every change)

- **Dawn design system**: colors/fonts live in the CSS block at the top of
  `app.py` (AMBER momentum / BLUE mean-reversion / INK background, Space
  Grotesk + IBM Plex Mono). New UI reuses `.ticket`, `.opt`, `.lvl`,
  `table.wl`, `.meta` classes rather than inventing new styles.
- **Mobile-first single column**, max-width 720px. No sidebars, no
  multi-column layouts beyond the small header row.
- **Plotly only**, `config={"displayModeBar": False}`, `width="stretch"`
  (never `use_container_width` — deprecated, removal imminent).
- **Every network fetch** is `@st.cache_data(ttl=600, show_spinner=False)`
  and its render path is wrapped in try/except — a failed fetch degrades to
  a missing section, never a blank page.
- **Option chains are fetched lazily** — only for the champion during the
  scan and per-symbol when a detail view is opened. Never bulk-fetch chains
  in `run_scan`; scan time stays flat.
- **No scipy** — Black–Scholes is hand-rolled in `engine.py`
  (`bs_call_price`, `_norm_cdf` via `math.erf`).
- **No platform-specific strftime** (`%-I`/`%#I`) — use `_fmt_asof`-style
  manual formatting; code must run on Windows and Linux.
- **Timezone**: all market logic goes through `now_et()` /
  `America/New_York`. VWAP is regular-session only (9:30–16:00 ET).
- **Data-sanity guards stay on every path**: split-quarantine in
  `build_features`, >25% quote-vs-daily mismatch guard in `live_snapshot`
  and `render_intraday`.
- Widget state (`detail_sym`) can outlive a rescan — any lookup keyed by a
  symbol from session state must tolerate the symbol vanishing.
- Commit messages end with the Claude co-author trailer; push to `main`
  triggers the deploy.

## TODO (cosmetic — deferred from 2026-07-01 code review)

- [ ] Champion ticket and detail ticket share ~80% of their HTML — extract a
      common template helper.
- [ ] `quote_time` is collected in `live_snapshot` but never displayed.
- [ ] `rsi14` is computed in `build_features` but unused downstream.
- [ ] `PANEL` color constant in `app.py` is unused (CSS hardcodes #121922).
- [ ] `market_phase` treats US market holidays as normal weekdays.
- [ ] Payoff-chart vlines can sit off-canvas when a mean-reversion stop
      (prev_low) is below −2 ATR; could widen the x-range to include stop.
- [ ] README doesn't yet mention the detail view, payoff projections, or
      per-name option plays.
