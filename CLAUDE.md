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

- `engine.py` — pure data layer, no Streamlit imports. Split into
  `fetch_features()` (stage 1: ~500-ticker daily history, the slow half)
  → `scan_market(prefetched=…)` (stage 2: live quotes + ranking) →
  `build_output(scan, settings)` (cheap, pure: gates, plans, per-style
  champions) → `enrich_card()` (network extras for headless callers).
  `run_scan()` composes everything for headless use. Never raises for
  data-shaped reasons (empty frames degrade to the error dict); the two
  bulk yf.download calls retry with exponential backoff (`_retry`).
- `app.py` — all UI. Two cache layers: `cached_features()` (ttl 2700s)
  under `cached_scan()` (ttl 600s) — rescans inside the stage-1 window
  take seconds. `build_output()` re-runs on every rerun so Settings
  changes are instant and never re-trigger a scan. Call sites are
  try/except-guarded.
- `journal.py` — headless capture/scorer run by GitHub Actions. No
  Streamlit imports (same rule as engine).
- `data_sources.py` — ALL external fetches route here. Alpaca (IEX
  real-time quotes/bars, news) and Finnhub (earnings calendar, news
  fallback) when keys exist; **always degrades to yfinance — the app
  must run fine with zero keys**. Every call: try/except + `_get`
  retry/backoff. May lazily READ `st.secrets` in `_secret()` but never
  renders UI. `POLYGON_KEY` is wired but inactive (future historical +
  options upgrade slot).

## Secrets (never in code)

Same names in BOTH places — workflows and dashboard each need them:
- **GitHub → repo → Settings → Secrets and variables → Actions**:
  `ALPACA_KEY_ID`, `ALPACA_SECRET`, `FINNHUB_KEY`, `NTFY_TOPIC`,
  optional `ANTHROPIC_API_KEY`, stub `POLYGON_KEY`.
- **Streamlit Cloud → app → Settings → Secrets** (TOML `KEY = "value"`):
  same names.
Workflows export them as env vars; `data_sources._secret()` checks env
first, then `st.secrets`. Missing keys silently disable that source.

## Journal (repo as database)

- `journal/YYYY-MM-DD/prelim.json` (~9:35 ET), `official.json` (~9:45 ET,
  the frozen decision point outcomes are scored against), `outcomes.json`
  (nightly ~20:30 ET: stop/target/time sequencing with stop-first on
  ambiguous bars, MFE/MAE, realized R for both the 9:45 model entry and a
  ~10:00 ET realistic fill, option P&L).
- Workflows: `.github/workflows/journal-morning.yml` and
  `journal-nightly.yml`. Cron is UTC and jittery → both fire early across
  EDT/EST offsets and `journal.py` gates on the actual ET clock; runs are
  idempotent (existing stage files skip; `--force` overrides).
- **Journal commits trigger Streamlit redeploys** — that is how the app
  sees new journal files (it reads the local `journal/` dir).
- Dry-runs: dispatch with `dry_run=true` → `dryrun-*` files the app
  ignores. Manual dispatch dry-run of BOTH workflows is the acceptance
  test after any workflow/journal change.

## UI component vocabulary (build from these, never inline-invent)

Dawn tokens: INK `#0B0F14` bg · AMBER `#FFB454` momentum · BLUE `#5CC8FF`
mean-reversion · RED `#E5484D` stops/warnings · `#C7D2DC` neutral chart
lines (VWAP, option curve) · Space Grotesk display · IBM Plex Mono
numerals/labels. Spacing scale **4/8/12/16/24**; radii 14 (cards) / 9
(nested blocks, buttons) / 5 (chips).

- **Type roles (only three)**: display = Space Grotesk (`.db-wordmark`,
  `.sym`/`.sym-sm`); label = mono smallcaps (`.eyebrow`, `.chip`, `.lab`,
  `th`, `.db-pill`); numerals/data = mono (`.meta`, `.px-line`,
  `.lvl .val`, `td`, buttons, `.notice`). Descriptive sentences (`.sub`,
  `.why li`, `.foot`) are default sans. No `#####` markdown headings —
  use `section()` → `.eyebrow`.
- **Cards**: `.card` + `.card-rule` (dawn hairline) shell; `card_head()`
  chip row; `levels_html(p, accent, option, atr)` entry/stop/target grid —
  pass the contract and each level also shows the per-contract option
  price (`.ct` sub-line: quoted mid at entry, ≈ modeled exit values at
  stop/target via `option_exit_value`); `plan_meta_html()`; `.opt` nested
  block via `option_block_html()`. Champion, detail, no-trade, and option
  blocks are all assembled from these helpers.
- **Chips** via `chip(label, color, variant)`: solid (style, TRIGGERED),
  outline (STALKING, LATE ENTRY), muted (NO TRADE), warn (EARNINGS).
  Color rides a `--c` CSS custom property.
- **Watchlist rows are `st.button` tap targets** (min 44px, mono,
  `white-space:pre`, unicode `▰▱` score bar); selected row =
  `type="primary"` (amber border). Selection key `detail_sym` is set
  manually — no widget owns it.
- **States**: every degraded fetch path calls `notice()` (dashed inline
  marker); stale quotes via `quote_stale_txt()` (red >15m); the no-trade
  card is a designed moment (`.horizon` glow, display-type statement).
- **Charts family**: `chart_layout()` + `candles()` + `show_chart()` —
  ink bg, LINE grid, mono axis font, right axis, no toolbar; VWAP and
  option curves in neutral `#C7D2DC`; payoff shades profit amber .10 /
  loss red .10; heights 300–340 (phone-first).
- **Motion**: exactly one entrance animation (`db-rise`, .28s) on
  `.card`/`.notice`; `prefers-reduced-motion` disables all of it.
- **390px acceptance**: no horizontal scroll (dense mono surfaces step to
  .72rem under 430px), interactive elements ≥44px, three type roles only.

## Conventions (hold these on every change)

- **Dawn design system**: colors/fonts live in the CSS block at the top of
  `app.py`; build new UI from the component vocabulary above.
- **Mobile-first single column**, max-width 720px, 390px is the primary
  canvas. No sidebars; columns only for small control rows (header,
  prev/next chevrons).
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
