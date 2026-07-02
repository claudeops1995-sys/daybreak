"""
DAYBREAK journal — frozen decision points and nightly outcome scoring.

Runs headless (GitHub Actions; no Streamlit). The repo is the database:
    journal/YYYY-MM-DD/prelim.json     ~9:35 ET snapshot (context only)
    journal/YYYY-MM-DD/official.json   ~9:45 ET frozen decision point
    journal/YYYY-MM-DD/outcomes.json   nightly scoring vs 5-min bars

Cron is UTC and jittery, so workflows fire early and often; this script
gates on the actual ET clock (DST-proof) and is idempotent — a stage that
already exists for the day is skipped unless --force.

Usage:
    python journal.py capture [--stage auto|prelim|official] [--force]
                              [--prefix dryrun-] [--outdir PATH]
    python journal.py score   [--force] [--prefix dryrun-] [--outdir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

import data_sources as ds
import engine

SCHEMA_VERSION = 1

# ET gates for cron runs (manual --force bypasses them).
PRELIM_WINDOW = (dtime(9, 32), dtime(9, 43))
OFFICIAL_WINDOW = (dtime(9, 44), dtime(10, 15))
SCORE_WINDOW = (dtime(20, 0), dtime(23, 30))


def log(msg: str) -> None:
    print(f"[journal] {msg}", flush=True)


def day_dir(outdir: Path, d) -> Path:
    p = outdir / d.isoformat()
    p.mkdir(parents=True, exist_ok=True)
    return p


def df_records(df: pd.DataFrame) -> list[dict]:
    """NaN-safe JSON records (pandas converts NaN -> null)."""
    return json.loads(df.round(4).reset_index().to_json(orient="records"))


def in_window(now_t: dtime, window: tuple[dtime, dtime]) -> bool:
    return window[0] <= now_t <= window[1]


# ---------------------------------------------------------------- capture ---

def pick_stage(now: datetime) -> str | None:
    if in_window(now.time(), PRELIM_WINDOW):
        return "prelim"
    if in_window(now.time(), OFFICIAL_WINDOW):
        return "official"
    return None


def capture(stage: str, force: bool, prefix: str, outdir: Path) -> int:
    now = engine.now_et()
    if not force:
        if now.weekday() >= 5:
            log("weekend — skipping")
            return 0
        if stage == "auto":
            stage = pick_stage(now)
            if stage is None:
                log(f"outside capture windows ({now:%H:%M} ET) — skipping")
                return 0
        elif stage == "prelim" and not in_window(now.time(), PRELIM_WINDOW):
            log("outside prelim window — skipping")
            return 0
        elif stage == "official" and not in_window(now.time(),
                                                   OFFICIAL_WINDOW):
            log("outside official window — skipping")
            return 0
    elif stage == "auto":
        stage = pick_stage(now) or "official"

    ddir = day_dir(outdir, now.date())
    path = ddir / f"{prefix}{stage}.json"
    if path.exists() and not force:
        log(f"{path.name} already exists — skipping (idempotent)")
        return 0

    log(f"capturing {stage} @ {now:%H:%M:%S} ET")
    res = engine.run_scan(progress=log)

    rec: dict = {
        "schema": SCHEMA_VERSION,
        "date": now.date().isoformat(),
        "stage": stage,
        "forced": bool(force),
        "captured_at_et": now.isoformat(),
    }
    if "error" in res:
        rec["error"] = res["error"]
        rec["diag"] = res.get("diag", {})
    else:
        # Enrich BOTH style champions (run_scan only enriches the overall
        # card) so the journal has each contract at the frozen moment,
        # plus headlines and — when a Claude key exists — a one-line
        # "why it's moving" summary (skipped silently otherwise).
        for style, sc in res["style_cards"].items():
            if sc.get("no_trade"):
                continue
            if sc.get("option") is None:
                engine.enrich_card(sc)
            items = []
            try:
                items = ds.news(sc["symbol"], 3, 48)
                sc["news"] = [{**it, "ts": it["ts"].isoformat()}
                              for it in items]
            except Exception:
                sc["news"] = []
            try:
                why = ds.claude_why(sc["symbol"], sc.get("gap_pct"), items)
                if why:
                    sc["why_moving"] = why
            except Exception:
                pass
        try:
            rec["tape"] = engine.market_tape()
        except Exception:
            rec["tape"] = {}
        rec.update({
            "phase": res["phase"],
            "asof": res["asof"],
            "style_cards": res["style_cards"],
            "gates": res["gates"],
            "watchlist": df_records(res["watchlist"]),
            "settings": engine.DEFAULT_SETTINGS,
            "diag": res["diag"],
        })
        if stage == "official":
            rec["changed_from_prelim"] = _diff_vs_prelim(
                ddir / f"{prefix}prelim.json", res["style_cards"])

    path.write_text(json.dumps(rec, indent=1, default=str),
                    encoding="utf-8")
    log(f"wrote {path}")
    return 0


def _diff_vs_prelim(prelim_path: Path, style_cards: dict) -> dict:
    """Did the champion change between 9:35 and 9:45?"""
    out: dict = {"prelim_found": prelim_path.exists()}
    if not prelim_path.exists():
        return out
    try:
        prelim = json.loads(prelim_path.read_text(encoding="utf-8"))
        for style, sc in style_cards.items():
            now_sym = None if sc.get("no_trade") else sc["symbol"]
            was = prelim.get("style_cards", {}).get(style, {})
            was_sym = None if was.get("no_trade") else was.get("symbol")
            out[style] = {"prelim": was_sym, "official": now_sym,
                          "changed": was_sym != now_sym}
    except Exception as e:  # a corrupt prelim must not block the official
        out["error"] = str(e)
    return out


# ------------------------------------------------------------------ score ---

def _day_bars(symbol: str, d) -> pd.DataFrame:
    bars = yf.download([symbol], period="5d", interval="5m", prepost=False,
                       group_by="ticker", auto_adjust=True, progress=False)
    df = bars[symbol].dropna()
    return df[[ts.date() == d for ts in df.index]]


def simulate(bars: pd.DataFrame, start: datetime, entry: float, stop: float,
             target: float, exit_ts: datetime, shares: int,
             entry_kind: str = "market") -> dict:
    """Walk 5-min bars: fill (per entry_kind), then stop / target / time
    exit. Stop-first on ambiguous bars (both levels inside one bar) — the
    conservative read.

    entry_kind: market = filled at `entry` on the first bar;
    stop_over = filled when a bar trades up through `entry` (stalking
    momentum); limit = filled when a bar trades down to `entry` (stalking
    mean-reversion). No touch by the exit cutoff -> no_fill.
    """
    active = bars[(bars.index >= start) & (bars.index <= exit_ts)]
    if active.empty:
        return {"error": "no bars in trade window"}
    if entry_kind == "stop_over":
        hit = active[active["High"] >= entry]
        if hit.empty:
            return {"no_fill": True, "entry_kind": entry_kind,
                    "exit_reason": "no_fill"}
        active = active[active.index >= hit.index[0]]
    elif entry_kind == "limit":
        hit = active[active["Low"] <= entry]
        if hit.empty:
            return {"no_fill": True, "entry_kind": entry_kind,
                    "exit_reason": "no_fill"}
        active = active[active.index >= hit.index[0]]
    exit_px, reason, exit_at = None, None, None
    for ts, b in active.iterrows():
        if float(b["Low"]) <= stop:
            exit_px, reason, exit_at = stop, "stop", ts
            break
        if float(b["High"]) >= target:
            exit_px, reason, exit_at = target, "target", ts
            break
    if exit_px is None:
        exit_px = float(active["Close"].iloc[-1])
        reason, exit_at = "time", active.index[-1]
    held = active[active.index <= exit_at]
    risk = entry - stop
    r_of = (lambda x: round(x / risk, 2)) if risk > 0 else (lambda x: None)
    return {
        "entry": round(entry, 2), "exit": round(exit_px, 2),
        "entry_kind": entry_kind,
        "filled_at_et": str(active.index[0]),
        "exit_reason": reason, "exit_at_et": str(exit_at),
        "realized_r": r_of(exit_px - entry),
        "mfe_r": r_of(float(held["High"].max()) - entry),
        "mae_r": r_of(entry - float(held["Low"].min())),
        "pnl_stock": round(shares * (exit_px - entry), 0),
    }


def score(force: bool, prefix: str, outdir: Path) -> int:
    now = engine.now_et()
    d = now.date()
    if not force and not in_window(now.time(), SCORE_WINDOW):
        log(f"outside score window ({now:%H:%M} ET) — skipping")
        return 0

    ddir = outdir / d.isoformat()
    official = ddir / f"{prefix}official.json"
    if not official.exists():
        log(f"no {official.name} for {d} — nothing to score")
        return 0
    out_path = ddir / f"{prefix}outcomes.json"
    if out_path.exists() and not force:
        log(f"{out_path.name} already exists — skipping (idempotent)")
        return 0

    rec = json.loads(official.read_text(encoding="utf-8"))
    if "error" in rec:
        log("official card was an error record — nothing to score")
        return 0

    ex_t = engine.exit_time(d)
    outcomes: dict = {"schema": SCHEMA_VERSION, "date": d.isoformat(),
                      "scored_at_et": now.isoformat(), "styles": {}}
    for style, sc in rec.get("style_cards", {}).items():
        if sc.get("no_trade"):
            outcomes["styles"][style] = {"no_trade": True}
            continue
        try:
            outcomes["styles"][style] = _score_card(sc, d, ex_t)
        except Exception as e:  # one bad symbol must not sink the night
            outcomes["styles"][style] = {"error": str(e)}

    out_path.write_text(json.dumps(outcomes, indent=1, default=str),
                        encoding="utf-8")
    log(f"wrote {out_path}")
    return 0


def _score_card(sc: dict, d, ex_t: dtime) -> dict:
    sym, plan = sc["symbol"], sc["plan"]
    bars = _day_bars(sym, d)
    if bars.empty:
        return {"error": "no 5-min bars for the day"}
    tz = bars.index.tz
    decision = datetime.combine(d, dtime(9, 45), tzinfo=tz)
    fill_t = datetime.combine(d, dtime(10, 0), tzinfo=tz)
    exit_ts = datetime.combine(d, ex_t, tzinfo=tz)

    entry, stop, target = plan["entry"], plan["stop"], plan["target"]
    shares = int(plan["shares"])
    # Stalking plans fill via their trigger (stop-over / limit); triggered
    # plans are market entries at the frozen price.
    stalking = plan.get("status") == "stalking"
    kind = plan.get("entry_kind", "market") if stalking else "market"
    out: dict = {"symbol": sym, "style": sc["style"],
                 "plan": {"entry": entry, "stop": stop, "target": target,
                          "shares": shares, "status": plan.get("status"),
                          "entry_kind": kind}}

    # Model basis: from the 9:45 decision point.
    out["model"] = simulate(bars, decision, entry, stop, target,
                            exit_ts, shares, entry_kind=kind)

    # Realistic-fill basis: from ~10:00 ET after operator review. A
    # stalking plan uses the same trigger level; a triggered plan is a
    # market entry at the 10:00 print.
    after = bars[bars.index >= fill_t]
    if after.empty:
        out["fill"] = {"error": "no bars at/after 10:00 ET"}
    else:
        fill_px = float(after["Open"].iloc[0])
        out["fill_ref_price"] = round(fill_px, 2)
        f_entry = entry if stalking else fill_px
        out["fill"] = simulate(bars, after.index[0], f_entry, stop, target,
                               exit_ts, shares, entry_kind=kind)
        if out["model"].get("realized_r") is not None and \
           out["fill"].get("realized_r") is not None:
            out["slippage_r"] = round(out["fill"]["realized_r"]
                                      - out["model"]["realized_r"], 2)

    # Option P&L on the model basis, same valuation the app shows.
    o = sc.get("option")
    if o and "contract" in o and "exit" in out["model"]:
        try:
            v = engine.option_exit_value(o, out["model"]["exit"], on=d)
            out["option"] = {
                "contract": o["contract"], "cost": o["cost"],
                "exit_value": round(v, 0),
                "pnl": round(v - o["cost"], 0),
            }
        except Exception as e:
            out["option"] = {"error": str(e)}
    return out


# ------------------------------------------------------------------- main ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["capture", "score"])
    ap.add_argument("--stage", default="auto",
                    choices=["auto", "prelim", "official"])
    ap.add_argument("--force", action="store_true",
                    help="bypass ET windows and overwrite existing files")
    ap.add_argument("--prefix", default="",
                    help="filename prefix (dryrun- for pipeline tests; "
                         "the app ignores prefixed files)")
    ap.add_argument("--outdir", default=str(Path(__file__).parent / "journal"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if args.command == "capture":
        return capture(args.stage, args.force, args.prefix, outdir)
    return score(args.force, args.prefix, outdir)


if __name__ == "__main__":
    sys.exit(main())
