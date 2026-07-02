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
NOTIFY_WINDOW = (dtime(9, 44), dtime(10, 25))
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
        try:
            wsb = ds.wsb_map()
        except Exception:
            wsb = {}
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
            # Sentiment columns recorded from day one — display/testing
            # only, never a gate (see CLAUDE.md).
            senti: dict = {}
            try:
                senti.update(ds.stocktwits(sc["symbol"]))
            except Exception:
                pass
            w = wsb.get(sc["symbol"])
            if w:
                senti["wsb_rank"] = w["rank"]
                senti["wsb_mentions"] = w["mentions"]
            sc["sentiment"] = senti
        try:
            rec["tape"] = engine.market_tape()
        except Exception:
            rec["tape"] = {}
        try:
            rec["sources"] = ds.capabilities()
        except Exception:
            pass
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
        if sc.get("plan", {}).get("kind") == "swing" or \
                style == "mean-reversion":
            # Multi-day swing — graded by the open-position tracker, not
            # the same-day simulator.
            outcomes["styles"][style] = {
                "swing": True, "note": "tracked in positions.json"}
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


# -------------------------------------------------------- open positions ---
# MR is a swing trade now: positions live in journal/positions.json and
# the nightly job walks each one forward — stop / RSI-strength exit /
# day-cap — and writes tomorrow's instruction for the morning alert.

def _positions_path(outdir: Path, prefix: str) -> Path:
    return outdir / f"{prefix}positions.json"


def _load_positions(outdir: Path, prefix: str) -> dict:
    p = _positions_path(outdir, prefix)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"open": [], "closed": [], "skipped": [], "paper_mom": []}


def _save_positions(outdir: Path, prefix: str, book: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    _positions_path(outdir, prefix).write_text(
        json.dumps(book, indent=1, default=str), encoding="utf-8")


def _daily(symbol: str) -> pd.DataFrame:
    df = yf.download([symbol], period="3mo", interval="1d",
                     group_by="ticker", auto_adjust=True, progress=False)
    return df[symbol].dropna()


def _update_position(pos: dict, d) -> dict | None:
    """Walk one open position through today's daily bar. Returns a closed
    record (to move to `closed`) or None if it stays open."""
    bars = _daily(pos["symbol"])
    if bars.empty:
        pos["note"] = "no bars — unchanged"
        return None
    if bars.index[-1].date() != d:
        # No bar for today (holiday / run after midnight): mark to the
        # latest bar but run no exit logic — that fires on trade days.
        c = float(bars["Close"].iloc[-1])
        pos["last_close"] = round(c, 2)
        pos["pnl"] = round(int(pos["shares"]) * (c - pos["entry"]), 0)
        held = [ts for ts in bars.index if ts.date()
                >= datetime.fromisoformat(pos["opened"]).date()]
        pos["days_held"] = len(held)
        rsi2 = engine._rsi_last(bars["Close"], 2)
        pos["rsi2"] = round(float(rsi2), 1) if pd.notna(rsi2) else None
        pos["note"] = f"marked as of {bars.index[-1].date().isoformat()}"
        return None
    pos.pop("note", None)
    row = bars.iloc[-1]
    o, lo, c = float(row["Open"]), float(row["Low"]), float(row["Close"])
    stop, entry, shares = pos["stop"], pos["entry"], int(pos["shares"])
    held = [ts for ts in bars.index
            if ts.date() >= datetime.fromisoformat(pos["opened"]).date()]
    pos["days_held"] = len(held)
    pos["last_close"] = round(c, 2)
    pos["pnl"] = round(shares * (c - entry), 0)

    def close(px: float, reason: str) -> dict:
        return {**pos, "exit": round(px, 2), "exit_date": d.isoformat(),
                "exit_reason": reason,
                "pnl": round(shares * (px - entry), 0)}

    # 1) stop logic first — a gap below the stop fills at the open price.
    if o < stop:
        return close(o, "stopped_gap")
    if lo <= stop:
        return close(stop, "stopped")
    # 2) a pending exit set last night executes at today's close.
    if pos.get("pending_exit"):
        return close(c, pos.get("pending_reason", "rsi_exit"))
    # 3) otherwise: tonight's RSI(2) decides tomorrow's instruction.
    rsi2 = engine._rsi_last(bars["Close"], 2)
    pos["rsi2"] = round(float(rsi2), 1) if pd.notna(rsi2) else None
    if pos["rsi2"] is not None and pos["rsi2"] > pos.get("exit_rsi", 65.0):
        pos["pending_exit"] = True
        pos["pending_reason"] = "rsi_exit"
        pos["instruction"] = "SELL AT CLOSE — strength returned"
    elif pos["days_held"] >= pos.get("max_days", 10):
        pos["pending_exit"] = True
        pos["pending_reason"] = "day10"
        pos["instruction"] = f"DAY-{pos.get('max_days', 10)} SELL — time cap"
    else:
        pos["instruction"] = f"HOLD — day {pos['days_held']}"
    return None


def _try_open_from_official(book: dict, rec: dict, d) -> None:
    """Open today's MR signal if its limit filled and we're under the
    concurrent-position cap."""
    sc = (rec.get("style_cards") or {}).get("mean-reversion") or {}
    if sc.get("no_trade") or not sc.get("symbol"):
        return
    sym, plan = sc["symbol"], sc.get("plan", {})
    settings = rec.get("settings", {})
    if any(p["symbol"] == sym for p in book["open"]):
        book["skipped"].append({"date": d.isoformat(), "symbol": sym,
                                "reason": "already held"})
        return
    cap = int(settings.get("mr_max_open", 5))
    if len(book["open"]) >= cap:
        book["skipped"].append({"date": d.isoformat(), "symbol": sym,
                                "reason": "signal only — at capacity"})
        return
    try:
        bars = _daily(sym)
        if bars.empty or bars.index[-1].date() != d:
            return
        row = bars.iloc[-1]
        entry = float(plan["entry"])
        if float(row["Low"]) > entry:
            book["skipped"].append({"date": d.isoformat(), "symbol": sym,
                                    "reason": "limit not touched"})
            return
        # A gap-down open below the limit fills at the (better) open.
        fill = min(entry, float(row["Open"]))
        pos = {
            "symbol": sym, "style": "mean-reversion",
            "entry": round(fill, 2), "stop": float(plan["stop"]),
            "shares": int(plan["shares"]),
            "notional": round(int(plan["shares"]) * fill, 0),
            "opened": d.isoformat(),
            "exit_rsi": float(settings.get("mr_exit_rsi", 65.0)),
            "max_days": int(settings.get("mr_max_days", 10)),
            "pending_exit": False, "instruction": "HOLD — day 1",
        }
        closed = _update_position(pos, d)  # same-day stop is possible
        if closed is not None:
            book["closed"].append(closed)
        else:
            book["open"].append(pos)
    except Exception as e:
        book["skipped"].append({"date": d.isoformat(), "symbol": sym,
                                "reason": f"fill check failed: {e}"})


def _paper_grade_momentum(book: dict, rec: dict, d) -> None:
    """Data-only: paper-grade momentum as a 2–3 day hold nightly (the
    flipped-positive variant is being graded, not traded)."""
    try:
        papers = book.setdefault("paper_mom", [])
        sc = (rec.get("style_cards") or {}).get("momentum") or {}
        if sc.get("symbol") and not sc.get("no_trade"):
            if not any(p["date"] == d.isoformat() for p in papers):
                papers.append({"date": d.isoformat(), "symbol": sc["symbol"],
                               "entry": float(sc["plan"]["entry"]),
                               "closes": []})
        for p in papers:
            if len(p["closes"]) >= 3:
                continue
            bars = _daily(p["symbol"])
            if bars.empty or bars.index[-1].date() != d:
                continue
            opened = datetime.fromisoformat(p["date"]).date()
            if d > opened and (not p["closes"]
                               or p["closes"][-1]["date"] != d.isoformat()):
                c = float(bars["Close"].iloc[-1])
                p["closes"].append({
                    "date": d.isoformat(), "close": round(c, 2),
                    "pnl_pct": round(c / p["entry"] - 1, 4)})
        del papers[:-30]  # keep the last 30 records
    except Exception:
        pass


def positions(force: bool, prefix: str, outdir: Path,
              seed_test: str | None = None) -> int:
    now = engine.now_et()
    d = now.date()
    if not force and not in_window(now.time(), SCORE_WINDOW):
        log(f"outside positions window ({now:%H:%M} ET) — skipping")
        return 0
    book = _load_positions(outdir, prefix)

    if seed_test:
        # Pipeline test: fabricate an already-open position two sessions
        # back so the updater exercises the full path.
        try:
            bars = _daily(seed_test)
            if len(bars) >= 3:
                entry = float(bars["Close"].iloc[-3])
                book["open"].append({
                    "symbol": seed_test, "style": "mean-reversion",
                    "entry": round(entry, 2),
                    "stop": round(entry * 0.92, 2),
                    "shares": int(5000 // entry),
                    "notional": round(int(5000 // entry) * entry, 0),
                    "opened": bars.index[-3].date().isoformat(),
                    "exit_rsi": 65.0, "max_days": 10,
                    "pending_exit": False, "instruction": "HOLD",
                    "seeded_test": True,
                })
                log(f"seeded test position: {seed_test} @ {entry:.2f}")
        except Exception as e:
            log(f"seed failed: {e}")

    still_open = []
    for pos in book["open"]:
        try:
            closed = _update_position(pos, d)
        except Exception as e:
            pos["note"] = f"update failed: {e}"
            closed = None
        if closed is not None:
            book["closed"].append(closed)
            log(f"closed {closed['symbol']} — {closed['exit_reason']} "
                f"{closed['pnl']:+,.0f}")
        else:
            still_open.append(pos)
    book["open"] = still_open

    official = outdir / d.isoformat() / f"{prefix}official.json"
    if official.exists():
        try:
            rec = json.loads(official.read_text(encoding="utf-8"))
            if "error" not in rec:
                _try_open_from_official(book, rec, d)
                _paper_grade_momentum(book, rec, d)
        except Exception as e:
            log(f"official processing failed: {e}")

    book["deployed"] = round(sum(p.get("notional", 0)
                                 for p in book["open"]), 0)
    book["updated_at_et"] = now.isoformat()
    _save_positions(outdir, prefix, book)
    log(f"positions: {len(book['open'])} open, deployed "
        f"${book['deployed']:,.0f}")
    return 0


# ------------------------------------------------------------------ notify ---

def notify(force: bool, prefix: str, outdir: Path) -> int:
    """Morning push: actions first, then new signals, then deployment."""
    now = engine.now_et()
    d = now.date()
    if not force:
        if now.weekday() >= 5:
            log("weekend — skipping notify")
            return 0
        if not in_window(now.time(), NOTIFY_WINDOW):
            log(f"outside notify window ({now:%H:%M} ET) — skipping")
            return 0
    book = _load_positions(outdir, prefix)
    if book.get("last_notified") == d.isoformat() and not force:
        log("already notified today — skipping")
        return 0

    actions, holds = [], []
    for p in book["open"]:
        line = (f'{p["symbol"]}: {p.get("instruction", "HOLD")} '
                f'({p.get("pnl", 0):+,.0f})')
        (actions if p.get("pending_exit") else holds).append(line)
    for c in book.get("closed", [])[-4:]:
        if c.get("exit_date") and (d - datetime.fromisoformat(
                c["exit_date"]).date()).days <= 1:
            actions.append(f'{c["symbol"]}: STOPPED at {c["exit"]} '
                           f'({c["pnl"]:+,.0f})')

    signals = []
    official = outdir / d.isoformat() / f"{prefix}official.json"
    if official.exists():
        try:
            rec = json.loads(official.read_text(encoding="utf-8"))
            for style, sc in (rec.get("style_cards") or {}).items():
                if sc.get("no_trade"):
                    signals.append(f"{style}: no trade")
                    continue
                plan = sc.get("plan", {})
                if plan.get("kind") == "swing":
                    signals.append(
                        f'NEW BUY: {plan.get("shares")} {sc["symbol"]} '
                        f'limit {plan.get("entry")} · stop '
                        f'{plan.get("stop")}')
                else:
                    signals.append(
                        f'DAY TRADE: {sc["symbol"]} '
                        f'{plan.get("entry_note", "")[:60]}')
        except Exception:
            pass

    lines = []
    if actions:
        lines.append("ACTION TODAY:")
        lines += [f"• {a}" for a in actions]
    if holds:
        lines += [f"• {h}" for h in holds]
    if signals:
        lines.append("SIGNALS:")
        lines += [f"• {s}" for s in signals]
    if not lines:
        lines = ["No actions — no open positions, nothing new qualified."]
    lines.append(f'Deployed: ${book.get("deployed", 0):,.0f} in '
                 f'{len(book["open"])} open position'
                 + ("s" if len(book["open"]) != 1 else ""))
    msg = "\n".join(lines)
    sent = ds.ntfy_send(f"DAYBREAK — {d.strftime('%a %b %d')}", msg,
                        priority="high" if actions else "default")
    log(f"ntfy sent: {sent}\n{msg}")
    if sent:
        book["last_notified"] = d.isoformat()
        _save_positions(outdir, prefix, book)
    return 0


# ------------------------------------------------------------------- main ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",
                    choices=["capture", "score", "positions", "notify"])
    ap.add_argument("--stage", default="auto",
                    choices=["auto", "prelim", "official"])
    ap.add_argument("--force", action="store_true",
                    help="bypass ET windows and overwrite existing files")
    ap.add_argument("--prefix", default="",
                    help="filename prefix (dryrun- for pipeline tests; "
                         "the app ignores prefixed files)")
    ap.add_argument("--seed-test", default=None,
                    help="positions only: fabricate an open test position "
                         "for SYMBOL before updating (pipeline dry-runs)")
    ap.add_argument("--outdir", default=str(Path(__file__).parent / "journal"))
    args = ap.parse_args()

    try:  # ops visibility: which data sources this run actually has
        log(f"sources: {ds.capabilities(probe=True)}")
    except Exception:
        pass
    outdir = Path(args.outdir)
    if args.command == "capture":
        return capture(args.stage, args.force, args.prefix, outdir)
    if args.command == "positions":
        return positions(args.force, args.prefix, outdir, args.seed_test)
    if args.command == "notify":
        return notify(args.force, args.prefix, outdir)
    return score(args.force, args.prefix, outdir)


if __name__ == "__main__":
    sys.exit(main())
