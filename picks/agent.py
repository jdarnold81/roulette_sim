"""Daily football picks agent: the 5 most favorable NFL / college bets vs. current odds.

Usage:  python -m picks [--top 5] [--leagues nfl,ncaaf] [--out reports] [--no-injuries]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from . import backtest, espn, model, odds_api

LEAGUE_LABEL = {"nfl": "NFL", "ncaaf": "College"}
MAX_PROB = 0.62  # a bigger "edge" than this almost always means the model is missing news
QB_FLAGS = {"Out", "Doubtful", "Questionable", "Suspension"}


def fmt_line(x: float) -> str:
    return f"{x:+g}" if x else "PK"


def fmt_price(p: int) -> str:
    return f"{p:+d}"


def team_record(team_id: str, history) -> tuple[str, float]:
    w = l = t = 0
    margins = []
    for g in sorted(history, key=lambda g: g.date):
        if team_id not in (g.home_id, g.away_id):
            continue
        m = g.home_score - g.away_score
        m = m if g.home_id == team_id else -m
        margins.append(m)
        w, l, t = w + (m > 0), l + (m < 0), t + (m == 0)
    last3 = sum(margins[-3:]) / max(1, len(margins[-3:]))
    rec = f"{w}-{l}" + (f"-{t}" if t else "")
    return rec, last3


def describe(pick: dict, game, ratings, ranks, history) -> str:
    p = model.PARAMS[game.league]
    if pick["market"] == "spread":
        fav_model = game.home if pick["model"] > 0 else game.away
        fav_mkt = game.home if pick["market_value"] > 0 else game.away
        s = (f"Model makes it {fav_model} by {abs(pick['model']):.1f}; "
             f"the market says {fav_mkt} by {abs(pick['market_value']):.1f}. "
             f"That's a {pick['edge_pts']:.1f}-pt edge toward {pick['team']}.")
    else:
        s = (f"Model projects {pick['model']:.1f} total points vs. a market total of "
             f"{pick['market_value']:g} ({pick['edge_pts']:+.1f} toward the {pick['side']}).")
    bits = []
    for tid, name in ((game.away_id, game.away), (game.home_id, game.home)):
        rec, last3 = team_record(tid, history)
        bits.append(f"{name} {rec}, power #{ranks.get(tid, '?')} "
                    f"(off {ratings.off.get(tid, 0):+.1f}, def {ratings.deff.get(tid, 0):+.1f}), "
                    f"last-3 avg margin {last3:+.1f}")
    s += " " + "; ".join(bits) + "."
    if game.neutral:
        s += " Neutral site."
    s += (f" After shrinking {int((1 - p['model_weight']) * 100)}% back toward the market, "
          f"cover probability is {pick['prob']:.1%} vs. {model.breakeven(pick['price']):.1%} breakeven.")
    return s


def run(leagues: list[str], top: int, out_dir: str, with_injuries: bool, with_backtest: bool = True) -> dict:
    now = datetime.now(timezone.utc)
    use_oa = odds_api.enabled()
    candidates = []
    notes = []
    all_history = {}
    history_by_league = {}
    for lg in leagues:
        upcoming, history, meta = espn.current_and_history(lg)
        all_history.update({g.event_id: g for g in history})
        history_by_league[lg] = history
        ratings = model.fit(history, lg)
        p = model.PARAMS[lg]
        rated = [t for t, n in ratings.games.items() if n >= p["min_games"]]
        ranks = {t: i + 1 for i, t in enumerate(sorted(rated, key=ratings.power, reverse=True))}
        notes.append(f"{LEAGUE_LABEL[lg]}: {meta['season']} week {meta['week']}, "
                     f"{len(history)} completed games rated, {len(upcoming)} upcoming games.")

        oa_events = []
        if use_oa:
            try:
                oa_events = odds_api.fetch(lg)
            except RuntimeError as exc:
                notes.append(f"The Odds API unavailable for {LEAGUE_LABEL[lg]} ({exc}); used ESPN/DraftKings lines.")

        for g in upcoming:
            if min(ratings.games.get(g.home_id, 0), ratings.games.get(g.away_id, 0)) < p["min_games"]:
                continue
            quotes = []
            if oa_events:
                ev = odds_api.match(g, oa_events)
                if ev:
                    quotes = [q for q in odds_api.books(g, ev) if q.get("home_spread") is not None or q.get("total") is not None]
            if quotes:
                market = odds_api.consensus(quotes)
                market_src = f"consensus of {len(quotes)} books"
            elif g.odds:
                market, market_src, quotes = g.odds, g.odds.get("book") or "ESPN", [g.odds]
            else:
                continue
            for c in model.evaluate(g, ratings, market):
                # line-shop: best EV for this market/side across available books
                best = None
                for q in quotes:
                    for qc in model.evaluate(g, ratings, q):
                        if qc["market"] == c["market"] and qc["side"] == c["side"]:
                            if best is None or qc["ev"] > best["ev"]:
                                best = dict(qc, book=q.get("book"))
                if best is None:
                    continue
                c = dict(best, market_src=market_src)
                c.update(league=lg, game=g, ratings=ratings, ranks=ranks, history=history)
                candidates.append(c)

    # one pick per game, positive EV, sanity-capped probability
    best_per_game = {}
    for c in candidates:
        if c["ev"] <= 0 or c["prob"] > MAX_PROB:
            continue
        k = c["game"].event_id
        if k not in best_per_game or c["ev"] > best_per_game[k]["ev"]:
            best_per_game[k] = c
    picks = sorted(best_per_game.values(), key=lambda c: c["ev"], reverse=True)[:top]

    out_picks = []
    for i, c in enumerate(picks, 1):
        g = c["game"]
        if c["market"] == "spread":
            label = f"{c['team']} {fmt_line(c['line'])}"
        else:
            label = f"{c['side'].title()} {c['line']:g}"
        warn = []
        if with_injuries:
            for team, lst in espn.injuries(g.league, g.event_id).items():
                for inj in lst:
                    if inj["pos"] == "QB" and inj["status"] in QB_FLAGS:
                        warn.append(f"{team} QB {inj['name']} ({inj['status']})")
        out_picks.append({
            "rank": i,
            "league": LEAGUE_LABEL[g.league],
            "event_id": g.event_id,
            "game": f"{g.away} at {g.home}" + (" (neutral)" if g.neutral else ""),
            "kickoff_utc": g.date,
            "market": c["market"],
            "side": c["side"],
            "pick": label,
            "line": c["line"],
            "price": c["price"],
            "book": c.get("book"),
            "market_source": c["market_src"],
            "model_value": round(c["model"], 1),
            "market_value": c["market_value"],
            "edge_pts": round(c["edge_pts"], 1),
            "cover_prob": round(c["prob"], 4),
            "ev_per_unit": round(c["ev"], 4),
            "why": describe(c, g, c["ratings"], c["ranks"], c["history"]),
            "qb_watch": warn,
        })

    record = grade(out_dir, all_history)
    bt = None
    if with_backtest:
        try:
            bt = backtest.run(history_by_league, out_dir, top, MAX_PROB)
        except Exception as exc:  # the backtest must never block the daily picks
            notes.append(f"Backtest skipped: {exc}")
    report = {
        "generated_utc": now.isoformat(timespec="seconds"),
        "line_source": "The Odds API (multi-book)" if use_oa else "ESPN (DraftKings)",
        "notes": notes,
        "picks": out_picks,
        "record": record,
        "backtest": bt,
    }
    os.makedirs(out_dir, exist_ok=True)
    stamp = now.strftime("%Y-%m-%d")
    with open(os.path.join(out_dir, f"{stamp}.json"), "w") as f:
        json.dump(report, f, indent=2)
    md = to_markdown(report)
    with open(os.path.join(out_dir, f"{stamp}.md"), "w") as f:
        f.write(md)
    return report


def grade(out_dir: str, results: dict) -> dict:
    """Grade every past pick whose game has finished; returns season-to-date W/L/P."""
    path = os.path.join(out_dir, "record.json")
    try:
        with open(path) as f:
            graded = json.load(f)
    except (OSError, json.JSONDecodeError):
        graded = {}
    if os.path.isdir(out_dir):
        for fn in sorted(os.listdir(out_dir)):
            if not fn.endswith(".json") or fn == "record.json":
                continue
            try:
                with open(os.path.join(out_dir, fn)) as f:
                    rep = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            for p in rep.get("picks", []):
                key = f"{p['event_id']}:{p['market']}:{p['side']}"
                g = results.get(p["event_id"])
                if key in graded or g is None:
                    continue
                margin = g.home_score - g.away_score
                if p["market"] == "spread":
                    v = (margin if p["side"] == "home" else -margin) + p["line"]
                else:
                    v = (g.home_score + g.away_score - p["line"]) * (1 if p["side"] == "over" else -1)
                res = "W" if v > 0 else "L" if v < 0 else "P"
                units = model.payout(p["price"]) if res == "W" else -1.0 if res == "L" else 0.0
                graded[key] = {"pick": p["pick"], "game": p["game"], "date": p["kickoff_utc"],
                               "result": res, "units": round(units, 3)}
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(graded, f, indent=2, sort_keys=True)
    vals = list(graded.values())
    return {
        "W": sum(v["result"] == "W" for v in vals),
        "L": sum(v["result"] == "L" for v in vals),
        "P": sum(v["result"] == "P" for v in vals),
        "units": round(sum(v["units"] for v in vals), 2),
    }


def to_markdown(r: dict) -> str:
    lines = [f"# Football picks — {r['generated_utc'][:10]}", ""]
    rec = r["record"]
    lines.append(f"Lines: {r['line_source']}. Record to date: {rec['W']}-{rec['L']}-{rec['P']} "
                 f"({rec['units']:+.2f} units at 1 unit/pick).")
    if r.get("backtest"):
        lines += ["", backtest.summary(r["backtest"])]
    lines.append("")
    if not r["picks"]:
        lines.append("No bets cleared the bar today (no positive-EV edges among rated games).")
    for p in r["picks"]:
        book = f" @ {p['book']}" if p.get("book") else ""
        lines += [
            f"## {p['rank']}. {p['pick']} ({fmt_price(p['price'])}{book}) — {p['league']}",
            f"**{p['game']}**, kickoff {p['kickoff_utc']}  ",
            f"Edge {p['edge_pts']:+.1f} pts · cover prob {p['cover_prob']:.1%} · "
            f"EV {p['ev_per_unit']:+.3f} units",
            "",
            p["why"],
        ]
        if p["qb_watch"]:
            lines.append("")
            lines.append("⚠️ QB watch: " + "; ".join(p["qb_watch"]) + " — check news before betting.")
        lines.append("")
    lines += ["---", *[f"- {n}" for n in r["notes"]], "",
              "_Model-based suggestions, not guarantees. Bet responsibly._", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--leagues", default="nfl,ncaaf")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--no-injuries", action="store_true")
    ap.add_argument("--no-backtest", action="store_true")
    a = ap.parse_args(argv)
    report = run([x.strip() for x in a.leagues.split(",") if x.strip()], a.top, a.out, not a.no_injuries, not a.no_backtest)
    sys.stdout.write(to_markdown(report))
    return 0
