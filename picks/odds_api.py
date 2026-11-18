"""Optional multi-book line shopping via The Odds API (needs ODDS_API_KEY)."""

from __future__ import annotations

import difflib
import os
import re
import statistics
from datetime import datetime

from .espn import get_json

SPORT = {"nfl": "americanfootball_nfl", "ncaaf": "americanfootball_ncaaf"}
URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"


def enabled() -> bool:
    return bool(os.environ.get("ODDS_API_KEY"))


def fetch(league: str) -> list[dict]:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return []
    return get_json(URL.format(sport=SPORT[league]), {
        "apiKey": key, "regions": "us", "markets": "spreads,totals", "oddsFormat": "american",
    })


def _norm(s: str) -> str:
    s = s.lower().replace("&", "and").replace("state", "st").replace("st.", "st")
    return re.sub(r"[^a-z0-9 ]", "", s)


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def match(game, events: list[dict]) -> dict | None:
    best, best_score = None, 0.0
    for ev in events:
        try:
            if abs(_ts(ev["commence_time"]) - _ts(game.date)) > 18 * 3600:
                continue
        except (KeyError, ValueError):
            continue
        straight = _sim(game.home, ev["home_team"]) + _sim(game.away, ev["away_team"])
        swapped = _sim(game.home, ev["away_team"]) + _sim(game.away, ev["home_team"])
        score = max(straight, swapped)
        if score > best_score:
            best, best_score = ev, score
    return best if best_score >= 1.3 else None


def books(game, ev: dict) -> list[dict]:
    """Per-book quotes normalized to ESPN-style keys, oriented to the ESPN home team."""
    home_is_home = _sim(game.home, ev["home_team"]) >= _sim(game.home, ev["away_team"])
    oa_home = ev["home_team"] if home_is_home else ev["away_team"]
    out = []
    for bk in ev.get("bookmakers", []):
        q = {"book": bk.get("title", bk.get("key"))}
        for m in bk.get("markets", []):
            if m["key"] == "spreads":
                for o in m["outcomes"]:
                    if o["name"] == oa_home:
                        q["home_spread"], q["home_price"] = float(o["point"]), int(o["price"])
                    else:
                        q["away_price"] = int(o["price"])
            elif m["key"] == "totals":
                for o in m["outcomes"]:
                    if o["name"] == "Over":
                        q["total"], q["over_price"] = float(o["point"]), int(o["price"])
                    elif o["name"] == "Under":
                        q["under_price"] = int(o["price"])
        out.append(q)
    return out


def consensus(quotes: list[dict]) -> dict:
    out: dict = {"book": "consensus"}
    for k in ("home_spread", "total"):
        vals = [q[k] for q in quotes if q.get(k) is not None]
        if vals:
            out[k] = statistics.median(vals)
    for k in ("home_price", "away_price", "over_price", "under_price"):
        out[k] = -110
    return out
