"""Thin client for ESPN's public site API (no key required)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

BASE = "https://site.api.espn.com/apis/site/v2/sports/football"

LEAGUES = {
    "nfl": {"path": "nfl", "params": {}},
    # groups=80 -> FBS games only (includes FBS vs FCS matchups)
    "ncaaf": {"path": "college-football", "params": {"groups": "80", "limit": "400"}},
}


def get_json(url: str, params: dict | None = None, retries: int = 3) -> dict:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "football-picks-agent/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


@dataclass
class Game:
    league: str
    event_id: str
    name: str
    date: str
    season: int
    season_type: int
    week: int
    neutral: bool
    home_id: str
    home: str
    home_abbr: str
    away_id: str
    away: str
    away_abbr: str
    completed: bool
    state: str  # pre / in / post
    home_score: float | None = None
    away_score: float | None = None
    odds: dict = field(default_factory=dict)  # normalized market, see _parse_odds


def _american(s) -> int | None:
    if s in (None, "", "OFF", "EVEN"):
        return 100 if s == "EVEN" else None
    try:
        return int(str(s).replace("+", ""))
    except ValueError:
        return None


def _line(s) -> float | None:
    if s is None:
        return None
    s = str(s).lstrip("ou").replace("+", "")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_odds(comp: dict) -> dict:
    """Return {'book', 'home_spread', 'home_price', 'away_price', 'total', 'over_price', 'under_price'}."""
    odds_list = comp.get("odds") or []
    if not odds_list:
        return {}
    o = sorted(odds_list, key=lambda x: x.get("provider", {}).get("priority", 99))[0]
    out = {"book": o.get("provider", {}).get("displayName") or o.get("provider", {}).get("name")}
    ps = o.get("pointSpread") or {}
    tot = o.get("total") or {}

    def cur(side: dict) -> dict:
        return side.get("close") or side.get("current") or side.get("open") or {}

    home_sp = _line(cur(ps.get("home", {})).get("line"))
    if home_sp is None and o.get("spread") is not None:
        home_sp = float(o["spread"])
    out["home_spread"] = home_sp
    out["home_price"] = _american(cur(ps.get("home", {})).get("odds")) or -110
    out["away_price"] = _american(cur(ps.get("away", {})).get("odds")) or -110
    total = _line(cur(tot.get("over", {})).get("line"))
    if total is None and o.get("overUnder") is not None:
        total = float(o["overUnder"])
    out["total"] = total
    out["over_price"] = _american(cur(tot.get("over", {})).get("odds")) or -110
    out["under_price"] = _american(cur(tot.get("under", {})).get("odds")) or -110
    return out


def _parse_event(league: str, ev: dict) -> Game | None:
    comp = ev["competitions"][0]
    teams = {c["homeAway"]: c for c in comp["competitors"]}
    if "home" not in teams or "away" not in teams:
        return None
    st = comp["status"]["type"]
    h, a = teams["home"], teams["away"]

    def score(c):
        try:
            return float(c.get("score"))
        except (TypeError, ValueError):
            return None

    return Game(
        league=league,
        event_id=str(ev["id"]),
        name=ev.get("name", ""),
        date=ev.get("date", ""),
        season=int(ev.get("season", {}).get("year", 0)),
        season_type=int(ev.get("season", {}).get("type", 0)),
        week=int(ev.get("week", {}).get("number", 0)),
        neutral=bool(comp.get("neutralSite")),
        home_id=str(h["team"]["id"]),
        home=h["team"].get("displayName", ""),
        home_abbr=h["team"].get("abbreviation", ""),
        away_id=str(a["team"]["id"]),
        away=a["team"].get("displayName", ""),
        away_abbr=a["team"].get("abbreviation", ""),
        completed=bool(st.get("completed")),
        state=st.get("state", ""),
        home_score=score(h),
        away_score=score(a),
        odds=_parse_odds(comp),
    )


def scoreboard(league: str, **extra) -> dict:
    cfg = LEAGUES[league]
    params = dict(cfg["params"])
    params.update({k: str(v) for k, v in extra.items()})
    return get_json(f"{BASE}/{cfg['path']}/scoreboard", params)


def games_from(league: str, data: dict) -> list[Game]:
    out = []
    for ev in data.get("events", []):
        g = _parse_event(league, ev)
        if g:
            out.append(g)
    return out


def current_and_history(league: str) -> tuple[list[Game], list[Game], dict]:
    """Upcoming games for the current week, plus every completed game this season."""
    cur = scoreboard(league)
    season = int(cur.get("season", {}).get("year"))
    stype = int(cur.get("season", {}).get("type", 2))
    week = int(cur.get("week", {}).get("number", 1))
    upcoming = [g for g in games_from(league, cur) if g.state == "pre"]

    history: dict[str, Game] = {}
    weeks: list[tuple[int, int]] = []
    if stype == 2:
        weeks += [(2, w) for w in range(1, week + 1)]
    elif stype == 3:
        weeks += [(2, w) for w in range(1, 19)]
        weeks += [(3, w) for w in range(1, week + 1)]
    for st, wk in weeks:
        try:
            data = scoreboard(league, seasontype=st, week=wk, dates=season)
        except RuntimeError:
            continue
        for g in games_from(league, data):
            if g.completed and g.home_score is not None and g.away_score is not None:
                history[g.event_id] = g
    meta = {"season": season, "season_type": stype, "week": week}
    return upcoming, list(history.values()), meta


def injuries(league: str, event_id: str) -> dict[str, list[dict]]:
    """team displayName -> list of {name, pos, status}."""
    path = LEAGUES[league]["path"]
    try:
        d = get_json(f"{BASE}/{path}/summary", {"event": event_id})
    except RuntimeError:
        return {}
    out: dict[str, list[dict]] = {}
    for t in d.get("injuries", []) or []:
        name = t.get("team", {}).get("displayName", "")
        out[name] = [
            {
                "name": i.get("athlete", {}).get("displayName", ""),
                "pos": (i.get("athlete", {}).get("position") or {}).get("abbreviation", ""),
                "status": i.get("status", ""),
            }
            for i in t.get("injuries", []) or []
        ]
    return out
