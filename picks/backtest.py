"""Walk-forward backtest of the agent's exact pick rule on this season's closing lines.

For each week, ratings are fit only on games played before that week, every game is
evaluated against its closing DraftKings line (from ESPN's game summary), and the
top-N picks per calendar week are graded. Lines are cached in <out>/lines_cache.json.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from . import espn, model

MIN_WEEK = 4  # skip the first weeks: too little data to rate teams


def _closing_line(league: str, event_id: str) -> dict | None:
    try:
        d = espn.get_json(f"{espn.BASE}/{espn.LEAGUES[league]['path']}/summary", {"event": event_id})
    except RuntimeError:
        return None
    for p in d.get("pickcenter") or []:
        if p.get("spread") is not None or p.get("overUnder") is not None:
            return {"home_spread": p.get("spread"), "total": p.get("overUnder")}
    return {}


def load_lines(history_by_league: dict, out_dir: str) -> dict:
    path = os.path.join(out_dir, "lines_cache.json")
    try:
        with open(path) as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        cache = {}
    todo = [(lg, g.event_id) for lg, hist in history_by_league.items()
            for g in hist if g.week >= MIN_WEEK and g.event_id not in cache]
    if todo:
        with ThreadPoolExecutor(12) as ex:
            for (lg, eid), line in zip(todo, ex.map(lambda t: _closing_line(*t), todo)):
                if line is not None:
                    cache[eid] = line
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(cache, f)
    return cache


def run(history_by_league: dict, out_dir: str, top: int, max_prob: float) -> dict:
    lines = load_lines(history_by_league, out_dir)
    by_cal_week: dict[str, list] = {}
    for lg, hist in history_by_league.items():
        p = model.PARAMS[lg]
        for w in sorted({g.week for g in hist if g.season_type == 2}):
            if w < MIN_WEEK:
                continue
            ratings = model.fit([g for g in hist if g.season_type == 2 and g.week < w], lg, iters=80)
            for g in hist:
                if g.week != w or g.season_type != 2:
                    continue
                if min(ratings.games.get(g.home_id, 0), ratings.games.get(g.away_id, 0)) < p["min_games"]:
                    continue
                line = lines.get(g.event_id) or {}
                if line.get("home_spread") is None and line.get("total") is None:
                    continue
                best = None
                for c in model.evaluate(g, ratings, dict(line, home_price=-110, away_price=-110,
                                                          over_price=-110, under_price=-110)):
                    if c["ev"] > 0 and c["prob"] <= max_prob and (best is None or c["ev"] > best["ev"]):
                        best = c
                if best:
                    iso = datetime.fromisoformat(g.date.replace("Z", "+00:00")).isocalendar()
                    by_cal_week.setdefault(f"{iso[0]}-W{iso[1]:02d}", []).append((best, g))

    res = {"W": 0, "L": 0, "P": 0, "units": 0.0, "weeks": 0,
           "spread": [0, 0], "total": [0, 0], "nfl": [0, 0], "ncaaf": [0, 0]}
    for wk, cands in sorted(by_cal_week.items()):
        res["weeks"] += 1
        for c, g in sorted(cands, key=lambda x: x[0]["ev"], reverse=True)[:top]:
            margin = g.home_score - g.away_score
            if c["market"] == "spread":
                v = (margin if c["side"] == "home" else -margin) + c["line"]
            else:
                v = (g.home_score + g.away_score - c["line"]) * (1 if c["side"] == "over" else -1)
            if v == 0:
                res["P"] += 1
                continue
            win = v > 0
            res["W" if win else "L"] += 1
            res["units"] += model.payout(-110) if win else -1.0
            for k in (c["market"], g.league):
                res[k][0 if win else 1] += 1
    res["units"] = round(res["units"], 2)
    n = res["W"] + res["L"]
    res["win_pct"] = round(res["W"] / n, 3) if n else None
    return res


def summary(res: dict) -> str:
    if not res or not (res["W"] + res["L"]):
        return "Backtest: not enough completed games with lines yet."
    return (f"Backtest this season (same rule, top picks each week vs. closing DraftKings lines at -110): "
            f"{res['W']}-{res['L']}-{res['P']} ({res['win_pct']:.1%}, {res['units']:+.2f} units over {res['weeks']} weeks; "
            f"breakeven is 52.4%). Spreads {res['spread'][0]}-{res['spread'][1]}, totals {res['total'][0]}-{res['total'][1]}; "
            f"NFL {res['nfl'][0]}-{res['nfl'][1]}, college {res['ncaaf'][0]}-{res['ncaaf'][1]}.")
