"""Opponent-adjusted offense/defense power ratings and bet evaluation.

Each completed game gives two observations:
    home_pts = mu + hfa/2 + off[home] - def[away]
    away_pts = mu - hfa/2 + off[away] - def[home]
Ratings are fit by weighted ridge least squares (coordinate descent), with
recent games weighted more and blowout margins capped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Per-league tuning. sigma = std-dev of actual result around a good forecast.
PARAMS = {
    "nfl": dict(hfa=1.7, ridge=1.0, half_life=8.0, margin_cap=24.0,
                spread_sigma=13.5, total_sigma=13.0, model_weight=0.35, min_games=4),
    "ncaaf": dict(hfa=2.8, ridge=1.0, half_life=8.0, margin_cap=28.0,
                  spread_sigma=15.5, total_sigma=15.5, model_weight=0.35, min_games=5),
}


@dataclass
class Ratings:
    mu: float
    hfa: float
    off: dict[str, float]
    deff: dict[str, float]
    games: dict[str, int]

    def predict(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        """Return (home margin, total points)."""
        h = 0.0 if neutral else self.hfa
        oh, dh = self.off.get(home, 0.0), self.deff.get(home, 0.0)
        oa, da = self.off.get(away, 0.0), self.deff.get(away, 0.0)
        home_pts = self.mu + h / 2 + oh - da
        away_pts = self.mu - h / 2 + oa - dh
        return home_pts - away_pts, home_pts + away_pts

    def power(self, team: str) -> float:
        return self.off.get(team, 0.0) + self.deff.get(team, 0.0)


def fit(games, league: str, iters: int = 150) -> Ratings:
    p = PARAMS[league]
    if not games:
        return Ratings(22.0, p["hfa"], {}, {}, {})
    max_week = max(_ord(g) for g in games)
    obs = []  # (team, opp, pts, home_adv_sign, weight)
    count: dict[str, int] = {}
    for g in games:
        hs, as_ = g.home_score, g.away_score
        cap = p["margin_cap"]
        if hs - as_ > cap:
            hs = as_ + cap
        elif as_ - hs > cap:
            as_ = hs + cap
        w = 0.5 ** ((max_week - _ord(g)) / p["half_life"])
        s = 0.0 if g.neutral else 1.0
        obs.append((g.home_id, g.away_id, hs, s, w))
        obs.append((g.away_id, g.home_id, as_, -s, w))
        count[g.home_id] = count.get(g.home_id, 0) + 1
        count[g.away_id] = count.get(g.away_id, 0) + 1

    teams = list(count)
    off = {t: 0.0 for t in teams}
    deff = {t: 0.0 for t in teams}
    by_off: dict[str, list] = {t: [] for t in teams}
    by_def: dict[str, list] = {t: [] for t in teams}
    for o in obs:
        by_off[o[0]].append(o)
        by_def[o[1]].append(o)
    hfa = p["hfa"]
    lam = p["ridge"]
    wsum = sum(o[4] for o in obs)
    mu = sum(o[2] * o[4] for o in obs) / wsum
    for _ in range(iters):
        mu = sum((o[2] - o[3] * hfa / 2 - off[o[0]] + deff[o[1]]) * o[4] for o in obs) / wsum
        for t in teams:
            num = sum((o[2] - mu - o[3] * hfa / 2 + deff[o[1]]) * o[4] for o in by_off[t])
            off[t] = num / (sum(o[4] for o in by_off[t]) + lam)
        for t in teams:
            num = sum((mu + o[3] * hfa / 2 + off[o[0]] - o[2]) * o[4] for o in by_def[t])
            deff[t] = num / (sum(o[4] for o in by_def[t]) + lam)
    return Ratings(mu, hfa, off, deff, count)


def _ord(g) -> int:
    # postseason weeks come after regular-season weeks
    return g.week + (30 if g.season_type == 3 else 0)


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def payout(american: int) -> float:
    """Profit per 1 unit staked."""
    return american / 100 if american > 0 else 100 / -american


def breakeven(american: int) -> float:
    return 1 / (1 + payout(american))


def evaluate(game, ratings: Ratings, odds: dict | None = None) -> list[dict]:
    """Candidate bets (both sides of spread and total) with probability and EV."""
    p = PARAMS[game.league]
    odds = odds or game.odds
    model_margin, model_total = ratings.predict(game.home_id, game.away_id, game.neutral)
    out = []
    sp = odds.get("home_spread")
    if sp is not None:
        market_margin = -sp  # market-implied home margin
        blended = market_margin + p["model_weight"] * (model_margin - market_margin)
        p_home = norm_cdf((blended + sp) / p["spread_sigma"])
        for side, prob, price, team, line in (
            ("home", p_home, odds.get("home_price", -110), game.home, sp),
            ("away", 1 - p_home, odds.get("away_price", -110), game.away, -sp),
        ):
            out.append(dict(market="spread", side=side, team=team, line=line, price=price,
                            prob=prob, ev=prob * payout(price) - (1 - prob),
                            model=model_margin, market_value=market_margin,
                            edge_pts=(model_margin - market_margin) * (1 if side == "home" else -1)))
    tot = odds.get("total")
    if tot is not None:
        blended = tot + p["model_weight"] * (model_total - tot)
        p_over = 1 - norm_cdf((tot - blended) / p["total_sigma"])
        for side, prob, price in (("over", p_over, odds.get("over_price", -110)),
                                  ("under", 1 - p_over, odds.get("under_price", -110))):
            out.append(dict(market="total", side=side, team=None, line=tot, price=price,
                            prob=prob, ev=prob * payout(price) - (1 - prob),
                            model=model_total, market_value=tot,
                            edge_pts=(model_total - tot) * (1 if side == "over" else -1)))
    return out
