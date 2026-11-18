import json
import os
import tempfile
import unittest

from picks import agent, model
from picks.espn import Game, _line, _american


def game(h, a, hs, as_, week=1, neutral=False, league="nfl", eid=None, odds=None):
    return Game(league=league, event_id=eid or f"{h}{a}{week}", name=f"{a} at {h}", date="2026-11-20T01:00Z",
                season=2026, season_type=2, week=week, neutral=neutral, home_id=h, home=h, home_abbr=h,
                away_id=a, away=a, away_abbr=a, completed=True, state="post",
                home_score=hs, away_score=as_, odds=odds or {})


class ModelTests(unittest.TestCase):
    def test_strong_team_rated_higher(self):
        games = []
        for wk in range(1, 9):
            games.append(game("A", "B", 31, 10, wk))
            games.append(game("C", "B", 20, 17, wk))
            games.append(game("A", "C", 28, 14, wk))
        r = model.fit(games, "nfl")
        self.assertGreater(r.power("A"), r.power("C"))
        self.assertGreater(r.power("C"), r.power("B"))
        margin, total = r.predict("A", "B", neutral=True)
        self.assertGreater(margin, 10)
        self.assertGreater(total, 30)

    def test_spread_sides_are_complementary(self):
        r = model.Ratings(22.0, 1.7, {"A": 3, "B": -3}, {"A": 2, "B": -2}, {"A": 9, "B": 9})
        g = game("A", "B", 0, 0, odds={"home_spread": -3.5, "home_price": -110, "away_price": -110,
                                       "total": 44.5, "over_price": -110, "under_price": -110})
        c = {(x["market"], x["side"]): x for x in model.evaluate(g, r)}
        self.assertAlmostEqual(c[("spread", "home")]["prob"] + c[("spread", "away")]["prob"], 1.0)
        self.assertAlmostEqual(c[("total", "over")]["prob"] + c[("total", "under")]["prob"], 1.0)
        # model has home by ~11.7 vs market 3.5 -> home side favored
        self.assertGreater(c[("spread", "home")]["prob"], 0.5)
        self.assertEqual(c[("spread", "away")]["line"], 3.5)

    def test_odds_math(self):
        self.assertAlmostEqual(model.payout(-110), 100 / 110)
        self.assertAlmostEqual(model.payout(150), 1.5)
        self.assertAlmostEqual(model.breakeven(-110), 0.5238, places=4)
        self.assertEqual(_line("o44.5"), 44.5)
        self.assertEqual(_line("+12.5"), 12.5)
        self.assertEqual(_american("+120"), 120)
        self.assertEqual(_american("EVEN"), 100)


class GradeTests(unittest.TestCase):
    def test_grades_spread_and_total(self):
        with tempfile.TemporaryDirectory() as d:
            picks = [
                {"event_id": "1", "market": "spread", "side": "away", "line": 7.0, "price": -110,
                 "pick": "B +7", "game": "B at A", "kickoff_utc": "x"},
                {"event_id": "2", "market": "total", "side": "under", "line": 40.0, "price": -110,
                 "pick": "Under 40", "game": "D at C", "kickoff_utc": "x"},
                {"event_id": "3", "market": "spread", "side": "home", "line": -3.0, "price": -110,
                 "pick": "E -3", "game": "F at E", "kickoff_utc": "x"},
            ]
            with open(os.path.join(d, "2026-11-01.json"), "w") as f:
                json.dump({"picks": picks}, f)
            results = {
                "1": game("A", "B", 24, 20),   # B +7 covers
                "2": game("C", "D", 30, 14),   # 44 > 40, under loses
                "3": game("E", "F", 20, 17),   # push
            }
            rec = agent.grade(d, results)
            self.assertEqual((rec["W"], rec["L"], rec["P"]), (1, 1, 1))
            self.assertAlmostEqual(rec["units"], round(100 / 110 - 1, 2))
            # grading is idempotent
            self.assertEqual(agent.grade(d, results), rec)


if __name__ == "__main__":
    unittest.main()
