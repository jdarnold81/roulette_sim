import unittest

from picks import odds_api
from picks.espn import Game


def g():
    return Game(league="ncaaf", event_id="1", name="", date="2026-11-21T20:30Z", season=2026, season_type=2,
                week=12, neutral=False, home_id="251", home="Texas Longhorns", home_abbr="TEX",
                away_id="8", away="Arkansas Razorbacks", away_abbr="ARK", completed=False, state="pre")


EVENTS = [
    {"home_team": "Texas A&M Aggies", "away_team": "Auburn Tigers", "commence_time": "2026-11-21T20:30:00Z",
     "bookmakers": []},
    {"home_team": "Texas Longhorns", "away_team": "Arkansas Razorbacks", "commence_time": "2026-11-21T20:30:00Z",
     "bookmakers": [
         {"title": "BookA", "markets": [
             {"key": "spreads", "outcomes": [{"name": "Texas Longhorns", "point": -20.5, "price": -110},
                                             {"name": "Arkansas Razorbacks", "point": 20.5, "price": -110}]},
             {"key": "totals", "outcomes": [{"name": "Over", "point": 51.5, "price": -105},
                                            {"name": "Under", "point": 51.5, "price": -115}]}]},
         {"title": "BookB", "markets": [
             {"key": "spreads", "outcomes": [{"name": "Texas Longhorns", "point": -21.5, "price": -105},
                                             {"name": "Arkansas Razorbacks", "point": 21.5, "price": -115}]}]},
     ]},
]


class OddsApiTests(unittest.TestCase):
    def test_match_and_books(self):
        ev = odds_api.match(g(), EVENTS)
        self.assertEqual(ev["home_team"], "Texas Longhorns")
        quotes = odds_api.books(g(), ev)
        self.assertEqual(quotes[0]["home_spread"], -20.5)
        self.assertEqual(quotes[0]["total"], 51.5)
        self.assertEqual(quotes[1]["away_price"], -115)
        self.assertEqual(odds_api.consensus(quotes)["home_spread"], -21.0)

    def test_no_match_far_in_time(self):
        far = [dict(EVENTS[1], commence_time="2026-12-05T20:30:00Z")]
        self.assertIsNone(odds_api.match(g(), far))


if __name__ == "__main__":
    unittest.main()
