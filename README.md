# Football picks agent

A daily agent that suggests the **5 most favorable NFL / college football bets** against the
current odds (spreads and totals).

## How it works

1. **Games & lines**: ESPN's public API (no key) supplies this week's schedule plus the
   DraftKings spread and total for each game. If `ODDS_API_KEY` is set, The Odds API is used too:
   the market line becomes the multi-book consensus and each pick is shopped for the best book.
2. **Ratings**: every completed game this season feeds opponent-adjusted offense/defense
   ratings (weighted ridge regression, recent games weighted more, blowouts capped).
3. **Edge**: the model's projected margin/total is compared with the line, shrunk 65% back
   toward the market, and turned into a cover probability and expected value at the listed price.
4. **Picks**: one bet per game, positive EV only, top 5 by EV. QB injuries for each pick are
   flagged from ESPN's injury report.
5. **Accountability**: each run grades earlier picks (`reports/record.json`) and re-runs a
   walk-forward backtest of the same rule on this season's closing lines. Both appear in the report.

> The score-only model is weaker than the betting market on average (backtest: higher margin
> error than the closing line). Treat the picks as suggestions, check injury news, and bet responsibly.

## Run

```bash
python -m picks                    # writes reports/YYYY-MM-DD.{md,json} and prints the markdown
python -m picks --leagues nfl --top 3 --no-injuries --no-backtest
python -m unittest                 # tests
```

Python 3.10+, standard library only.

## Configuration

| Setting | Where |
|---|---|
| `ODDS_API_KEY` (optional) | Environment variable; enables multi-book lines |
| Model tuning (home field, shrinkage, recency, sigmas) | `PARAMS` in `picks/model.py` |
| Probability sanity cap | `MAX_PROB` in `picks/agent.py` |

Network hosts used: `site.api.espn.com`, `api.the-odds-api.com`.
