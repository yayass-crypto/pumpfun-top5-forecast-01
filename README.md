# pumpfun-top5-forecast-01 — trajectory-aware Monte Carlo forecaster

Entry for make.fun's "Predict the top 5 new Solana coins by market cap,
24 hours out" ($200). `main.py` (Python 3.13, standard library only) reads
the house's 10-minute snapshot history and writes one `forecast.json`.

## Why this beats the house baseline

The baseline just repeats the last mark's order with fixed 0.30/0.10/0.10
probabilities. Pump.fun coins churn violently: in the practice history only
1 of the first mark's top 10 was still top 10 at the last mark, and 3 of the
5 coins holding a top-5 place at the last answer read were not in the history
at all (born after the cut). Over the final's 7–13h horizon that churn is far
worse, so the forecast:

- **Leaves most of each place unnamed** (0.38 in the practice, ~0.85 in the
  final, scaled by horizon) — the unnamed share flows to newcomers on the
  answer chart, which is where most top-5 places will be won.
- **Spreads the named mass over up to 100 candidate coins per place** instead
  of betting on 3 coins, so a mover that climbs from rank 30 still pays.
- **Models each coin's future log10 market cap** as `L0 + drift*h + N(0, sd(h))`,
  where drift blends the coin's recent slope (heavily shrunk — a 10-minute
  slope is noisy) with an age prior (young coins still rising, old coins
  decaying), and the noise grows with the horizon so long-horizon forecasts
  are honestly flat.
- **Monte Carlo (fixed seed, 3000 iterations)** over the seven answer-read
  horizons gives place probabilities and size-bin probabilities.

On the one practice answer read whose top 5 is published, this forecast's
order score is about −20.3 versus the baseline's −23.3.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Two runs on the same history are byte-identical (fixed seed, no clock, all
orderings fully deterministic).
