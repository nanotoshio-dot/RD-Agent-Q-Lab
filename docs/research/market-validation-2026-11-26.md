# Market validation runs, 2026-11-26

Runs on a CPU-only machine with 12 GB of memory, a local 7B model and local Qlib backtests: China CSI300
(data to 2026-11-20) and the US S&P 500 (the rebuilt Yahoo-based provider). All returns below are backtests;
nothing was traded.

## Runs

1. **CI** of `e1346bf5` passed.
2. **S&P 500 baseline** on the rebuilt provider (data to 2026-11-23), test 2021-01-04 to 2026-11-20: IC 0.0006,
   excess return with cost -1.13% a year (without cost +3.62%), maximum drawdown of the excess return -38.4%.
   The previous build of the provider gave IC 0.0027 and -3.5%.
3. **One S&P 500 factor cycle** (59 minutes): 10- and 20-day price change factors. The model defined them with
   the sign inverted, the 20-day code also differs from its own formula, and the coder evaluator accepted both.
   Pruning kept them (t of the rank IC 1.6 and 2.1). The feedback model asked to replace the baseline, claiming a
   better return; the return guard kept the baseline. The comparison in this cycle used unequal feature
   preprocessing (see below).
4. **Model path measurement** (a hand-written GRU, 2 epochs, China, test 2020-2024): 7.8 minutes, about 2
   minutes per epoch and 4.4 GB of memory. IC 0.0118 and excess return +2.97% against +8.47% for the LightGBM
   baseline. The first epoch was the best one. With the loop's suggested settings (early stop 10, 100 epochs)
   this model would run at least 11 epochs (about 26 minutes) and meet the 3600 s limit around epoch 28. No
   model written by the language model was run.
5. **US provider extension to 2026-11-25**: see [US provider](#us-provider).

## Defects found and fixed

- `920038d3` Only `conf_combined_factors.yaml` skipped `RobustZScoreNorm` and `Fillna`, so new factors were
  compared with the baseline on differently prepared features; rows lacking one factor were also dropped once
  earlier factors existed. Runner caches are now keyed by the workspace templates. **Earlier factor results and
  keep/replace decisions are not comparable with later ones; start new runs instead of resuming old sessions.**
- `ff352f8e` Neural training logs with positive losses were filtered out. Backtests now report the best
  boosting round, distinct scores per day and the share of top-k places decided by ties, with a warning when the
  model barely ranks.
- `0ede246a` A single stale Yahoo quote for a delisted ticker was kept as data.

## Backtests repeated after the fix

Same saved factor files, same data and periods. Excess return and drawdown are with cost.

| S&P 500, 2021-01 to 2026-11 | IC | Excess a year | Max drawdown | Best round | Top places decided by ties |
| --- | --- | --- | --- | --- | --- |
| Baseline (unchanged template) | 0.0006 | -1.13% | -38.4% | 1 | 34% |
| Price change factors, before | 0.0061 | -2.15% | -36.7% | 3 | not measured |
| Price change factors, after | 0.0017 | -7.27% | -55.9% | 1 | 30% |

| CSI300, 2020 to 2024 | IC before / after | Excess a year before / after | Max drawdown before / after |
| --- | --- | --- | --- |
| Baseline | 0.0144 / 0.0144 | +8.47% / +8.47% | -8.9% / -8.9% |
| MACD (had been accepted) | 0.0139 / 0.0131 | +9.17% / +8.01% | -16.4% / -17.8% |
| Momentum | 0.0159 / 0.0154 | +5.88% / +5.81% | -13.7% / -9.1% |
| Volatility | 0.0152 / 0.0152 | +6.27% / +4.84% | -18.6% / -19.4% |

- With equal preprocessing, none of the tested factors beats the baseline.
- At the template settings the US LightGBM model barely trains: the baseline stops at the first tree, gives about
  21 distinct scores a day to about 481 names, and its validation error is no lower than predicting zero. US
  differences between runs are weak evidence.
- The China baseline stops at 5 rounds but ranks the names (about 278 distinct scores for 300 names).

## US provider

A first build to 2026-11-25 was stopped when its plan showed that single stale quotes would be kept for three
delisted tickers. A second build (`us_data_2026_v10`) used the fix and the raw files already downloaded. Its
audit: benchmarks complete to 2026-11-25; no misaligned series, non-positive closes or bars of other
securities; 2 open/high/low/close violations, the same plan and the same cut series as the previous build; 499
of 499 priced S&P 500 members on 2026-11-25 (503 in the membership source) and 100 of 100 for the Nasdaq-100;
member days up by 998 (S&P 500) and 200 (Nasdaq-100), two for each current member; no file ends earlier or has
fewer rows than before. KVUE, PSKY and WBD still have no data. This provider has not replaced the one in use.

## Open points

- **LightGBM settings.** The template values come from Qlib's China benchmark. A selection protocol that uses the
  validation period only, with rules fixed before the first fit, was prepared and not run.
- **Test periods have been used many times** (China 2020-2024, US 2021-2026). China 2025-01 to 2026-11 has not
  been used by any saved run.
- **Backtest limits.** The benchmarks are price indexes while stock prices include dividends, which overstates
  the excess return; there is no slippage; the US data lack companies delisted after 2020.
- `model_runner.py` ignores a single accepted factor experiment (`> 1`).
- US market prompts still show China tickers in an output example.
- The coder evaluator can accept wrong factor code, which is then kept as successful coder knowledge.
