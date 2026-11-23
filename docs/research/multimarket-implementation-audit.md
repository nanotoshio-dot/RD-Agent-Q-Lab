# Multi-market implementation audit

**Status:** initial repository audit  
**Working branch:** `feat/multimarket-foundation`  
**Base:** `development` at `b75f69483ec5d4b5ac1e9c46d8f92aa73019ece0`

This project extends Microsoft RD-Agent(Q) and Qlib. It must not reimplement their quantitative research pipeline.

## Reuse, adapt, and develop

| Pipeline stage | Existing implementation | Project action |
| --- | --- | --- |
| Data acquisition | Qlib collectors under `scripts/data_collector/` | Reuse collectors; audit US index history and data coverage before relying on results |
| Data processing | Qlib provider, data handlers, processors, and binary provider | Reuse; configure independent provider paths/calendars per data domain |
| Factor R&D | RD-Agent Qlib factor proposal, coder, runner, and feedback | Reuse; adapt market context and experiment configuration |
| Factor pruning | Factor CoSTEER selection settings and Qlib factor evaluation exist | Do not claim full equivalence to the diagram's pruning stage until selection behavior and factor-correlation/redundancy handling are audited |
| Model R&D/training | RD-Agent model proposal, coder, runner, and model templates | Reuse; adapt market context and experiment configuration |
| Market backtest | Qlib backtest, strategies, and portfolio analysis records | Reuse; validate benchmark, instrument universe, calendar, costs, and date splits for each market |
| Financial analysis/feedback | Qlib signal/portfolio records and RD-Agent experiment feedback | Reuse; add only market-specific reporting that is missing |

## Existing orchestration to reuse

The official RD-Agent(Q) workflow is implemented in:

- `rdagent/app/qlib_rd_loop/quant.py` — `QuantRDLoop` coordinates hypothesis generation, factor/model experiment conversion, coding, running, and feedback.
- `rdagent/app/qlib_rd_loop/conf.py` — factor, model, and quant settings are configurable through environment-prefixed settings.
- `rdagent/scenarios/qlib/proposal/quant_proposal.py` — factor/model action selection and hypothesis context.
- `rdagent/scenarios/qlib/experiment/factor_template/` and `model_template/` — Qlib experiment templates.
- `rdagent/scenarios/qlib/experiment/workspace.py` — Qlib experiment execution and result extraction.

The intended multi-market design is to configure and isolate these existing components, not fork or rewrite their research loop.

## Existing project extension

The `development` branch already contains:

- `rdagent/core/market/context.py` — immutable market and research-period context.
- `rdagent/core/market/process_config.py` — environment-variable mapping for a market-specific process.
- `rdagent/core/market/__init__.py` — public exports.

The market context currently represents one market at a time. The intended initial profiles are:

| market_id | Region | Currency | Qlib region | Universe |
| --- | --- | --- | --- | --- |
| `china` | CN | CNY | cn | CSI 300 |
| `us-sp500` | US | USD | us | S&P 500 |
| `us-nasdaq100` | US | USD | us | Nasdaq 100 |

These are target profiles, not a claim that the underlying datasets or benchmark mappings have already passed validation.

## Known gaps to resolve before research runs

1. **Market profile source of truth:** define and validate the three market profiles without duplicating Qlib's experiment engine.
2. **Workspace rendering:** the published `QlibFBWorkspace.execute()` runs a named Qlib YAML directly. The local Jinja-rendering change discussed during development is not present in the published `development` branch; it must be reconciled deliberately, not overwritten.
3. **US data coverage:** audit historical constituent membership, delisted tickers, failed NASDAQ100 membership dates, and calendar coverage before producing a research provider.
4. **Benchmark mapping:** verify the benchmark instrument exists in each provider and matches the intended index. Do not assume SPY is an appropriate benchmark for both US universes.
5. **Isolation:** each market needs independent workspace, cache, memory, logs, and date settings. Sharing a US price provider is acceptable only if the S&P 500 and Nasdaq 100 universes remain distinct.
6. **Execution integration:** MT5 connectivity, order routing, and live execution are out of scope for this research-foundation phase.

## Validation policy

- Unit tests must validate market context and environment construction without downloading data or calling external APIs.
- GitHub Actions may run lint and offline tests.
- No data collection, model training, agent loops, paid API calls, live trading, or broker connections should be triggered by CI.
- The `development` branch remains the integration target. Work is prepared on a feature branch and reviewed through a pull request.
