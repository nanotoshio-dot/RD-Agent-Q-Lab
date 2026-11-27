# RD-Agent rankings in MetaTrader 5 (demo accounts only)

> **No advantage has been demonstrated.** The ranking comes from the Alpha20 LightGBM baseline of the Qlib factor
> template, which in the tests run so far neither beat the market after costs on the S&P 500 nor passed an
> out-of-sample test on China (see `docs/research/market-validation-2026-11-26.md`). These tools exercise the
> pipeline on a demo account. They are not investment advice, and the experts refuse other account types.

## Files

| File | Purpose |
| --- | --- |
| `gerar_sinais.py` | Trains the baseline on the data in use and writes the top 20 of the last day |
| `atualizar_sinais.sh` | Runs it and copies the file to MetaTrader's common folder (`Common\Files\RD`) |
| `auto_atualizar.sh` | Daily job: rebuilds the US data, audits it, swaps it in, publishes the ranking |
| `auditar_dados.py` | The gate that decides whether rebuilt data may replace the data in use |
| `mql5/Experts/RD/RD_Sombra.mq5` | Records what it would buy; it has no order code |
| `mql5/Experts/RD/RD_Demo.mq5` | Sends orders on a demo hedging account in USD, off until switched on |
| `tests/teste_auto.sh` | Tests the daily job with a fake builder in a temporary folder (about 1 minute) |

Runs in WSL2 with MetaTrader 5 on Windows. Messages are in Portuguese.

## Set up

1. Install the project (`.venv`), clone microsoft/qlib into `~/qlib-src`, put the original Qlib US data in
   `~/.qlib/qlib_data/us_data` (the daily job rebuilds from it, so keep it) and build the provider:

   ```bash
   python -m rdagent.app.us_market_data build --source ~/.qlib/qlib_data/us_data \
       --target ~/.qlib/qlib_data/us_data_2026
   ```
2. In MetaTrader: *File → Open Data Folder*, copy `mql5/Experts/RD` into `MQL5\Experts`, and compile both experts
   in MetaEditor.
3. Publish a ranking: `tools/rd_mt5/atualizar_sinais.sh`.
4. Attach one expert to any chart of a **demo** account.
   - `RD_Sombra` only writes `Common\Files\RD\rd_diario.csv`.
   - `RD_Demo` shows a panel. It sends orders only with *Enviar ordens* (`InpOperarDemo`) set to true and algo
     trading allowed. Defaults: minimum lot, 10% stop, at most 5 positions and 10 buys a day, each position at most
     50% and all positions at most 100% of equity, close a position after 10 days or when its ticker leaves the
     ranking (so a daily update can change positions every day), ignore rankings older than 7 days. It only manages
     its own positions and records orders in `Common\Files\RD\rd_ordens.csv`. Run it on one computer only.
     MetaTrader's *Algo Trading* button enables every expert: turn it off before logging into another account.
   - A file `Common\Files\RD\PARAR` stops **all** order activity of `RD_Demo`, closes and stop placement
     included; open positions keep only the stops already set at the broker.

## Daily update

`auto_atualizar.sh` asks for data up to the last NYSE trading day before today. It builds a new provider (1 to 2
hours, mostly Yahoo downloads), keeps the data in use unless the build finishes and passes the audit, keeps the
previous version as `us_data_2026_anterior`, postpones the swap while research or another reader is running, and
deletes only folders it created. It needs 6 GB free. Logs are in `~/.rdagent/rd_mt5`.

Schedule it from PowerShell (replace the distribution, user and path), at a time before the US market opens:

```powershell
$a = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d <distro> -u <user> -e <path>/tools/rd_mt5/auto_atualizar.sh"
$t = New-ScheduledTaskTrigger -Daily -At 07:00
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 4)
Register-ScheduledTask -TaskName "RD sinais" -Action $a -Trigger $t -Settings $s
```

The task runs while the user is logged on, and the computer must stay awake until it finishes; a missed run starts
when the computer is back. Remove it with `Unregister-ScheduledTask -TaskName "RD sinais" -Confirm:$false`.

**Known limit:** the data builder stops when more than a few current index members (1% of those it needs, at least
2) get no data from Yahoo. Three members already get none (KVUE, PSKY, WBD on 2026-11-26), so a few more would
stop the daily updates until the membership list changes. The window and `auto_build.log` name them.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `RD_PROJETO` | two folders above the scripts | Project root |
| `RD_PYTHON` | `$RD_PROJETO/.venv/bin/python` | Python of the project |
| `RD_DADOS` | `~/.qlib/qlib_data` | Qlib data folder |
| `RD_ESTADO` | `~/.rdagent/rd_mt5` | Logs, lock, list of builds, last ranking |
| `RD_MT5_COMUM` | `%APPDATA%\MetaQuotes\Terminal\Common\Files\RD` | Folder the experts read |
