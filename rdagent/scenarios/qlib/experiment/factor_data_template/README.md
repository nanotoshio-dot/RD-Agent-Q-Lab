# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename       | Description                                                      |
| -------------- | -----------------------------------------------------------------|
| "daily_pv.h5"  | Adjusted daily price and volume data.                            |


# For different data, We have some basic knowledge for them

## Daily price and volume data
$open: open price of the stock on that day.
$close: close price of the stock on that day.
$high: high price of the stock on that day.
$low: low price of the stock on that day.
$volume: volume of the stock on that day.
$factor: price adjustment factor for splits and dividends (adjusted price / raw price). The prices above are already adjusted, so $factor is not a trading signal.

## How to use daily_pv.h5
- The index is a MultiIndex with the levels "datetime" and "instrument". They are index levels, not columns.
- Every column name starts with "$": write df["$close"], never df["close"].
- Run rolling windows, shifts and other time-series operations on each instrument separately, for example
  `df.groupby(level="instrument")["$close"].transform(lambda s: s.rolling(20).mean())`.
  A plain `df["$close"].rolling(20)` mixes rows of different stocks.
- Do not call set_index or reset_index: the index is already the ("datetime", "instrument") index that
  result.h5 needs.
- Turn the factor series into the result with `series.to_frame("Factor_Name")`, as below.
  `pd.DataFrame(series, columns=["Factor_Name"])` keeps no data when the series has another name.

## A complete factor.py
```Python
import pandas as pd

df = pd.read_hdf("daily_pv.h5", key="data")  # already indexed by (datetime, instrument)
volume_ma5 = df.groupby(level="instrument")["$volume"].transform(lambda s: s.rolling(5).mean())
volume_ma5.to_frame("Volume_MA5").to_hdf("result.h5", key="data")
```
