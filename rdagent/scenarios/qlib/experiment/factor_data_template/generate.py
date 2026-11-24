import os

import qlib

# The market launcher sets these; without them this is the original China setup.
qlib.init(
    provider_uri=os.environ.get("RD_MARKET_PROVIDER_URI") or "~/.qlib/qlib_data/cn_data",
    region=os.environ.get("RD_MARKET_REGION") or "cn",
)

from qlib.data import D

# Keep in sync with FACTOR_DATA_START in rdagent/app/market_launcher.py.
DATA_START = "2008-12-29"

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
# Raw fields need no history, so loading from DATA_START gives the same rows as slicing afterwards.
data = (
    D.features(instruments, fields, start_time=DATA_START, freq="day")
    .swaplevel()
    .sort_index()
    .loc[DATA_START:]
    .sort_index()
)
if data.empty:
    raise SystemExit(f"The provider has no price data from {DATA_START}")

data.to_hdf("./daily_pv_all.h5", key="data")


fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
debug_data = (
    D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day").swaplevel().sort_index()
)
# Instruments delisted before the debug window have no rows in it.
available = set(debug_data.index.get_level_values("instrument"))
debug_instruments = [i for i in data.index.get_level_values("instrument").unique() if i in available][:100]
if not debug_instruments:
    raise SystemExit("No instrument has price data in the 2018-2019 debug window")
data = debug_data.swaplevel().loc[debug_instruments].swaplevel().sort_index()

data.to_hdf("./daily_pv_debug.h5", key="data")
