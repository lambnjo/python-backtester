""" in house functions"""

import vectorbt as vbt
import pandas as pd
import numpy as np


def _compute_funding_fees(combined, funding_fee):
    """Funding fees from binance"""
    combined = combined[combined].to_frame(name="signals")
    combined["t"] = combined.index
    combined["tm1"] = combined["t"].shift(1)
    combined = combined.bfill()
    combined["holding_dt"] = combined["t"] - combined["tm1"]
    combined["holding_sec"] = combined["holding_dt"].apply(lambda x: x.total_seconds())
    combined["holding_h"] = combined["holding_sec"] / 3600
    combined["funding_fee"] = funding_fee * combined["holding_h"] / 8
    combined = combined["funding_fee"]
    mask = pd.Series(
        data=[0.0 if i % 2 else np.nan for i in range(1, len(combined) + 1)],
        index=combined.index,
    )
    combined.update(mask)
    return combined.reindex(entries.index, fill_value=0.0).to_frame()


def get_funding_fees(entries, exits, funding_fee=0.0001):
    """Funding fees"""
    combined = entries | exits
    l_funding_fees = []
    for par in combined.columns:
        sub_combined = combined[par]
        fee = _compute_funding_fees(combined=sub_combined, funding_fee=funding_fee)
        fee.columns = pd.MultiIndex.from_tuples([par], names=combined.columns.names)
        l_funding_fees.append(fee)
    return pd.concat(l_funding_fees, axis=1)


btc = vbt.GBMData.download(
    "BTC", start="2024-01-01", end="2024-01-02", freq="1min", seed=42
).get()
eth = vbt.GBMData.download(
    "ETH", start="2024-01-01", end="2024-01-02", freq="1min", seed=41
).get()
cryptos = pd.concat([btc, eth], axis=1)
cryptos.columns = ["BTC", "ETH"]
fast_ma = vbt.MA.run(cryptos, [10, 30, 40, 50], short_name="fast")
slow_ma = vbt.MA.run(cryptos, [20, 60, 23, 454], short_name="slow")
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)
funding_fees = get_funding_fees(entries=entries, exits=exits)
pf = vbt.Portfolio.from_signals(btc, entries=entries, exits=exits, fees=funding_fees)
# print(pf.total_return())
# print()
# print(getattr(pf, "sharpe_ratio"))
for par in funding_fees.columns:
    sub = funding_fees[par]
    print(sub[sub != 0.0])
    print()
