""" Walk forward optimization"""

import vectorbt as vbt

btc = vbt.GBMData.download(
    "BTC", start="2024-01-01", end="2024-01-02", freq="1min", seed=42
).get()

fast_ma = vbt.MA.run(btc, 10, short_name="fast")
slow_ma = vbt.MA.run(btc, 20, short_name="slow")
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

print(entries)
