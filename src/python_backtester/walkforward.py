""" Walk forward optimization"""

import vectorbt as vbt
import pandas as pd
import numpy as np


def generate_signals(prices, strategy_func, **strategy_params):
    """Generate signals entries and exits based on a strat"""
    return strategy_func(prices, **strategy_params)


def simulate_all_params(prices, strategy_func, strategy_params, pf_kwargs):
    """simulate strategy with different parameters"""
    entries, exits = generate_signals(prices, strategy_func, **strategy_params)
    pf = vbt.Portfolio.from_signals(prices, entries=entries, exits=exits, **pf_kwargs)
    return getattr(pf, "sharpe_ratio")()


def ma_crossover_strategy(prices, fast_window, slow_window):
    """Sma"""
    fast_ma = vbt.MA.run(
        prices, window=fast_window, short_name="fast", param_product=True
    )
    slow_ma = vbt.MA.run(
        prices, window=slow_window, short_name="slow", param_product=True
    )
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    return entries, exits


def roll_in_and_out_samples(prices, kwargs):
    """Rolling and split prices into in and out samples"""
    return prices.vbt.rolling_split(**kwargs)


def simulate_holding(prices, kwargs):
    """Simulate holding"""
    pf = vbt.Portfolio.from_holding(prices, **kwargs)
    return getattr(pf, "sharpe_ratio")()


def simulate_best_params(prices, best_fast_windows, best_slow_windows, kwargs):
    """Simulate with best parameters"""
    fast_ma = vbt.MA.run(prices, window=best_fast_windows, per_column=True)
    slow_ma = vbt.MA.run(prices, window=best_slow_windows, per_column=True)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    pf = vbt.Portfolio.from_holding(prices, **kwargs)
    return getattr(pf, "sharpe_ratio")()


btc = vbt.GBMData.download(
    "BTC", start="2014-09-17", end="2021-08-25", freq="1D", seed=42
).get()
eth = vbt.GBMData.download(
    "ETH", start="2024-01-01", end="2024-01-02", freq="1min", seed=41
).get()
cryptos = pd.concat([btc, eth], axis=1)
cryptos.columns = ["BTC", "ETH"]
price = btc

split_kwargs = {
    "n": 30,
    "window_len": 365 * 2,
    "set_lens": (180,),
    "left_to_right": False,
}
pf_kwargs = {"direction": "both", "freq": "d"}
param_strat = {"fast_window": np.arange(1, 50), "slow_window": np.arange(50, 99)}

(in_price, in_indexes), (out_price, out_indexes) = roll_in_and_out_samples(
    prices=price, kwargs=split_kwargs
)
in_hold_perf = simulate_holding(prices=in_price, kwargs=pf_kwargs)
in_perf = simulate_all_params(
    prices=in_price,
    strategy_func=ma_crossover_strategy,
    strategy_params=param_strat,
    pf_kwargs=pf_kwargs,
)
best_index = in_perf[in_perf.groupby("split_idx").idxmax()].index
in_best_fast = best_index.get_level_values("fast_window").to_numpy()
in_best_slow = best_index.get_level_values("slow_window").to_numpy()

out_hold_perf = simulate_holding(prices=out_price, kwargs=pf_kwargs)

print(out_hold_perf)

out_perf = simulate_best_params(
    prices=out_price,
    best_fast_windows=in_best_fast,
    best_slow_windows=in_best_slow,
    kwargs=pf_kwargs,
)
print(out_perf)
out_perf_opt = simulate_all_params(
    prices=out_price,
    strategy_func=ma_crossover_strategy,
    strategy_params=param_strat,
    pf_kwargs=pf_kwargs,
)
cv_results = pd.DataFrame(
    {
        "in_sample_hold": in_hold_perf.values,
        "in_sample_median": in_perf.groupby("split_idx").median().values,
        "in_sample_best": in_perf[best_index].values,
        "out_sample_hold": out_hold_perf.values,
        "out_sample_median": out_perf_opt.groupby("split_idx").median().values,
        "out_sample_test": out_perf.values,
    }
)
print(cv_results)
