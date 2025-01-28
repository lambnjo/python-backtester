""" Back"""

import vectorbt as vbt
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BacktestConfig:
    """config backtest"""

    split_kwargs: dict = field(
        default_factory=lambda: {
            "n": 30,
            "window_len": 365 * 2,
            "set_lens": (180,),
            "left_to_right": False,
        }
    )
    pf_kwargs: dict = field(default_factory=lambda: {"direction": "both", "freq": "1d"})
    metrics: list = field(default_factory=lambda: ["sharpe_ratio"])
    strat_kwargs: dict = field(
        default_factory=lambda: {"fast_window": [10, 20], "slow_window": [50, 60]}
    )


class Strategy(ABC):
    """abstract class for strategies"""

    @abstractmethod
    def create_indicator(self):
        """Create indicator"""
        pass

    def get_optimization_params(self):
        """Optimize parameters"""
        pass

    @property
    def get_name(self) -> str:
        """Get strat name"""
        return self.__class__.__name__


class Ma(Strategy):
    """SMA"""

    def create_indicator(self):
        return vbt.IndicatorFactory(
            class_name="s1",
            short_name="ma",
            input_names=["price"],
            param_names=["fast_window", "slow_window"],
            output_names=["entries", "exits"],
        ).from_apply_func(self.generate_signals)

    def run(self, price, fast_window, slow_window):
        return self.create_indicator().run(
            price,
            fast_window=fast_window,
            slow_window=slow_window,
            keep_pd=True,
        )

    def generate_signals(self, price, fast_window, slow_window):
        """Logic"""
        fast_ma = price.rolling(window=fast_window).mean()
        slow_ma = price.rolling(window=slow_window).mean()
        entries = fast_ma >= slow_ma
        exits = fast_ma <= slow_ma
        return entries, exits

    def get_optimization_params(self):
        """Get optimal parameters"""
        return {
            "fast_window": [10, 20],
            "slow_window": [50, 56],
        }


# "param_product": True,
#             "per_column": False,
class TradingSystem:

    def __init__(self, config: BacktestConfig, strategy: Strategy):
        self.pf_kwargs = config.pf_kwargs
        self.metrics = config.metrics
        self.strategy = strategy

    def get_metrics(self, pf, strat):
        """Get metrics"""
        # l_metrics = list(map(lambda x: getattr(pf, x)(), self.metrics))
        # if strat:
        #     return getattr(pf, "sharpe_ratio")()
        #     return pd.concat(l_metrics, axis=1)

        # return pd.DataFrame([l_metrics], columns=self.metrics)
        return getattr(pf, "sharpe_ratio")()

    def simulate_portfolio(self, price):
        """Simulate portfolio"""

        params = self.strategy.get_optimization_params()
        res = self.strategy.run(price, **params)
        pf = vbt.Portfolio.from_signals(
            price, entries=res.entries, exits=res.exits, **self.pf_kwargs
        )
        return self.get_metrics(pf, strat=True)

    def simulate_holding_portfolio(self, price):
        """Simulate holding portfolio"""
        pf = vbt.Portfolio.from_holding(price, freq="1d")
        return self.get_metrics(pf, strat=False)


class WalkForwardOptimization:
    """Walk forward class"""

    def __init__(self, config: BacktestConfig, trading_system: TradingSystem):
        self.trading_system = trading_system
        self.split_kwargs = config.split_kwargs
        self.strat_kwargs = config.strat_kwargs

    def split_samples(self, price):
        """Split closing prices"""
        return price.vbt.rolling_split(**self.split_kwargs)

    def run_optimization(self, price):
        """Run optimization"""

        (in_price, _), (out_price, _) = self.split_samples(price)

        # In-sample optimization
        in_perf = self.trading_system.simulate_portfolio(in_price)
        in_perf_median = in_perf.groupby("split_idx").median()
        in_hold_perf = self.trading_system.simulate_holding_portfolio(in_price)
        print(in_perf)
        # # Get best parameters
        # best_idx = in_perf[in_perf.groupby("split_idx").idxmax()].index
        # best_params = {
        #     name: best_idx.get_level_values(name).to_numpy()
        #     for name in param_names
        #     if name not in ["param_product", "per_column"]
        # }
        # best_params.update({"param_product": False, "per_column": True})

        # # Out-sample testing
        # out_perf_opt = self.trading_system.optimize_parameters(out_price)
        # out_perf_median = out_perf_opt.groupby("split_idx").median()
        # out_hold_perf = self.trading_system.simulate_holding(out_price)
        # out_perf_test = self.trading_system.optimize_parameters(out_price, best_params)

        # return pd.DataFrame(
        #     {
        #         "in_sample_hold": in_hold_perf,
        #         "in_sample_median": in_perf_median,
        #         "out_hold_perf": out_hold_perf,
        #         "out_sample_median": out_perf_median,
        #         "out_sample_test": out_perf_test.values,
        #     }
        # )


prices = vbt.GBMData.download(
    "BTC", start="2014-09-17", end="2021-08-25", freq="1D", seed=42
).get()

config = BacktestConfig()
param_strat = {"fast_window": np.arange(1, 50), "slow_window": np.arange(50, 99)}
ma = Ma()

system = TradingSystem(config, ma)
w = WalkForwardOptimization(config, system).run_optimization(prices)
# print(system.simulate_portfolio(prices, entries, exits))
# print(system.simulate_holding_portfolio(prices))
