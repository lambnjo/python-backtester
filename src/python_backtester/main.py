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
    # metrics: list = field(default_factory=lambda: ["sharpe_ratio"])


class Strategy(ABC):
    """abstract class for strategies"""

    @abstractmethod
    def create_indicator(self):
        """Create indicator"""

    @abstractmethod
    def get_optimization_params(self):
        """Optimize parameters"""

    @property
    def get_name(self) -> str:
        """Get strat name"""
        return self.__class__.__name__


class Ma(Strategy):
    """SMA"""

    def generate_signals(self, price, fast_window, slow_window):
        """Logic of strategy"""
        fast_ma = price.rolling(window=fast_window).mean()
        slow_ma = price.rolling(window=slow_window).mean()
        entries = fast_ma >= slow_ma
        exits = fast_ma <= slow_ma
        return entries, exits

    def create_indicator(self):
        return vbt.IndicatorFactory(
            class_name="s1",
            short_name="ma",
            input_names=["price"],
            param_names=["fast_window", "slow_window"],
            output_names=["entries", "exits"],
        ).from_apply_func(self.generate_signals)

    # disable:too-many-arguments
    def run(
        self, price, fast_window, slow_window, param_product, per_column, keep_pd=True
    ):
        """Run custom strategy"""
        return self.create_indicator().run(
            price,
            fast_window=fast_window,
            slow_window=slow_window,
            keep_pd=keep_pd,
            param_product=param_product,
            per_column=per_column,
        )

    def get_optimization_params(self):
        """Get optimal parameters"""
        return {
            "fast_window": [10, 20],
            "slow_window": [50, 56],
            "param_product": True,
            "per_column": False,
        }


class TradingSystem:
    """Trading Simulation Sytem"""

    def __init__(self, config: BacktestConfig, strategy: Strategy):
        self.config = config
        self.strategy = strategy

    def get_metrics(self, pf, strat=True):
        """Get metrics"""
        # l_metrics = list(map(lambda x: getattr(pf, x)(), self.metrics))
        # if strat:
        #     return getattr(pf, "sharpe_ratio")()
        #     return pd.concat(l_metrics, axis=1)

        # return pd.DataFrame([l_metrics], columns=self.metrics)
        return getattr(pf, "sharpe_ratio")()

    def simulate_portfolio(self, price, params=None):
        """Simulate portfolio"""
        if params is None:
            params = self.strategy.get_optimization_params()

        res = self.strategy.run(price, **params)
        pf = vbt.Portfolio.from_signals(
            price, entries=res.entries, exits=res.exits, **self.config.pf_kwargs
        )
        return self.get_metrics(pf, strat=True)

    def simulate_holding_portfolio(self, price):
        """Simulate holding portfolio"""
        pf = vbt.Portfolio.from_holding(price, freq="1d")
        return self.get_metrics(pf)


class WalkForwardOptimization:
    """Walk forward class"""

    def __init__(self, trading_system: TradingSystem):
        self.trading_system = trading_system

    def split_samples(self, price):
        """Split closing prices"""
        return price.vbt.rolling_split(**self.trading_system.config.split_kwargs)

    def get_best_parameters(self, in_perf):
        """Get best parameters from optimization"""
        if not isinstance(in_perf, pd.Series):
            raise ValueError("Not implemented yet to deal with multiple perf indicator")
        best_idx = in_perf[in_perf.groupby("split_idx").idxmax()].index
        out_param = {
            name.split("_", 1)[-1]: best_idx.get_level_values(name).to_numpy()
            for name in best_idx.names
            if name not in ("split_idx")
        }
        out_param["param_product"] = False
        out_param["per_column"] = True
        return out_param

    def run_optimization(self, price):
        """Run optimization"""

        (in_price, _), (out_price, _) = self.split_samples(price)

        # In-sample optimization
        in_perf = self.trading_system.simulate_portfolio(in_price)
        in_perf_median = in_perf.groupby("split_idx").median()
        in_hold_perf = self.trading_system.simulate_holding_portfolio(in_price)

        # Out-sample testing
        out_kwargs = self.get_best_parameters(in_perf)
        out_perf_opt = self.trading_system.simulate_portfolio(out_price)
        out_perf_median = out_perf_opt.groupby("split_idx").median()
        out_perf_test = self.trading_system.simulate_portfolio(out_price, out_kwargs)
        out_hold_perf = self.trading_system.simulate_holding_portfolio(out_price)

        return pd.DataFrame(
            {
                "in_sample_hold": in_hold_perf,
                "in_sample_median": in_perf_median,
                "out_hold_perf": out_hold_perf,
                "out_sample_median": out_perf_median,
                "out_sample_test": out_perf_test.values,
            }
        )


close = btc = vbt.GBMData.download(
    "BTC", start="2014-09-17", end="2021-08-25", freq="1D", seed=42
).get()

conf = BacktestConfig()
strat = Ma()
system = TradingSystem(conf, strat)
walkforward = WalkForwardOptimization(system)
res = walkforward.run_optimization(close)
print(res)
