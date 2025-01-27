"""
Module de backtest flexible supportant différentes stratégies de trading.
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Callable, List
from abc import ABC, abstractmethod


@dataclass
class BacktestConfig:
    """Configuration pour le backtest"""

    split_kwargs: Dict[str, Any] = None
    pf_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.split_kwargs is None:
            self.split_kwargs = {
                "n": 30,
                "window_len": 365 * 2,
                "set_lens": (180,),
                "left_to_right": False,
            }
        if self.pf_kwargs is None:
            self.pf_kwargs = {"direction": "both", "freq": "d"}


class Strategy(ABC):
    """Classe de base abstraite pour toutes les stratégies"""

    def __init__(self):
        self.params = {}
        self.optimal_params = {}

    @abstractmethod
    def generate_signals(
        self, prices: pd.Series, **params
    ) -> Tuple[pd.Series, pd.Series]:
        """Génère les signaux d'entrée et sortie pour la stratégie"""
        pass

    @abstractmethod
    def get_optimization_params(self) -> Dict[str, np.ndarray]:
        """Retourne les paramètres à optimiser"""
        pass

    @property
    def name(self) -> str:
        """Nom de la stratégie"""
        return self.__class__.__name__


class MACrossoverStrategy(Strategy):
    """Stratégie de croisement de moyennes mobiles"""

    def __init__(self, fast_range=(1, 50), slow_range=(50, 99)):
        super().__init__()
        self.fast_range = fast_range
        self.slow_range = slow_range

    def get_optimization_params(self) -> Dict[str, np.ndarray]:
        """
        Définit les paramètres à optimiser.
        Note: Les noms des paramètres doivent correspondre à ceux utilisés dans generate_signals
        """
        return {
            "fast_window": np.arange(self.fast_range[0], self.fast_range[1]),
            "slow_window": np.arange(self.slow_range[0], self.slow_range[1]),
            "param_product": True,
            "per_column": False,
        }

    def generate_signals(
        self, prices: pd.Series, **params
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Génère les signaux basés sur le croisement des moyennes mobiles.
        Les noms des paramètres doivent correspondre à ceux de get_optimization_params
        """
        fast_ma = vbt.MA.run(
            prices,
            window=params["fast_window"],
            short_name="fast",
            param_product=params["param_product"],
            per_column=params["per_column"],
        )
        slow_ma = vbt.MA.run(
            prices,
            window=params["slow_window"],
            short_name="slow",
            param_product=params["param_product"],
            per_column=params["per_column"],
        )
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        return entries, exits


class RSIStrategy(Strategy):
    """Stratégie basée sur le RSI"""

    def __init__(
        self, window_range=(2, 30), overbought_range=(70, 80), oversold_range=(20, 30)
    ):
        super().__init__()
        self.window_range = window_range
        self.overbought_range = overbought_range
        self.oversold_range = oversold_range

    def get_optimization_params(self) -> Dict[str, np.ndarray]:
        return {
            "window": np.arange(self.window_range[0], self.window_range[1]),
            "overbought": np.arange(self.overbought_range[0], self.overbought_range[1]),
            "oversold": np.arange(self.oversold_range[0], self.oversold_range[1]),
            "param_product": True,
            "per_column": False,
        }

    def generate_signals(
        self, prices: pd.Series, **params
    ) -> Tuple[pd.Series, pd.Series]:
        rsi = vbt.RSI.run(
            prices,
            window=params["window"],
            param_product=params["param_product"],
            per_column=params["per_column"],
        )
        print(rsi)
        entries = rsi.rsi_below(params["oversold"], crossover=True)
        print(entries)
        exits = rsi.rsi_above(params["overbought"], crossover=True)
        return entries, exits


class TradingSystem:
    """Système de trading principal"""

    def __init__(self, config: BacktestConfig, strategy: Strategy):
        self.config = config
        self.strategy = strategy
        self.results = {}

    def simulate_portfolio(
        self, prices: pd.Series, entries: pd.Series, exits: pd.Series
    ) -> float:
        pf = vbt.Portfolio.from_signals(
            prices, entries=entries, exits=exits, **self.config.pf_kwargs
        )
        return getattr(pf, "sharpe_ratio")()

    def simulate_holding(self, prices: pd.Series) -> float:
        pf = vbt.Portfolio.from_holding(prices, **self.config.pf_kwargs)
        return getattr(pf, "sharpe_ratio")()

    def optimize_parameters(self, prices: pd.Series, params: Dict = None) -> pd.Series:
        if params is None:
            params = self.strategy.get_optimization_params()
        entries, exits = self.strategy.generate_signals(prices, **params)
        return self.simulate_portfolio(prices, entries, exits)


class WalkForwardOptimization:
    """Classe pour l'optimisation walk-forward"""

    def __init__(self, trading_system: TradingSystem):
        self.trading_system = trading_system

    def split_samples(
        self, prices: pd.Series
    ) -> Tuple[Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series]]:
        return prices.vbt.rolling_split(**self.trading_system.config.split_kwargs)

    def run_optimization(self, prices: pd.Series) -> pd.DataFrame:
        (in_price, in_indexes), (out_price, out_indexes) = self.split_samples(prices)

        # In-sample optimization
        in_perf = self.trading_system.optimize_parameters(in_price)
        in_perf_median = in_perf.groupby("split_idx").median()
        in_hold_perf = self.trading_system.simulate_holding(in_price)
        print(in_perf)
        # Get best parameters
        best_idx = in_perf[in_perf.groupby("split_idx").idxmax()].index
        param_names = list(
            self.trading_system.strategy.get_optimization_params().keys()
        )
        print(param_names)
        print(in_perf)
        best_params = {
            name: best_idx.get_level_values(name).to_numpy()
            for name in param_names
            if name not in ["param_product", "per_column"]
        }
        best_params.update({"param_product": False, "per_column": True})

        # Out-sample testing
        out_perf_opt = self.trading_system.optimize_parameters(out_price)
        out_perf_median = out_perf_opt.groupby("split_idx").median()
        out_hold_perf = self.trading_system.simulate_holding(out_price)
        out_perf_test = self.trading_system.optimize_parameters(out_price, best_params)

        return pd.DataFrame(
            {
                "in_sample_hold": in_hold_perf,
                "in_sample_median": in_perf_median,
                "out_hold_perf": out_hold_perf,
                "out_sample_median": out_perf_median,
                "out_sample_test": out_perf_test.values,
            }
        )


def main():
    """Exemple d'utilisation"""
    # Configuration
    config = BacktestConfig()

    # Création des stratégies
    ma_strategy = MACrossoverStrategy()
    rsi_strategy = RSIStrategy()

    # Test avec MA Crossover
    trading_system_ma = TradingSystem(config, ma_strategy)
    optimizer_ma = WalkForwardOptimization(trading_system_ma)

    # Test avec RSI
    trading_system_rsi = TradingSystem(config, rsi_strategy)
    optimizer_rsi = WalkForwardOptimization(trading_system_rsi)

    # Données
    btc = vbt.GBMData.download(
        "BTC", start="2014-09-17", end="2021-08-25", freq="1D", seed=42
    ).get()

    # Exécution des optimisations
    print("\nRésultats MA Crossover:")
    results_ma = optimizer_ma.run_optimization(btc)
    print(results_ma)

    print("\nRésultats RSI:")
    results_rsi = optimizer_rsi.run_optimization(btc)
    print(results_rsi)


if __name__ == "__main__":
    main()
