"""
Module de backtest implémentant une stratégie de trading avec Walk Forward Optimization.
Utilise vectorbt pour l'analyse et l'optimisation des paramètres de trading.
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Callable


@dataclass
class BacktestConfig:
    """Configuration pour le backtest"""

    split_kwargs: Dict[str, Any] = None
    pf_kwargs: Dict[str, Any] = None
    param_strat: Dict[str, np.ndarray] = None

    def __post_init__(self):
        # Valeurs par défaut
        if self.split_kwargs is None:
            self.split_kwargs = {
                "n": 30,
                "window_len": 365 * 2,
                "set_lens": (180,),
                "left_to_right": False,
            }
        if self.pf_kwargs is None:
            self.pf_kwargs = {"direction": "both", "freq": "d"}

        if self.param_strat is None:
            self.param_strat = {
                "fast_window": np.arange(1, 50),
                "slow_window": np.arange(50, 99),
                "param_product": True,
                "per_column": False,
            }


class TradingStrategy:
    """Classe principale pour la stratégie de trading"""

    def __init__(self, config: BacktestConfig):
        self.config = config

    @staticmethod
    def ma_crossover_strategy(
        prices: pd.Series,
        fast_window: int,
        slow_window: int,
        param_product: bool,
        per_column: bool,
    ) -> Tuple[pd.Series, pd.Series]:
        """Stratégie de croisement de moyennes mobiles"""
        fast_ma = vbt.MA.run(
            prices,
            window=fast_window,
            short_name="fast",
            param_product=param_product,
            per_column=per_column,
        )
        slow_ma = vbt.MA.run(
            prices,
            window=slow_window,
            short_name="slow",
            param_product=param_product,
            per_column=per_column,
        )
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        return entries, exits

    def generate_signals(
        self, prices: pd.Series, strategy_func: Callable, **strategy_params
    ) -> Tuple[pd.Series, pd.Series]:
        """Génère les signaux d'entrée et de sortie"""
        return strategy_func(prices, **strategy_params)

    def simulate_portfolio(
        self, prices: pd.Series, entries: pd.Series, exits: pd.Series
    ) -> float:
        """Simule le portfolio et retourne le ratio de Sharpe"""
        pf = vbt.Portfolio.from_signals(
            prices, entries=entries, exits=exits, **self.config.pf_kwargs
        )
        return getattr(pf, "sharpe_ratio")()

    def simulate_holding(self, prices: pd.Series) -> float:
        """Simule une stratégie buy & hold"""
        pf = vbt.Portfolio.from_holding(prices, **self.config.pf_kwargs)
        return getattr(pf, "sharpe_ratio")()

    def optimize_parameters(self, prices: pd.Series, param_strat=None) -> pd.Series:
        """Optimise les paramètres sur l'échantillon d'entraînement"""
        print(param_strat)
        if param_strat is None:
            param_strat = self.config.param_strat

        entries, exits = self.generate_signals(
            prices, self.ma_crossover_strategy, **param_strat
        )
        return self.simulate_portfolio(prices, entries, exits)


class WalkForwardOptimization:
    """Classe pour l'optimisation walk-forward"""

    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy

    def split_samples(
        self, prices: pd.Series
    ) -> Tuple[Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series]]:
        """Divise les prix en échantillons in-sample et out-of-sample"""
        return prices.vbt.rolling_split(**self.strategy.config.split_kwargs)

    def run_optimization(self, prices: pd.Series) -> pd.DataFrame:
        """Exécute l'optimisation walk-forward complète"""

        (in_price, in_indexes), (out_price, out_indexes) = self.split_samples(prices)

        # In-sample performance
        in_perf = self.strategy.optimize_parameters(in_price)
        in_perf_median = in_perf.groupby("split_idx").median()
        in_hold_perf = self.strategy.simulate_holding(in_price)

        # Take best parameters from in sample
        best_idx = in_perf[in_perf.groupby("split_idx").idxmax()].index
        in_best_fast = best_idx.get_level_values("fast_window").to_numpy()
        in_best_slow = best_idx.get_level_values("slow_window").to_numpy()

        # Out-sample performance
        out_perf_opt = self.strategy.optimize_parameters(out_price)
        out_perf_median = out_perf_opt.groupby("split_idx").median()
        out_hold_perf = self.strategy.simulate_holding(out_price)

        out_perf_test = self.strategy.optimize_parameters(
            out_price,
            {
                "fast_window": in_best_fast,
                "slow_window": in_best_slow,
                "param_product": False,
                "per_column": True,
            },
        )
        results = pd.DataFrame(
            {
                "in_sample_hold": in_hold_perf,
                "in_sample_median": in_perf_median,
                "out_hold_perf": out_hold_perf,
                "out_sample_median": out_perf_median,
                "out_sample_test": out_perf_test.values,
            }
        )

        return pd.DataFrame(results)


def main():
    """Fonction principale"""
    # Configuration
    config = BacktestConfig()

    # Chargement des données
    btc = vbt.GBMData.download(
        "BTC", start="2014-09-17", end="2021-08-25", freq="1D", seed=42
    ).get()

    # Initialisation des classes
    strategy = TradingStrategy(config)
    optimizer = WalkForwardOptimization(strategy)

    # Exécution de l'optimisation
    results = optimizer.run_optimization(btc)
    print("\nRésultats de l'optimisation walk-forward:")
    print(results)


if __name__ == "__main__":
    main()
