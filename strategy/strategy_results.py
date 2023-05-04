import numpy as np
from logging_utils import log
TRADING_DAYS_PER_YEAR = 250


class StrategyResults:
    def __init__(self, all_profits):
        self.all_profits = np.array(all_profits)
        self.avg_profit = np.mean(self.all_profits)
        self.sum_profit = np.sum(self.all_profits)
        self.std = np.std(self.all_profits)
        self.annual = self.avg_profit * TRADING_DAYS_PER_YEAR
        # log(f"average profit --- {avg_profit} --- std: {std} --- {avg_profit * 10000.0} bps")
        # log(f"sum profit --- {sum_profit} --- {sum_profit * 10000.0} bps")
        # log(f"annualized profit --- {annual} --- {annual * 10000.0} bps")
        # return avg_profit, std, sum_profit, annual

    @staticmethod
    def from_strategy_results_list(results):
        all_profits = np.concatenate([r.all_profits for r in results])
        return StrategyResults(all_profits)

    def print(self, logger=None):
        log(f"average profit --- {self.avg_profit} --- std: {self.std} --- {self.avg_profit * 10000.0} bps", logger=logger)
        log(f"sum profit --- {self.sum_profit} --- {self.sum_profit * 10000.0} bps", logger=logger)
        log(f"annualized profit --- {self.annual} --- {self.annual * 10000.0} bps", logger=logger)
