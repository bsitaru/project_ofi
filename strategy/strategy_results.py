import math
from typing import List

import numpy as np
from logging_utils import log
TRADING_DAYS_PER_YEAR = 252
SHARPE_COEF = math.sqrt(float(TRADING_DAYS_PER_YEAR))

class DayResults:
    def __init__(self, values):
        values = np.array(values)
        self.values = values
        self.num = np.size(values)  # number of trades in day

        if self.num == 0:
            self.pnl, self.std, self.ppd = 0, 0, 0
            return

        self.pnl = np.sum(values)  # daily pnl
        self.std = np.std(values)  # std dev of daily pnl
        self.ppd = self.pnl / self.num  # pnl per dollar traded

class StrategyResults:
    def __init__(self, all_res: List[DayResults]):
        self.all_res = all_res
        self.pnls = np.array([r.pnl for r in all_res])
        self.avg_pnl = np.mean(self.pnls)
        self.std_pnl = np.std(self.pnls)

        values = np.concatenate([r.values for r in all_res])
        self.ppd = np.mean(values)
        self.std_ppd = np.std(values)

        self.sharpe = self.avg_pnl / self.std_pnl * SHARPE_COEF
        self.sharpe_ppd = self.ppd / self.std_ppd * SHARPE_COEF

    @staticmethod
    def from_strategy_results_list(results):
        all_res = sum([r.all_res for r in results], [])
        return StrategyResults(all_res)

    def print(self, logger=None):
        lines = [f'avg_pnl: {self.avg_pnl}, std: {self.std_pnl}',
                 f'ppd: {self.ppd}, std: {self.std_ppd}',
                 f'sharpe: {self.sharpe}',
                 f'sharpe_ppd: {self.sharpe_ppd}',
                 f'avg_pnl_bps: {self.avg_pnl * 10000.0}, std: {self.std_pnl * 10000.0}',
                 f'ppd_bps: {self.ppd * 10000.0}, std: {self.std_ppd * 10000.0}',
                 f'annualized_pnl_bps: {self.avg_pnl * TRADING_DAYS_PER_YEAR * 10000.0}']
        for l in lines:
            log(l, logger=logger)


