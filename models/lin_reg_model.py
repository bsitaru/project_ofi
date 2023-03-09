import numpy as np
import pandas as pd
import statsmodels.api as sm

from datetime import date
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from constants import levels_list
import math


def read_csv(file: str, process_df) -> (np.ndarray, np.ndarray):
    df = pd.read_csv(file)
    return process_df(df)


class LinRegModel(ABC):
    def __init__(self):
        self.model = None
        self.results = None
        # self.train_data = self.process_df(train_df)
        # self.test_data = self.process_df(test_df) if test_df is not None else None
        self.tst_txt = ''
        self.tst_r2 = None
        self.lr_model = None
        self.lr_r2 = None
        self.name = None

    def fit(self, train_df) -> ():
        x, y = self.process_df(train_df)
        self.model = sm.OLS(y, sm.add_constant(x))
        self.results = self.model.fit()
        self.lr_model = LinearRegression().fit(x, y)
        self.lr_r2 = self.lr_model.score(x, y)

    def score_test(self, test_df) -> ():
        if test_df is not None:
            x, y = self.process_df(test_df)
            r2 = self.lr_model.score(x, y)
            self.tst_r2 = r2
            self.tst_txt = f'Out-of-sample r^2: {r2}'

    def summary(self) -> str:
        sum_txt = self.results.summary().as_text()
        return '\n'.join([sum_txt, self.tst_txt])

    def df_summary(self, ticker: str, d: date) -> pd.DataFrame:
        results: sm.regression.linear_model.RegressionResults = self.results
        df = {'date': d, 'ticker': ticker, 'name': self.name, 'ins_r2': results.rsquared,
              'adj_r2': results.rsquared_adj, 'oos_r2': self.tst_r2}
        df = pd.DataFrame(df, index=[0])
        return df

    def get_adj_r2(self):
        if math.isnan(self.results.rsquared_adj):
            return self.lr_r2
        return self.results.rsquared_adj
        # return self.results.rsquared_adj

    def get_oos_r2(self):
        return self.tst_r2

    @abstractmethod
    def process_df(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        pass


class SplitOFIModel(LinRegModel):
    def __init__(self, levels: int, return_type: str):
        self.levels = levels
        self.return_col = 'return_now' if return_type == 'current' else 'return_future'
        super().__init__()
        self.name = f"SplitOFI_{levels}_{return_type}"

    def process_df(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        levels = self.levels
        cols = levels_list('ofi_add', levels) + levels_list('ofi_cancel', levels) + levels_list('ofi_trade', levels)
        x = df[cols].to_numpy()
        y = df[[self.return_col]].to_numpy()
        return x, y


class OFIModel(LinRegModel):
    def __init__(self, levels: int, return_type: str):
        self.levels = levels
        self.return_col = 'return_now' if return_type == 'current' else 'return_future'
        super().__init__()
        self.name = f"OFI_{levels}_{return_type}"

    def process_df(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        levels = self.levels
        cols = levels_list('ofi', levels)
        x = df[cols].to_numpy()
        y = df[[self.return_col]].to_numpy()
        return x, y
