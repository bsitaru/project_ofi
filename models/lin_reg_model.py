import numpy as np
import pandas as pd
import statsmodels.api as sm

from datetime import date
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from constants import levels_list


def read_csv(file: str, process_df) -> (np.ndarray, np.ndarray):
    df = pd.read_csv(file)
    return process_df(df)


class LinRegModel(ABC):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        self.model = None
        self.results = None
        self.train_data = self.process_df(train_df)
        self.test_data = self.process_df(test_df) if test_df is not None else None
        self.tst_txt = ''
        self.tst_r2 = None
        self.lr_model = None
        self.name = None

    def fit(self) -> ():
        x, y = self.train_data
        self.model = sm.OLS(y, sm.add_constant(x))
        self.results = self.model.fit()
        self.lr_model = LinearRegression().fit(x, y)

    def score_test(self) -> ():
        if self.test_data:
            x, y = self.test_data
            r2 = self.lr_model.score(x, y)
            self.tst_r2 = r2
            self.tst_txt = f'Out-of-sample r^2: {r2}'

    def summary(self) -> ():
        sum_txt = self.results.summary().as_text()
        return '\n'.join([sum_txt, self.tst_txt])

    def df_summary(self, ticker: str, d: date) -> pd.DataFrame:
        results: sm.regression.linear_model.RegressionResults = self.results
        df = {'date': d, 'ticker': ticker, 'name': self.name, 'ins_r2': results.rsquared,
              'adj_r2': results.rsquared_adj, 'oos_r2': self.tst_r2}
        df = pd.DataFrame(df, index=[0])
        return df

    @abstractmethod
    def process_df(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        pass


class SplitOFIModel(LinRegModel):
    def __init__(self, train_df: pd.DataFrame, levels: int, return_type: str, test_df: pd.DataFrame = None):
        self.levels = levels
        self.return_col = 'return_now' if return_type == 'current' else 'return_future'
        super().__init__(train_df, test_df)
        self.name = f"SplitOFI_{levels}_{return_type}"

    def process_df(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        levels = self.levels
        cols = levels_list('ofi_add', levels) + levels_list('ofi_cancel', levels) + levels_list('ofi_trade', levels)
        x = df[cols].to_numpy()
        y = df[[self.return_col]].to_numpy()
        return x, y


class OFIModel(LinRegModel):
    def __init__(self, train_df: pd.DataFrame, levels: int, return_type: str, test_df: pd.DataFrame = None):
        self.levels = levels
        self.return_col = 'return_now' if return_type == 'current' else 'return_future'
        super().__init__(train_df, test_df)
        self.name = f"OFI_{levels}_{return_type}"

    def process_df(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        levels = self.levels
        cols = levels_list('ofi', levels)
        x = df[cols].to_numpy()
        y = df[[self.return_col]].to_numpy()
        return x, y
