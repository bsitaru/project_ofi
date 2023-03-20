import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from models.regression_results import RegressionResults
from data_manipulation.bucket_ofi import compute_normalized_ofi, compute_ofi_df_from_split

from abc import ABC, abstractmethod
from constants import levels_list


class LinRegModel(ABC):
    def __init__(self):
        self.model = None
        self.results = None
        self.os_r2 = None
        self.name = None
        self.col_names = None

    def fit(self, train_df) -> ():
        x, y = self.process_df(train_df)
        self.model = sm.OLS(y, sm.add_constant(x))
        self.results = self.model.fit()

    def score_test(self, test_df) -> ():
        if test_df is not None:
            x, y = self.process_df(test_df)
            ypred = self.results.predict(x)
            r2 = r2_score(y, ypred)
            self.os_r2 = r2

    def get_results(self):
        return RegressionResults.from_lin_reg_results(self.results, self.os_r2, ['intercept'] + self.col_names)

    def run(self, train_df, test_df) -> RegressionResults:
        self.fit(train_df)
        self.score_test(test_df)
        return self.get_results()

    @abstractmethod
    def process_df(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        pass


class BaseOFIModel(LinRegModel):
    def __init__(self, levels: int):
        super().__init__()
        self.levels = levels

    def process_df(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        x = df[self.col_names].to_numpy()
        x = sm.add_constant(x)
        y = df[['return']].to_numpy()
        return x, y

    @staticmethod
    def process_bucket_ofi_df(df: pd.DataFrame) -> pd.DataFrame:
        return compute_normalized_ofi(df)


class SplitOFIModel(BaseOFIModel):
    def __init__(self, levels: int):
        super().__init__(levels)
        self.col_names = levels_list('ofi_add', levels) + levels_list('ofi_cancel', levels) + levels_list('ofi_trade',
                                                                                                          levels)
        self.name = f"SplitOFI_{levels}"


class OFIModel(BaseOFIModel):
    def __init__(self, levels: int):
        super().__init__(levels)
        self.col_names = levels_list('ofi', levels)
        self.name = f"OFI_{levels}"

    @staticmethod
    def process_bucket_ofi_df(df: pd.DataFrame) -> pd.DataFrame:
        df = BaseOFIModel.process_bucket_ofi_df(df)
        return compute_ofi_df_from_split(df)


def model_factory(model_name: str):
    levels = 10
    cls = OFIModel

    if model_name.startswith('OFI'):
        levels = int(model_name[4:])
        cls = OFIModel
    elif model_name.startswith('SplitOFI'):
        levels = int(model_name[9:])
        cls = SplitOFIModel

    class ModelClass(cls):
        def __init__(self):
            super().__init__(levels=levels)

    return ModelClass
