import numpy as np
import pandas as pd

from constants import levels_list

from models.lin_reg_model import LinRegModel, read_csv


class SplitOFICurrentReturn(LinRegModel):
    def __init__(self, train_data_file: str, levels: int, test_data_file: str = None):
        super().__init__()
        self.name = f"SplitOFICurrentReturn_{levels}"
        self.levels = levels
        self.train_data = read_csv(train_data_file, self.process_df)
        if test_data_file is not None:
            self.test_data = read_csv(test_data_file, self.process_df)

    def process_df(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        levels = self.levels
        cols = levels_list('ofi_add', levels) + levels_list('ofi_cancel', levels) + levels_list('ofi_trade', levels)
        x = df[cols].to_numpy()
        y = df[['return_now']].to_numpy()
        return x, y
