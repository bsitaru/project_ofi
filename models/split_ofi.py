import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from constants import levels_list


def get_data_from_df(df: pd.DataFrame, levels: int, **kwargs) -> (np.ndarray, np.ndarray):
    cols = levels_list('ofi_add', levels) + levels_list('ofi_cancel', levels) + levels_list('ofi_trade', levels)
    # cols = levels_list('ofi', levels)
    x = df[cols].to_numpy()
    y = df[['return_now']].to_numpy()
    return x, y


def run_model(train_data_path: str, test_data_path: str, levels: int, **kwargs) -> ():
    def get_data_from_file(path: str):
        df = pd.read_csv(path)
        return get_data_from_df(df, levels)

    train_x, train_y = get_data_from_file(train_data_path)
    test_x, test_y = get_data_from_file(test_data_path)

    model = LinearRegression()
    model.fit(train_x, train_y)

    train_score = model.score(train_x, train_y)
    test_score = model.score(test_x, test_y)

    print(f"Train: {train_score}")
    print(f"Test: {test_score}")
