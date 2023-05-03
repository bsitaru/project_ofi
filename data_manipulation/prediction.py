import os
from datetime import date

import numpy as np
import pandas as pd

import constants

columns = ['date', 'time_now', 'ticker', 'y_true', 'y_pred', 'y_train_std']

c_dtype = {}
c_dtype['date'] = str
c_dtype['time_now'] = int
# c_dtype['time_predict'] = int
c_dtype['ticker'] = str
c_dtype['y_true'] = float
c_dtype['y_pred'] = float
c_dtype['y_train_std'] = float


def empty_df():
    return pd.DataFrame(columns=list(c_dtype.keys()))


def create_prediction_df(d, pred_intervals, y_res):
    lines = [[d, time_now, t, y_true, y_pred, y_train_std] for ((time_now, time_predict), dct) in
             zip(pred_intervals, y_res) for (t, (y_true, y_pred, y_train_std)) in dct.items()]
    df = pd.DataFrame(lines, columns=columns)
    df['y_pred'] = np.around(df['y_pred'], decimals=constants.ROUNDING_RET)
    df['y_train_std'] = np.around(df['y_train_std'], decimals=8)
    return df


def get_prediction_df(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, dtype=c_dtype)
        df['date'] = df['date'].apply(date.fromisoformat)
        return df
    except:
        return empty_df()


def get_all_predictions(folder_path: str) -> pd.DataFrame:
    file_list = os.listdir(folder_path)
    dfs = []
    for file_name in file_list:
        if 'predict' not in file_name:
            continue
        df = get_prediction_df(os.path.join(folder_path, file_name))
        dfs.append(df)

    df = pd.concat(dfs)
    return df
