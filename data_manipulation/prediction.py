from datetime import date

import numpy as np
import pandas as pd

import constants

columns = ['date', 'time_now', 'time_predict', 'ticker', 'y_true', 'y_pred']

c_dtype = {}
c_dtype['date'] = str
c_dtype['time_now'] = int
c_dtype['time_predict'] = int
c_dtype['ticker'] = str
c_dtype['y_true'] = float
c_dtype['y_pred'] = float


def empty_df():
    return pd.DataFrame(columns=list(c_dtype.keys()))


def create_prediction_df(d, pred_intervals, y_res):
    lines = [[d, time_now, time_predict, t, y_true, y_pred] for ((time_now, time_predict), dct) in
             zip(pred_intervals, y_res) for (t, (y_true, y_pred)) in dct.items()]
    df = pd.DataFrame(lines, columns=columns)
    df['y_pred'] = np.around(df['y_pred'], decimals=constants.ROUNDING_RET)
    return df


def get_prediction_df(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, dtype=c_dtype)
        return df
    except:
        return empty_df()
