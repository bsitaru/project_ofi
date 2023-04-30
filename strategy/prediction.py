import numpy as np
import pandas as pd

columns = ['date', 'time_now', 'time_predict', 'ticker', 'y_true', 'y_pred']


def create_prediction_df(d, pred_intervals, y_res):
    lines = [[d, time_now, time_predict, t, y_true, y_pred] for ((time_now, time_predict), dct) in
             zip(pred_intervals, y_res) for (t, (y_true, y_pred)) in dct.items()]
    df = pd.DataFrame(lines, columns=columns)
    return df
