import numpy as np
import pandas as pd

import constants


def get_heat_ratio(df: pd.DataFrame):
    y_true = df['y_true'].to_numpy()
    y_pred = df['y_pred'].to_numpy()
    y_pred = np.around(y_pred, decimals=constants.ROUNDING_RET)

    tot_num = np.shape(y_true)[0]
    correct = np.sum(np.sign(y_pred) == np.sign(y_true))
    ratio = correct / tot_num
    return ratio, correct, tot_num
