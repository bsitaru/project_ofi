import numpy as np
import pandas as pd
import constants

from sklearn.metrics import confusion_matrix


def get_heat_ratio(df: pd.DataFrame):
    y_true = df['y_true'].to_numpy()
    y_pred = df['y_pred'].to_numpy()
    y_pred = np.around(y_pred, decimals=constants.ROUNDING_RET)

    y_true = np.sign(y_true)
    y_pred = np.sign(y_pred)
    cnf_matrix = confusion_matrix(y_true, y_pred, normalize=None, labels=[-1., 0., 1.])
    tot_num = np.shape(y_true)[0]
    correct = np.sum(y_pred == y_true)
    ratio = correct / tot_num
    return ratio, correct, tot_num, cnf_matrix
