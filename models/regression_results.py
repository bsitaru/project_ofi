import os
from typing import Union, List

import numpy as np
import pickle
import statsmodels.api as sm


class RegressionResults:
    def __init__(self, in_r2: float, os_r2: float, param_values: list[float], tvalues: list[float],
                 pvalues: list[float]):
        vals = [in_r2, os_r2] + list(param_values) + list(tvalues) + list(pvalues)
        self.values = np.array(vals)

    @staticmethod
    def from_lin_reg_results(results: sm.regression.linear_model.OLSResults, os_r2: float):
        return RegressionResults(in_r2=results.rsquared_adj, os_r2=os_r2, param_values=results.params,
                                 tvalues=results.tvalues, pvalues=results.pvalues)

    @staticmethod
    def column_names(cols: List[str]):
        def addl(text, l):
            return list(map(lambda x: text + x, l))
        return ['in_r2', 'os_r2'] + cols + addl('t_', cols) + addl('p_', cols)

    def contains_nan(self):
        return np.isnan(self.values).any()



class AveragedRegressionResults:
    def __init__(self, l: List, column_names: List[str] = None):
        self.column_names = column_names

        l = list(filter(lambda x: not x.contains_nan(), l))

        if len(l) == 0:
            self.values = np.ndarray(shape=(0, ))
            self.average = np.ndarray(shape=(0, ))
            self.std = np.ndarray(shape=(0, ))
            return

        # self.stats_names = l[0].stats_names
        if type(l[0]) == RegressionResults:
            self.values = np.stack([r.values for r in l], axis=-1)
        elif type(l[0]) == AveragedRegressionResults:
            self.values = np.concatenate([r.values for r in l], axis=1)
            self.column_names = l[0].column_names

        self.average = np.average(self.values, axis=1)
        self.std = np.std(self.values, axis=1)

    def summary(self):
        vals = self.average.tolist()
        vals = list(map(lambda x: str(x), vals))
        stds = self.std.tolist()
        stds = list(map(lambda x: str(x), stds))
        if self.column_names is None:
            return [f"{v} , std {s}" for (v, s) in zip(vals, stds)]
        else:
            return [f"{c}: {v} , std {s} " for (c, v, s) in zip(self.column_names, vals, stds)]

    def log(self, logger):
        lines = self.summary()
        for l in lines:
            logger.info(l)

    def contains_nan(self):
        return np.isnan(self.values).any()
    def save_pickle(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(file_path: str):
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        return results

    @staticmethod
    def from_directory(path: str):
        file_list = os.listdir(path)
        l = [AveragedRegressionResults.from_pickle(os.path.join(path, file_name)) for file_name in file_list if
             file_name.endswith('.pickle')]
        return AveragedRegressionResults(l)
