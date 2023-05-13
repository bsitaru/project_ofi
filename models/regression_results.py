import os
from typing import Union, List

import numpy as np
import pickle
import statsmodels.api as sm

from logging_utils import log


class RegressionResults:
    def __init__(self, values):
        # vals = [in_r2, os_r2] + list(param_values) + list(tvalues)
        self.values = np.array(values)

    def set_os(self, os_r2):
        self.values[1] = os_r2

    @staticmethod
    def from_lin_reg_results(results: sm.regression.linear_model.OLSResults, os_r2: float):
        if results.df_resid == 0:
            in_r2 = results.rsquared
        else:
            in_r2 = results.rsquared_adj
        values = [in_r2, os_r2] + list(results.params) + list(results.tvalues)
        return RegressionResults(values)

    @staticmethod
    def from_lasso_reg_results(results: sm.regression.linear_model.OLSResults, os_r2: float):
        if results.df_resid == 0:
            in_r2 = results.rsquared
        else:
            in_r2 = results.rsquared_adj
        params = np.array(results.params)
        eps = 1e-9
        values = [in_r2, os_r2] + list(params.tolist()) + list(np.where(np.abs(params) <= eps, 0.0, 1.0).tolist())
        return RegressionResults(values)

    @staticmethod
    def from_regression_results(results, os_r2, regression_type):
        if regression_type == 'linear':
            return RegressionResults.from_lin_reg_results(results, os_r2)
        elif regression_type == 'lasso':
            return RegressionResults.from_lasso_reg_results(results, os_r2)
        else:
            raise ValueError(f"Invalid regression type {regression_type}")

    @staticmethod
    def column_names(cols: List[str], regression_type='linear'):
        def addl(text, l):
            return list(map(lambda x: text + x, l))

        if regression_type == 'linear':
            return ['in_r2', 'os_r2'] + cols + addl('t_', cols)
        elif regression_type == 'lasso':
            return ['in_r2', 'os_r2'] + cols + addl('used_', cols)
        else:
            raise ValueError(f"Invalid regression type {regression_type}")

    def contains_nan(self):
        return np.isnan(self.values).any()



class AveragedRegressionResults:
    # self.values is stored transposed compared to what you expect. (each line represents the values for a coefficient)
    def __init__(self, l: List, column_names: List[str] = None):
        self.column_names = column_names

        l = list(filter(lambda x: x is not None, l))
        l = list(filter(lambda x: not x.contains_nan(), l))

        if len(l) == 0:
            self.values = np.ndarray(shape=(0, ))
            self.average = np.ndarray(shape=(0, ))
            self.std = np.ndarray(shape=(0, ))
            return

        if type(l[0]) == RegressionResults:
            self.values = np.stack([r.values for r in l], axis=-1)
        elif type(l[0]) == AveragedRegressionResults:
            self.values = np.concatenate([r.values for r in l], axis=1)
            if l[0].column_names is not None:
                self.column_names = l[0].column_names

        self.average = np.average(self.values, axis=1)
        self.std = np.std(self.values, axis=1)

    def set_os(self, os_r2):
        self.average[1] = os_r2
        self.values[1] = np.ones_like(self.values[0]) * os_r2

    def summary(self):
        vals = self.average.tolist()
        vals = list(map(lambda x: str(x), vals))
        stds = self.std.tolist()
        stds = list(map(lambda x: str(x), stds))
        assert len(vals) == len(stds)
        if self.column_names is None:
            return [f"{v} , std {s}" for (v, s) in zip(vals, stds)]
        else:
            assert len(vals) == len(self.column_names)
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

def print_contemp_stats(args, results, logger):
    if args.selector.type in ['OFI', 'AddOFI']:
        var_types = 1
    elif args.selector.type == 'OTOFI':
        var_types = 2
    elif args.selector.type == 'SplitOFI':
        var_types = 3
    else:
        var_types = 0

    levels = args.selector.levels
    tot_values = var_types * levels
    if 'pca' in args.selector or 'multipca' in args.selector:
        return

    arr = results.average[tot_values + 4:].tolist()
    cols = results.column_names[3:tot_values + 3]

    if len(arr) != len(cols):
        return

    def group(get_group, aggr=sum, grouping_type=''):
        dct = {}
        for (c, val) in zip(cols, arr):
            g = get_group(c)
            if g not in dct:
                dct[g] = []
            dct[g].append(val)

        dct = {k: aggr(v) for k, v in dct.items()}
        logger.info(f"Grouping by {grouping_type}:")
        for k, v in dct.items():
            logger.info(f"{k}: {v}")

    if args.selector.type == 'SplitOFI':
        def get_ofi_type_group(name):
            return name.split('_')[1]
        group(get_ofi_type_group, grouping_type='ofi type')
    def get_level_group(name):
        return name.split('_')[-1]
    group(get_level_group, grouping_type='level')

    def print_nonempty_regressions(results):
        vals = np.transpose(results.values[tot_values+4:, :])
        sum_rows = np.sum(vals, axis=1)
        count = np.count_nonzero(sum_rows)
        num_tot = np.size(sum_rows)
        ans = float(count) / float(num_tot)
        logger.info(f"Non empty regressions: {ans} --- {count} / {num_tot}")
    print_nonempty_regressions(results)

def print_future_stats(args, results, logger):
    if args.selector.type == 'OFI':
        var_types = 1
    elif args.selector.type == 'OTOFI':
        var_types = 2
    elif args.selector.type == 'SplitOFI':
        var_types = 3
    else:
        var_types = 0

    horizonts = len(args.selector.multi_horizonts)
    levels = args.selector.levels

    vars_per_horizont = var_types * levels
    if 'multipca' in args.processor:
        vars_per_horizont = args.processor.multipca.components

    tot_values = vars_per_horizont * horizonts

    # in_r2, os_r2, intercept  --- before
    arr = results.average[tot_values + 4:].tolist()
    cols = results.column_names[3:tot_values + 3]

    assert len(arr) == len(cols)

    def group(get_group, aggr=sum, grouping_type=''):
        dct = {}
        for (c, val) in zip(cols, arr):
            g = get_group(c)
            if g not in dct:
                dct[g] = []
            dct[g].append(val)

        dct = {k: aggr(v) for k, v in dct.items()}
        logger.info(f"Grouping by {grouping_type}:")
        for k, v in dct.items():
            logger.info(f"{k}: {v}")

    def get_horizont_group(name):
        return name.split('_')[-1]
    group(get_horizont_group, grouping_type='horizont')

    if args.selector.type == 'SplitOFI' and 'multipca' not in args.selector:
        def get_ofi_type_group(name):
            return name.split('_')[1]
        group(get_ofi_type_group, grouping_type='ofi type')

        def get_ofi_and_horizont(name):
            name = name.split('_')
            return f"{name[1]}_{name[-1]}"
        group(get_ofi_and_horizont, grouping_type='ofi type and horizont')

    if 'multipca' not in args.selector:
        def get_level_group(name):
            return name.split('_')[-2]
        group(get_level_group, grouping_type='level')

        def get_level_and_horizont(name):
            name = name.split('_')
            return f"{name[-2]}_{name[-1]}"
        group(get_level_and_horizont, grouping_type='level and horizont')

    def print_nonempty_regressions(results):
        vals = np.transpose(results.values[tot_values+4:, :])
        sum_rows = np.sum(vals, axis=1)
        count = np.count_nonzero(sum_rows)
        num_tot = np.size(sum_rows)
        ans = float(count) / float(num_tot)
        logger.info(f"Non empty regressions: {ans} --- {count} / {num_tot}")
    print_nonempty_regressions(results)