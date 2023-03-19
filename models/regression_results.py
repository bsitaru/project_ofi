import statsmodels
import numpy as np


class RegressionResults:
    def __init__(self, in_r2: float, os_r2: float, param_values: list[float], tvalues: list[float], param_names=None):
        self.param_names = [f'x{i}' for i in range(len(param_values))] if param_names is None else param_names

        self.stats_names = ['in_r2', 'os_r2'] + self.param_names + [f't_{s}' for s in self.param_names]

        vals = [in_r2, os_r2] + list(param_values) + list(tvalues)
        self.values = np.array(vals)

    @staticmethod
    def from_lin_reg_results(results: statsmodels.regression.linear_model.RegressionResults, os_r2: float,
                             param_names=None):
        return RegressionResults(in_r2=results.rsquared_adj, os_r2=os_r2, param_values=results.params,
                                 tvalues=results.tvalues, param_names=param_names)


class AveragedRegressionResults:
    def __init__(self, l: list[RegressionResults]):
        if len(l) == 0:
            return

        self.stats_names = l[0].stats_names
        self.values = np.stack([r.values for r in l], axis=-1)

        self.average = np.average(self.values, axis=1)
        self.std = np.std(self.values, axis=1)
