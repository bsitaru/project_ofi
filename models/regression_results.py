import numpy as np

class RegressionResults:
    def __init__(self, in_r2: float, os_r2: float, param_values: list[float], tvalues: list[float]):
        # self.param_names = [f'x{i}' for i in range(len(param_values))] if param_names is None else param_names
        #
        # self.stats_names = ['in_r2', 'os_r2'] + self.param_names + [f't_{s}' for s in self.param_names]

        vals = [in_r2, os_r2] + list(param_values) + list(tvalues)
        self.values = np.array(vals)

    @staticmethod
    def from_lin_reg_results(results, os_r2: float):
        return RegressionResults(in_r2=results.rsquared_adj, os_r2=os_r2, param_values=results.params,
                                 tvalues=results.tvalues)


class AveragedRegressionResults:
    def __init__(self, l: list[RegressionResults]):
        l = list(filter(lambda x: not self._has_nan(x), l))

        if len(l) == 0:
            return

        # self.stats_names = l[0].stats_names
        if type(l[0]) == RegressionResults:
            self.values = np.stack([r.values for r in l], axis=-1)
        elif type(l[0]) == AveragedRegressionResults:
            self.values = np.concatenate([r.values for r in l], axis=1)

        self.average = np.average(self.values, axis=1)
        self.std = np.std(self.values, axis=1)

    def _has_nan(self, r: RegressionResults):
        return np.isnan(r.values).any()
