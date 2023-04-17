import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from models.regression_results import RegressionResults

from statsmodels.tools.tools import pinv_extended
import statsmodels.api as sm
import sklearn
import statsmodels


def regression_analysis(X, y, model, lib):
    is_statsmodels = False
    is_sklearn = False

    # check for accepted linear models
    if lib == 'sklearn':
        is_sklearn = True
    elif lib == 'statsmodels':
        is_statsmodels = True
    else:
        print("Only linear models are supported!")
        return None

    has_intercept = False

    if is_statsmodels and all(np.array(X)[:, 0] == 1):
        # statsmodels add_constant has been used already
        has_intercept = True
    elif is_sklearn and model.intercept_:
        has_intercept = True

    if is_statsmodels:
        # add_constant has been used already
        x = X
        model_params = model.params
    else:  # sklearn model
        if has_intercept:
            x = sm.add_constant(X)
            model_params = np.hstack([np.array([model.intercept_]), model.coef_])
        else:
            x = X
            model_params = model.coef_

    # y = np.array(y).ravel()

    # define the OLS model
    olsModel = sm.OLS(y, x)

    pinv_wexog, _ = pinv_extended(x)
    normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))

    return sm.regression.linear_model.RegressionResults(olsModel, model_params, normalized_cov_params)


def run_linear_regression(regression_type: str, train_dataset: (np.ndarray, np.ndarray),
                          test_dataset: (np.ndarray, np.ndarray)) -> RegressionResults:
    x_train, y_train = train_dataset
    x_train = sm.add_constant(x_train)
    if regression_type == "linear":
        model = sm.OLS(y_train, x_train).fit()
    elif regression_type == 'lasso':
        model = sm.OLS(y_train, x_train).fit_regularized(method='elastic_net', L1_wt=1.)
    else:
        raise ValueError(f'invalid regression type {regression_type}')
    results = regression_analysis(x_train, y_train, model, lib='statsmodels')

    x_test, y_test = test_dataset
    x_test = sm.add_constant(x_test)
    y_pred = results.predict(x_test)
    os_r2 = r2_score(y_test, y_pred)
    return RegressionResults.from_lin_reg_results(results, os_r2)
