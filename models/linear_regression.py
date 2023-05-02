import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from models.regression_results import RegressionResults

from statsmodels.tools.tools import pinv_extended
import statsmodels.api as sm
import sklearn
import statsmodels
from sklearn.linear_model import LassoCV, LassoLarsCV, RidgeCV


def r2_oos_score(y_true, y_pred, y_train_mean):
    # R2 in sample =     1 - Sum[ (y_pred - y_true)**2 ] / Sum[ (y_true.mean()  - y_true)**2  ]
    # # one particular case of R2 out of sample
    # R2 out of sample = 1 - Sum[ (y_pred - y_true)**2 ] / Sum[ (y_train.mean() - y_true)**2  ]

    up = np.sum(np.power(y_pred - y_true, 2))
    down = np.sum(np.power(y_true - y_train_mean, 2))
    return 1.0 - up / down


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
            x = sm.add_constant(X, has_constant='add')
            model_params = np.hstack([np.array([model.intercept_]), model.coef_])
        else:
            x = sm.add_constant(X, has_constant='add')
            model_params = np.hstack([np.array([0]), model.coef_])

    # y = np.array(y).ravel()

    # define the OLS model
    olsModel = sm.OLS(y, x)

    pinv_wexog, _ = pinv_extended(x)
    normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))

    return sm.regression.linear_model.RegressionResults(olsModel, model_params, normalized_cov_params)


def run_regression(regression_type: str, train_dataset: (np.ndarray, np.ndarray),
                   test_dataset: (np.ndarray, np.ndarray)) -> (RegressionResults, np.ndarray):
    x_train, y_train = train_dataset
    if regression_type == "linear":
        x_train = sm.add_constant(x_train, has_constant='add')
        model = sm.OLS(y_train, x_train).fit()
        # results = regression_analysis(x_train, y_train, model, lib='statsmodels')
        results = model
    elif regression_type == 'lasso':
        # model = sm.OLS(y_train, x_train).fit_regularized(method='elastic_net', L1_wt=1.)
        model = LassoLarsCV(fit_intercept=True, max_iter=2000, cv=3)
        model.fit(x_train, y_train)
        results = regression_analysis(x_train, y_train, model, lib='sklearn')
        # print(f"is --- {results.rsquared_adj} --- alpha --- {model.alpha_}", )
    else:
        raise ValueError(f'invalid regression type {regression_type}')

    x_test, y_test = test_dataset
    if np.shape(results.params)[0] > np.shape(x_test)[1]:
        x_test = sm.add_constant(x_test, has_constant='add')
    y_pred = results.predict(x_test)
    return results, y_pred


def run_regression_prediction(regression_type: str, train_dataset: (np.ndarray, np.ndarray),
                              test_dataset: (np.ndarray, np.ndarray)) -> (RegressionResults, np.ndarray):
    results, y_pred = run_regression(regression_type, train_dataset, test_dataset)
    y_train_std = np.std(np.exp(results.fittedvalues))
    return RegressionResults.from_lin_reg_results(results, 0), y_pred, y_train_std


def run_regression_results(regression_type: str, train_dataset: (np.ndarray, np.ndarray),
                           test_dataset: (np.ndarray, np.ndarray)) -> RegressionResults:
    results, y_pred = run_regression(regression_type, train_dataset, test_dataset)
    _, y_test = test_dataset
    os_r2 = r2_score(y_test, y_pred)
    return RegressionResults.from_lin_reg_results(results, os_r2)
