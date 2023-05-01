import os
from datetime import date

import numpy as np
import pandas as pd

import data_loader.dates as dates_loader
import data_loader.data_selector as data_selector
import data_loader.one_day as loader
import data_loader.data_processor as data_processor
import constants

from experiments.clustering import get_clusters, get_kneighbours
from models.regression_results import RegressionResults, AveragedRegressionResults
from models.linear_regression import run_regression_prediction, r2_score
from joblib import Parallel, delayed
from logging_utils import get_logger, log, log_tickers
from data_manipulation.bucket_ofi import compute_bucket_ofi_df_from_bucket_ofi, BucketOFIProps
from experiments.contemporaneous import compute_datasets_for_interval, compute_concatenated_dataset

from strategy.prediction import create_prediction_df


def load_day_dataframes(d, tickers, x_selector, args):
    dfs = {}
    for t in tickers:
        if args.load_all_horizonts:
            now_dfs = []
            for h in args.selector.multi_horizonts:
                df = loader.get_extracted_single_day_df_for_ticker(folder_path=os.path.join(args.folder_path, str(h)),
                                                                   ticker=t, d=d)
                now_dfs.append(df)
            df = x_selector.process(now_dfs)
        else:
            df = loader.get_extracted_single_day_df_for_ticker(folder_path=args.folder_path, ticker=t, d=d)
            if 'data_horizont' in args:
                props = BucketOFIProps(bucket_size=args.horizont, prev_bucket_size=args.data_horizont,
                                       rolling_size=args.horizont, rounding=True)
                df = compute_bucket_ofi_df_from_bucket_ofi(df, props)
            df = x_selector.process(df)

        if df is None or df.empty:
            continue

        dfs[t] = df
    return dfs


def experiment(args, tickers: list[str], logger=None, logger_name: str = None):
    if logger is None and logger_name is not None:
        logger = get_logger(None, logger_name)

    if logger is not None:
        logger.info(f"Tickers: {tickers}")

    in_sample_size, os_size, rolling = args.experiment.in_sample_size, args.experiment.os_size, args.experiment.os_size

    folder_path = args.folder_path
    if args.load_all_horizonts:
        folder_path = os.path.join(folder_path, '1')
    dates = dates_loader.get_dates_in_majority_from_folder(folder_path=folder_path, tickers=tickers,
                                                           start_date=args.start_date, end_date=args.end_date)

    x_selector = data_selector.factory(args)
    y_lag = args.horizont
    y_selector = data_selector.return_factory(y_lag=y_lag)

    start_time = constants.START_TRADE + constants.VOLATILE_TIMEFRAME + args.horizont * (
            args.selector.multi_horizonts[-1] - 1)
    end_time = constants.END_TRADE - constants.VOLATILE_TIMEFRAME - in_sample_size - os_size - y_lag + 1

    def run_one_date(d):
        logger = None
        if logger_name is not None:
            logger = get_logger(None, logger_name)
        log(f"Running {d} - {log_tickers(tickers)} ...")

        # Load dataframes for all available tickers on this day
        dfs = load_day_dataframes(d, tickers, x_selector, args)

        def one_experiment(interval_left):
            train_datasets, test_datasets = compute_datasets_for_interval(interval_left, dfs, x_selector, y_selector,
                                                                          args, add_train_if_no_test=False)
            if len(test_datasets.keys()) == 0:
                return None

            pred_interval = (interval_left + args.experiment.in_sample_size,
                             interval_left + args.experiment.in_sample_size + args.experiment.os_size)

            def get_regression_results(train_x, train_y, test_x, test_y, tickers):
                processor = data_processor.factory_group(args.processor)
                train_x = processor.fit(train_x)
                test_x = processor.process(test_x)

                try:
                    results, y_pred = run_regression_prediction(regression_type=args.regression.type,
                                                                train_dataset=(train_x, train_y),
                                                                test_dataset=(test_x, test_y))
                    return results, y_pred
                except Exception as e:
                    log(f'Error --- ticker {log_tickers(tickers)} --- day {d} --- time {interval_left} --- {e}',
                        logger=logger)
                return None, None

            def compute_y_res(keys, test_y, y_pred):
                return {t: (y_tr, y_pr) for t, y_tr, y_pr in zip(keys, test_y.tolist(), y_pred.tolist())}

            if args.experiment.name in ['individual_future', 'universal_future']:
                train_x, train_y = compute_concatenated_dataset(train_datasets.values())
                test_x, test_y = compute_concatenated_dataset(test_datasets.values())
                results, y_pred = get_regression_results(train_x, train_y, test_x, test_y, tickers)
                if results is None:
                    return None
                y_res = compute_y_res(list(test_datasets.keys()), test_y, y_pred)
                return pred_interval, results, y_res

            elif args.experiment.name in ['clustered_future']:
                keys = list(train_datasets.keys())
                clusters = get_clusters(train_datasets.values(), args.clustering)
                results = []
                y_res_list = []
                for cluster in clusters:
                    train_list, test_list = [], []
                    now_tickers = []
                    test_keys = []
                    for i in cluster:
                        key = keys[i]
                        now_tickers.append(key)
                        train_list.append(train_datasets[key])
                        if key in test_datasets:
                            test_list.append(test_datasets[key])
                            test_keys.append(key)

                    train_x, train_y = compute_concatenated_dataset(train_list)
                    if len(test_list) == 0:
                        continue
                    test_x, test_y = compute_concatenated_dataset(test_list)
                    res, y_pred = get_regression_results(train_x, train_y, test_x, test_y, now_tickers)
                    if res is not None:
                        y_res = compute_y_res(test_keys, test_y, y_pred)
                        y_res_list.append(y_res)
                        results.append(res)
                y_res = {k: v for dct in y_res_list for k, v in dct.items()}
                return pred_interval, AveragedRegressionResults(results), y_res
            elif args.experiment.name in ['neigh_future']:
                keys = list(train_datasets.keys())
                neighs_list = get_kneighbours(train_datasets.values(), args.neighbours)
                results = []
                y_res = {}
                for key, neighs in zip(keys, neighs_list):
                    if key not in test_datasets:
                        continue
                    train_list = []
                    now_tickers = []
                    for i in neighs:
                        curr_key = keys[i]
                        now_tickers.append(curr_key)
                        train_list.append(train_datasets[curr_key])
                    train_x, train_y = compute_concatenated_dataset(train_list)
                    test_x, test_y = test_datasets[key]
                    res, y_pred = get_regression_results(train_x, train_y, test_x, test_y, now_tickers)
                    if res is not None:
                        y_res[key] = (test_y[0], y_pred[0])
                        results.append(res)
                return pred_interval, AveragedRegressionResults(results), y_res
            else:
                raise ValueError(f'invalid experiment name {args.experiment.name}')

        results = Parallel(n_jobs=1)(delayed(one_experiment)(t) for t in range(start_time, end_time + 1, rolling))
        results = list(filter(lambda x: x is not None, results))

        if len(results) == 0:
            return None

        pred_intervals, res, y_res = zip(*results)

        # Can make PNL using (pred_interval, y_res)
        pred_df = create_prediction_df(d, pred_intervals, y_res)

        y_true = np.array([x for d in y_res for (x, _) in d.values()])
        y_pred = np.array([x for d in y_res for (_, x) in d.values()])
        os_r2 = r2_score(y_true, y_pred)
        res = list(res)
        for x in res:
            x.set_os(os_r2)
        res = AveragedRegressionResults(res)
        return res, pred_df

        # for interval_left in range(start_time, end_time + 1, rolling):
        #     results = one_experiment(interval_left)
        #     result_list.append(results)

    # result_list = []
    # for d in dates:
    #     results = run_one_date(d)
    #     result_list += results
    result_list = Parallel(n_jobs=args.parallel_jobs)(delayed(run_one_date)(d) for d in dates)
    result_list = list(filter(lambda x: x is not None, result_list))

    result_list, pred_dfs = zip(*result_list)
    pred_df = pd.concat(pred_dfs)

    columns = x_selector.column_names
    if 'pca' in args.processor:
        columns = [f"pca_{x}" for x in range(1, args.processor.pca + 1)]
    if 'multipca' in args.processor:
        columns = sum([[f"pca_{i + 1}_{j + 1}" for j in range(args.processor.multipca.components)] for i in
                       range(args.processor.multipca.groups)], [])
    columns = ['intercept'] + columns
    column_names = RegressionResults.column_names(columns)

    avg_res = AveragedRegressionResults(result_list, column_names=column_names)
    if logger is not None:
        avg_res.log(logger)
    return avg_res, pred_df
