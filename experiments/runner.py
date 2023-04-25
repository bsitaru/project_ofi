import numpy as np
import sys
import data_loader.dates as dates_loader
import data_loader.data_selector as data_selector
import data_loader.one_day as loader
import data_loader.data_processor as data_processor
import constants

from experiments.clustering import get_clusters, get_kneighbours
from models.regression_results import RegressionResults, AveragedRegressionResults
from models.linear_regression import run_linear_regression
from joblib import Parallel, delayed


def log(text: str, logger=None):
    print(text, file=sys.stderr, flush=True)
    if logger is not None:
        logger.info(text)


def log_tickers(tickers):
    if len(tickers) == 1:
        return tickers[0]
    elif len(tickers) <= 5:
        return ', '.join(tickers)
    else:
        return ', '.join(tickers[:5]) + ' ...'


def flatten_list(l):
    if l is None:
        return []
    if type(l) != list:
        return l
    ans = []
    for x in l:
        if type(x) == list:
            ans += x
        elif x is None:
            continue
        else:
            ans.append(x)
    return ans


def load_day_dataframes(d, tickers, x_selector, y_selector, args):
    dfs = {}
    for t in tickers:
        df = loader.get_extracted_single_day_df_for_ticker(folder_path=args.folder_path, ticker=t, d=d)
        df_x = x_selector.process(df)
        df_y = y_selector.process(df)
        if df_x is None or df_x.empty or df_y is None or df_y.empty:
            continue
        dfs[t] = (df_x, df_y)
    return dfs


def compute_datasets_for_interval(interval_left, dfs, x_selector, y_selector, args):
    train_datasets = {}
    test_datasets = {}
    in_sample_size, os_size, rolling = args.experiment.in_sample_size, args.experiment.os_size, args.experiment.os_size
    for (t, (df_x, df_y)) in dfs.items():
        train_x = x_selector.select_interval_df(df_x, interval_left, interval_left + in_sample_size)
        train_y = y_selector.select_interval_df(df_y, interval_left, interval_left + in_sample_size)

        test_x = x_selector.select_interval_df(df_x, interval_left + in_sample_size,
                                               interval_left + in_sample_size + os_size)
        test_y = y_selector.select_interval_df(df_y, interval_left + in_sample_size,
                                               interval_left + in_sample_size + os_size)

        if train_x.size == 0 or train_y.size == 0:
            continue

        if test_x.size == 0 or test_y.size == 0:
            test_x, test_y = None, None

        processor = data_processor.factory_individual(args.processor)
        train_x = processor.fit(train_x)
        if test_x is not None:
            test_x = processor.process(test_x)

        train_datasets[t] = train_x, train_y
        if test_x is not None:
            test_datasets[t] = test_x, test_y
    return train_datasets, test_datasets


def compute_concatenated_dataset(datasets):
    x = np.concatenate([x for (x, _) in datasets])
    y = np.concatenate([y for (_, y) in datasets])
    return x, y


def experiment(args, tickers: list[str], logger=None):
    if logger is not None:
        logger.info(f"Tickers: {tickers}")

    in_sample_size, os_size, rolling = args.experiment.in_sample_size, args.experiment.os_size, args.experiment.os_size

    dates = dates_loader.get_dates_in_majority_from_folder(folder_path=args.folder_path, tickers=tickers,
                                                           start_date=args.start_date, end_date=args.end_date)

    start_time = constants.START_TRADE + constants.VOLATILE_TIMEFRAME
    end_time = constants.END_TRADE - constants.VOLATILE_TIMEFRAME - os_size - in_sample_size + 1

    x_selector = data_selector.factory(args.selector)
    y_selector = data_selector.return_factory()

    result_list = []
    for d in dates:
        log(f"Running {d} - {log_tickers(tickers)} ...")

        # Load dataframes for all available tickers on this day
        dfs = load_day_dataframes(d, tickers, x_selector, y_selector, args)

        def one_experiment(interval_left):
            train_datasets, test_datasets = compute_datasets_for_interval(interval_left, dfs, x_selector, y_selector,
                                                                          args)
            if len(test_datasets.keys()) == 0:
                return None

            def get_regression_results(train_x, train_y, test_x, test_y, tickers):
                processor = data_processor.factory_group(args.processor)
                train_x = processor.fit(train_x)
                test_x = processor.process(test_x)

                try:
                    results = run_linear_regression(regression_type=args.regression.type,
                                                    train_dataset=(train_x, train_y),
                                                    test_dataset=(test_x, test_y))

                    if 'pca' in args.processor:
                        results.values = np.concatenate([results.values, processor.explained_variance_ratio()])

                    if results.values[1] < 0:
                        log(f'Negative OS: {results.values[1]} --- ticker {log_tickers(tickers)} --- day {d} --- time {interval_left}',
                            logger=logger)
                    return results
                except Exception as e:
                    log(f'Error --- ticker {log_tickers(tickers)} --- day {d} --- time {interval_left} --- {e}',
                        logger=logger)
                return None

            if args.experiment.name in ['individual_price_impact', 'universal_price_impact']:
                train_x, train_y = compute_concatenated_dataset(train_datasets.values())
                test_x, test_y = compute_concatenated_dataset(test_datasets.values())
                return get_regression_results(train_x, train_y, test_x, test_y, tickers)
            elif args.experiment.name in ['clustered_price_impact']:
                keys = list(train_datasets.keys())
                clusters = get_clusters(train_datasets.values(), args.clustering)
                results = []
                for cluster in clusters:
                    train_list, test_list = [], []
                    now_tickers = []
                    for i in cluster:
                        key = keys[i]
                        now_tickers.append(key)
                        train_list.append(train_datasets[key])
                        if key in test_datasets:
                            test_list.append(test_datasets[key])

                    train_x, train_y = compute_concatenated_dataset(train_list)
                    if len(test_list) == 0:
                        continue
                    test_x, test_y = compute_concatenated_dataset(test_list)
                    res = get_regression_results(train_x, train_y, test_x, test_y, now_tickers)
                    results.append(res)
                return results
            elif args.experiment.name in ['neigh_price_impact']:
                keys = list(train_datasets.keys())
                neighs_list = get_kneighbours(train_datasets.values(), args.neighbours)
                results = []
                for key, neighs in zip(keys, neighs_list):
                    if key == 'ISRG':
                        pass
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
                    res = get_regression_results(train_x, train_y, test_x, test_y, now_tickers)
                    results.append(res)
                return results
            else:
                raise ValueError(f'invalid experiment name {args.experiment.name}')

        results = Parallel(n_jobs=args.parallel_jobs)(delayed(one_experiment)(t) for t in range(start_time, end_time + 1, rolling))
        results = flatten_list(results)
        result_list += results

        # for interval_left in range(start_time, end_time + 1, rolling):
        #     results = one_experiment(interval_left)
        #     result_list.append(results)

    columns = x_selector.column_names
    if 'pca' in args.processor:
        columns = [f"pca_{x}" for x in range(1, args.processor.pca + 1)]
    if 'multipca' in args.processor:
        columns = sum([[f"pca_{i + 1}_{j + 1}" for j in range(args.processor.multipca.components)] for i in
                       range(args.processor.multipca.groups)], [])
    columns = ['intercept'] + columns
    column_names = RegressionResults.column_names(columns)
    if 'pca' in args.processor:
        column_names += [f"pca_explained_{x}" for x in range(1, args.processor.pca + 1)]

    avg_res = AveragedRegressionResults(result_list, column_names=column_names)
    if logger is not None:
        avg_res.log(logger)
    return avg_res
