import numpy as np
import data_loader.one_day as loader
import data_loader.dates as dates_loader
import data_loader.data_selector as data_selector
import data_loader.data_processor as data_processor
import constants
import sys
import os
import pickle
import logging
import yaml
from models.linear_regression import run_linear_regression
from models.regression_results import AveragedRegressionResults, RegressionResults

from joblib import Parallel, delayed


def get_logger(folder_path: str, logger_name: str = 'myapp') -> logging.Logger:
    logger = logging.getLogger(logger_name)
    hdlr = logging.FileHandler(os.path.join(folder_path, f"{logger_name}.log"))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


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


def experiment_for_tickers(args, tickers: list[str], logger=None):
    if logger is not None:
        logger.info(f"Tickers: {tickers}")

    in_sample_size = args.experiment.in_sample_size
    os_size = args.experiment.os_size
    rolling = args.experiment.rolling

    dates = dates_loader.get_dates_in_majority_from_folder(folder_path=args.folder_path, tickers=tickers,
                                                           start_date=args.start_date, end_date=args.end_date)

    start_time = constants.START_TRADE + constants.VOLATILE_TIMEFRAME
    end_time = constants.END_TRADE - constants.VOLATILE_TIMEFRAME - os_size - in_sample_size + 1

    x_selector = data_selector.factory(args.selector.type)(volume_normalize=args.selector.volume_normalize,
                                                           levels=args.selector.levels)
    y_selector = data_selector.factory('Return')()

    res_list = []
    for d in dates:
        log(f"Running {d} - {log_tickers(tickers)} ...")
        dfs = []
        for t in tickers:
            df = loader.get_extracted_single_day_df_for_ticker(folder_path=args.folder_path, ticker=t, d=d)
            df_x = x_selector.process(df)
            df_y = y_selector.process(df)
            if df_x is None or df_x.empty or df_y is None or df_y.empty:
                continue
            dfs.append((df_x, df_y))

        for interval_left in range(start_time, end_time + 1, rolling):
            train_xs, train_ys = [], []
            test_xs, test_ys = [], []

            for (df_x, df_y) in dfs:
                train_x = x_selector.select_interval_df(df_x, interval_left, interval_left + in_sample_size)
                train_y = y_selector.select_interval_df(df_y, interval_left, interval_left + in_sample_size)

                test_x = x_selector.select_interval_df(df_x, interval_left + in_sample_size,
                                                       interval_left + in_sample_size + os_size)
                test_y = y_selector.select_interval_df(df_y, interval_left + in_sample_size,
                                                       interval_left + in_sample_size + os_size)

                if train_x.size == 0 or test_x.size == 0 or train_y.size == 0 or test_y.size == 0:
                    continue

                processor = data_processor.factory_individual(args.processor)
                train_x = processor.fit(train_x)
                test_x = processor.process(test_x)

                train_xs.append(train_x)
                train_ys.append(train_y)
                test_xs.append(test_x)
                test_ys.append(test_y)

            if len(train_xs) == 0:
                continue

            train_x = np.concatenate(train_xs)
            train_y = np.concatenate(train_ys)
            test_x = np.concatenate(test_xs)
            test_y = np.concatenate(test_ys)

            processor = data_processor.factory_group(args.processor)
            train_x = processor.fit(train_x)
            test_x = processor.process(test_x)

            try:
                results = run_linear_regression(regression_type=args.regression.type, train_dataset=(train_x, train_y),
                                                test_dataset=(test_x, test_y))

                if 'pca' in args.processor:
                    results.values = np.concatenate([results.values, processor.explained_variance_ratio()])

                res_list.append(results)
                if results.values[1] < 0:
                    log(f'Negative OS: {results.values[1]} --- ticker {log_tickers(tickers)} --- day {d} --- time {interval_left}',
                        logger=logger)
            except Exception as e:
                log(f'Error --- ticker {log_tickers(tickers)} --- day {d} --- time {interval_left} --- {e}',
                    logger=logger)
    columns = x_selector.column_names
    if 'pca' in args.processor:
        columns = [f"pca_{x}" for x in range(1, args.processor.pca + 1)]
    columns = ['intercept'] + columns
    column_names = RegressionResults.column_names(columns)
    if 'pca' in args.processor:
        column_names += [f"pca_explained_{x}" for x in range(1, args.processor.pca + 1)]

    avg_res = AveragedRegressionResults(res_list, column_names=column_names)
    if logger is not None:
        avg_res.log(logger)
    return avg_res


def naming(args):
    r = [args.experiment.name]
    r += ["issize", str(args.experiment.in_sample_size)]
    r += ["ossize", str(args.experiment.os_size)]
    r += ["rolling", str(args.experiment.rolling)]
    r += ["seltype", args.selector.type]
    r += ["lvl", str(args.selector.levels)]
    if args.selector.volume_normalize:
        r += ["volnrm"]
    if args.processor.normalize:
        r += ["indvnrm"]
    if 'pca' in args.processor:
        r += ["pca", str(args.processor.pca)]
    r += ["regtype", str(args.regression.type)]
    return "_".join(r)


def experiment_init(args):
    print(f'Experiment: {args.experiment.name}')
    results_name = naming(args)
    results_path = os.path.join(args.results_path, results_name)
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    logger = get_logger(results_path, 'logs')
    logger.info(f'Experiment: {args.experiment.name}')

    with open(os.path.join(results_path, 'config.yaml'), 'w') as outfile:
        yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    return logger, results_path


def run_experiment_individual(args):
    logger, results_path = experiment_init(args)

    def run_experiment_for_ticker(ticker: str):
        logger_now = get_logger(results_path, ticker)

        results = experiment_for_tickers(tickers=[ticker], args=args, logger=logger_now)

        if results.average.size > 0:
            results_text = f'{ticker} --- INS : {results.average[0]} --- OOS : {results.average[1]}'
            log(results_text, logger=logger)

            results.save_pickle(os.path.join(results_path, f'{ticker}.pickle'))

    Parallel(n_jobs=args.parallel_jobs)(delayed(run_experiment_for_ticker)(t) for t in args.tickers)

    results = AveragedRegressionResults.from_directory(results_path)
    results.log(logger)


def run_experiment_universal(args):
    logger, results_path = experiment_init(args)

    with open(os.path.join(results_path, 'config.yaml'), 'w') as outfile:
        yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    results = experiment_for_tickers(args, tickers=args.tickers, logger=logger)
    results_text = f'{log_tickers(args.tickers)} --- INS : {results.average[0]} --- OOS : {results.average[1]}'
    log(results_text, logger=logger)
    results.save_pickle(os.path.join(results_path, 'universal.pickle'))
