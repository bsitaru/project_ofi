from datetime import date
from models.lin_reg_model import model_factory
from models.regression_results import AveragedRegressionResults, RegressionResults

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

from joblib import Parallel, delayed


def log(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def experiment_for_ticker(args, ticker):
    in_sample_size = args.experiment.in_sample_size
    os_size = args.experiment.os_size
    rolling = args.experiment.rolling

    dates = dates_loader.get_dates_from_folder(folder_path=args.folder_path, tickers=[ticker],
                                               start_date=args.start_date, end_date=args.end_date)

    start_time = constants.START_TRADE + constants.VOLATILE_TIMEFRAME
    end_time = constants.END_TRADE - constants.VOLATILE_TIMEFRAME - os_size - in_sample_size + 1

    x_selector = data_selector.factory(args.selector.type)(volume_normalize=args.selector.volume_normalize,
                                                           levels=args.selector.levels)
    y_selector = data_selector.factory('Return')()

    res_list = []
    for d in dates:
        log(f"Running {d} - {ticker}...")
        df = loader.get_extracted_single_day_df_for_ticker(folder_path=args.folder_path, ticker=ticker, d=d)
        df_x = x_selector.process(df)
        df_y = y_selector.process(df)
        if df_x is None or df_x.empty or df_y is None or df_y.empty:
            continue

        for t in range(start_time, end_time + 1, rolling):
            train_x = x_selector.select_interval_df(df_x, t, t + in_sample_size)
            train_y = y_selector.select_interval_df(df_y, t, t + in_sample_size)

            test_x = x_selector.select_interval_df(df_x, t + in_sample_size, t + in_sample_size + os_size)
            test_y = y_selector.select_interval_df(df_y, t + in_sample_size, t + in_sample_size + os_size)

            if train_x.size == 0 or test_x.size == 0 or train_y.size == 0 or test_y.size == 0:
                continue

            processor = data_processor.factory(args.processor)
            train_x = processor.fit(train_x)
            test_x = processor.process(test_x)

            try:
                results = run_linear_regression(regression_type=args.regression.type, train_dataset=(train_x, train_y),
                                                test_dataset=(test_x, test_y))
                res_list.append(results)
                if results.values[1] < 0:
                    log(f'Negative OS: {results.values[1]} --- ticker {ticker} --- day {d} --- time {t}')
            except AttributeError as e:
                log(f'Error --- ticker {ticker} --- day {d} --- time {t} --- {e}')

    avg_res = AveragedRegressionResults(res_list)
    return avg_res


def naming(args):
    r = [args.experiment.name,
         "issize", str(args.experiment.in_sample_size),
         "ossize", str(args.experiment.os_size),
         "rolling", str(args.experiment.rolling),
         "seltype", args.selector.type,
         "nrmvol", str(args.selector.volume_normalize),
         "lvl", str(args.selector.levels),
         "regtype", str(args.regression.type)]
    return "_".join(r)


def get_logger(folder_path: str) -> logging.Logger:
    logger = logging.getLogger('myapp')
    hdlr = logging.FileHandler(os.path.join(folder_path, 'logs.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def run_experiment_individual(args):
    print(f'Experiment: {args.experiment.name}')
    results_name = naming(args)
    results_path = os.path.join(args.results_path, results_name)
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    logger = get_logger(results_path)
    logger.info(f'Experiment: {args.experiment.name}')

    with open(os.path.join(results_path, 'config.yaml'), 'w') as outfile:
        yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    def run_experiment_for_ticker(ticker: str):
        results = experiment_for_ticker(ticker=ticker, args=args)

        if results.values is not None:
            results_text = f'{ticker} --- INS : {results.average[0]} --- OOS : {results.average[1]}'
            print(results_text, flush=True)
            print(results_text, flush=True, file=sys.stderr)
            logger.info(results_text)

            with open(os.path.join(results_path, f'{ticker}.pickle'), 'wb') as f:
                pickle.dump(results, f)

    Parallel(n_jobs=args.parallel_jobs)(delayed(run_experiment_for_ticker)(t) for t in args.tickers)
