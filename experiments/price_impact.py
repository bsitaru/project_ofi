from datetime import date
from models.lin_reg_model import model_factory
from models.regression_results import AveragedRegressionResults, RegressionResults
import data_loader.one_day as loader
import constants
import math
import sys
import os
import pickle
import data_loader.dates as dates_loader

from joblib import Parallel, delayed


def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def experiment_for_ticker(folder_path: str, temp_path: str, in_sample_size: int, os_size: int,
                          rolling: int, model_class, ticker: str, start_date: date = None,
                          end_date: date = None):
    dates = dates_loader.get_dates_from_archive_files(folder_path=folder_path, tickers=[ticker], start_date=start_date,
                                                      end_date=end_date)

    start_time = constants.START_TRADE + constants.VOLATILE_TIMEFRAME
    end_time = constants.END_TRADE - constants.VOLATILE_TIMEFRAME - os_size - in_sample_size + 1

    res_list = []
    for d in dates:
        log(f"Running {d} - {ticker}...")
        df = loader.get_single_date_df_for_ticker(folder_path=folder_path, temp_path=temp_path, d=d, ticker=ticker)
        df = model_class.process_bucket_ofi_df(df)
        if df is None or df.empty:
            continue
        for t in range(start_time, end_time + 1, rolling):
            train_df = df[(df['start_time'] >= t) & (df['start_time'] < t + in_sample_size)]
            test_df = df[
                (df['start_time'] >= t + in_sample_size) & (df['start_time'] < t + in_sample_size + os_size)]
            if train_df.size == 0 or test_df.size == 0 or train_df['event_count'].sum() == 0 or \
                    test_df['event_count'].sum() == 0:
                continue

            try:
                model = model_class()
                results = model.run(train_df, test_df)
                res_list.append(results)
            except:
                print(f'Error --- ticker {ticker} --- day {d} --- time {t}', file=sys.stderr, flush=True)

    avg_res = AveragedRegressionResults(res_list)
    return avg_res


def run_experiment_individual(folder_path: str, temp_path: str, results_path: str, in_sample_size: int,
                              model_name: str, os_size: int = None, rolling: int = None, tickers: list[str] = None,
                              start_date: date = None, end_date: date = None, parallel_jobs: int = 1):
    os_size = in_sample_size if os_size is None else os_size
    rolling = in_sample_size if rolling is None else rolling

    model_class = model_factory(model_name)

    def run_experiment_for_ticker(ticker: str):
        results = experiment_for_ticker(folder_path=folder_path, temp_path=temp_path, in_sample_size=in_sample_size,
                                        os_size=os_size, rolling=rolling, model_class=model_class, ticker=ticker,
                                        start_date=start_date, end_date=end_date)

        if results.values is not None:
            results_text = f'{ticker} --- INS : {results.average[0]} --- OOS : {results.average[1]}'
            print(results_text, flush=True)
            print(results_text, flush=True, file=sys.stderr)

            with open(os.path.join(results_path, ticker + '.pickle'), 'wb') as f:
                pickle.dump(results, f)

    Parallel(n_jobs=parallel_jobs)(delayed(run_experiment_for_ticker)(t) for t in tickers)
