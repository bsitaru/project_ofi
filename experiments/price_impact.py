from datetime import date
from models.lin_reg_model import SplitOFIModel, OFIModel
import data_loader.one_day as loader
import constants
import math
import sys
import os

from statistics import mean, stdev
from joblib import Parallel, delayed

def transpose(l):
    return list(map(list, zip(*l)))

def run_experiment(folder_path: str, temp_path: str, results_path: str, bucket_size: int, in_sample_size: int,
                   os_size: int = None,
                   rolling: int = None, tickers: list[str] = None, start_date: date = None, end_date: date = None,
                   parallel_jobs: int = 1):
    os_size = in_sample_size if os_size is None else os_size
    rolling = in_sample_size if rolling is None else rolling

    def create_model():
        return SplitOFIModel(levels=10, return_type='current')

    ins_r2 = []
    oos_r2 = []

    def run_experiment_for_ticker(ticker: str):
        dates = loader.get_dates_from_archive_files(folder_path=folder_path, tickers=[ticker])

        def filter_date(d: date):
            if start_date is not None and d < start_date:
                return False
            if end_date is not None and d > end_date:
                return False
            return True

        dates = list(filter(filter_date, dates))

        loc_ins_r2 = []
        loc_oos_r2 = []
        loc_params = []
        loc_tvalues = []

        for d in dates:
            print(f"Running {d}...", file=sys.stderr)
            df = loader.get_day_df(folder_path=folder_path, temp_path=temp_path, d=d, bucket_size=bucket_size,
                                   tickers=[ticker])
            start_time = constants.START_TRADE + constants.VOLATILE_TIMEFRAME
            end_time = constants.END_TRADE - constants.VOLATILE_TIMEFRAME - os_size - in_sample_size + 1
            for t in range(start_time, end_time + 1, rolling):
                train_df = df[(df['start_time'] >= t) & (df['start_time'] < t + in_sample_size)]
                test_df = df[
                    (df['start_time'] >= t + in_sample_size) & (df['start_time'] < t + in_sample_size + os_size)]
                if train_df.size == 0 or test_df.size == 0:
                    continue
                model = create_model()
                model.fit(train_df)
                model.score_test(test_df)
                # models.append(model)
                # Skip if r2 is nan
                if math.isnan(model.get_adj_r2()):
                    continue
                loc_ins_r2.append(model.get_adj_r2())
                loc_oos_r2.append(model.get_oos_r2())
                loc_params.append(model.results.params)
                loc_tvalues.append(model.results.tvalues)

        loc_params = transpose(loc_params)
        loc_tvalues = transpose(loc_tvalues)

        avg_r2_ins = mean(loc_ins_r2)
        avg_r2_oos = mean(loc_oos_r2)
        print(f'{ticker} --- INS : {avg_r2_ins} ; {stdev(loc_ins_r2)} --- OOS: {avg_r2_oos} ; {stdev(loc_oos_r2)}', flush=True)
        # print(f'R2 Out of Sample : {avg_r2_oos}')

        results_file = os.path.join(results_path, ticker)
        f = open(results_file, 'w')
        f.write('\n'.join([str(loc_ins_r2), str(loc_oos_r2), str(loc_params), str(loc_tvalues)]))
        f.write('\n\n')
        for i in range(31):
            f.write(f'{mean(loc_params[i])} ; {stdev(loc_params[i])} ; {mean(loc_tvalues[i])} ; {stdev(loc_tvalues[i])}\n')
        f.close()

        nonlocal ins_r2
        nonlocal oos_r2
        ins_r2 += loc_ins_r2
        oos_r2 += loc_oos_r2

    # for ticker in tickers:
    #     run_experiment_for_ticker(ticker)
    Parallel(n_jobs=parallel_jobs)(delayed(run_experiment_for_ticker)(t) for t in tickers)

    # avg_r2_ins = mean(ins_r2)
    # avg_r2_oos = mean(oos_r2)

    # print(f'R2 In Sample : {avg_r2_ins}')
    # print(f'R2 Out of Sample : {avg_r2_oos}')
