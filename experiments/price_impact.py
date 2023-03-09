import numpy as np
import pandas as pd
import os

from datetime import date
from models.lin_reg_model import SplitOFIModel, OFIModel
import data_loader.one_day as loader
import constants

from statistics import mean


def run_experiment(folder_path: str, temp_path: str, bucket_size: int, in_sample_size: int, os_size: int = None,
                   rolling: int = None, tickers: list[str] = None, start_date: date = None, end_date: date = None):
    os_size = in_sample_size if os_size is None else os_size
    rolling = in_sample_size if rolling is None else rolling

    def create_model():
        return SplitOFIModel(levels=10, return_type='current')

    dates = loader.get_dates_from_archive_files(folder_path=folder_path, tickers=tickers)

    def filter_date(d: date):
        if start_date is not None and d < start_date:
            return False
        if end_date is not None and d > end_date:
            return False
        return True

    dates = list(filter(filter_date, dates))

    ins_r2 = []
    oos_r2 = []
    for d in dates:
        print(f"Running {d}...")
        df = loader.get_day_df(folder_path=folder_path, temp_path=temp_path, d=d, bucket_size=bucket_size,
                               tickers=tickers)
        start_time = constants.START_TRADE + constants.VOLATILE_TIMEFRAME
        end_time = constants.END_TRADE - constants.VOLATILE_TIMEFRAME - os_size - in_sample_size + 1
        for t in range(start_time, end_time + 1, rolling):
            train_df = df[(df['start_time'] >= t) & (df['start_time'] < t + in_sample_size)]
            test_df = df[(df['start_time'] >= t + in_sample_size) & (df['start_time'] < t + in_sample_size + os_size)]
            model = create_model()
            model.fit(train_df)
            model.score_test(test_df)
            # models.append(model)
            ins_r2.append(model.get_adj_r2())
            oos_r2.append(model.get_oos_r2())

    avg_r2_ins = mean(ins_r2)
    avg_r2_oos = mean(oos_r2)

    print(f'R2 In Sample : {avg_r2_ins}')
    print(f'R2 Out of Sample : {avg_r2_oos}')
