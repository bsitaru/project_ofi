import shutil

import numpy as np
import pandas as pd
import os
import py7zr

from data_manipulation.data_file import SplitOFICSVFile
from data_manipulation.file_filters import SplitOFIArchiveFilter, SplitOFIFileFilter
from data_manipulation.multiday_bucket_ofi import get_multiday_df
from datetime import date
import data_process

import random
import string


def get_dates_from_archive_files(folder_path: str, tickers: list[str] = None):
    file_list = os.listdir(folder_path)
    archive_filter = SplitOFIArchiveFilter(tickers=tickers)
    archives = archive_filter.filter_list(file_list)
    dates = set()
    for name in archives:
        archive = py7zr.SevenZipFile(os.path.join(folder_path, name))
        file_list = archive.getnames()
        for csv_name in file_list:
            try:
                f = SplitOFICSVFile(csv_name)
                dates.add(f.d)
            except ValueError:
                pass
    return sorted(list(dates))


def get_day_df(folder_path: str, temp_path: str, d: date, bucket_size: int, tickers: list[str] = None):
    temp_file = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20)) + '_' + str(d) + '.csv'
    temp_file = os.path.join(temp_path, temp_file)
    tickers = None if tickers is None else ' '.join(tickers)
    data_process.multiday(folder_path=folder_path, temp_path=temp_path, out_file=temp_file, bucket_size=bucket_size,
                          start_date=str(d), end_date=str(d), tickers=tickers, verbose=False)
    df = get_multiday_df(temp_file)
    os.remove(temp_file)
    return df
