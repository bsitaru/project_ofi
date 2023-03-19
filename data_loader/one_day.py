import os
import shutil

from data_manipulation.archive_process import SplitOFIArchiveProcessor
from data_manipulation.multiday_bucket_ofi import get_multiday_df
from data_manipulation.bucket_ofi import get_bucket_ofi_df
from datetime import date
import data_process

import random
import string


def get_single_date_df_for_ticker(folder_path: str, temp_path: str, d: date, ticker: str):
    temp_folder = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
    temp_path = os.path.join(temp_path, temp_folder)
    os.mkdir(temp_path)
    archive_processor = SplitOFIArchiveProcessor(verbose=False, start_date=d, end_date=d, tickers=[ticker])
    archive_processor.process_archive(folder_path=folder_path, temp_path=temp_path, out_path=None,
                                      remove_after_process=False, archive_output=False,
                                      extracted_archive_processor=None)

    file_list = os.listdir(temp_path)
    file_path = ''
    if len(file_list) > 0:
        file_path = os.path.join(temp_path, file_list[0])
        file_list = os.listdir(file_path)
        if len(file_list) > 0:
            file_path = os.path.join(file_path, file_list[0])
    df = get_bucket_ofi_df(file_path)

    shutil.rmtree(temp_path)
    return df


def get_single_day_df(folder_path: str, temp_path: str, d: date, tickers: list[str] = None):
    temp_file = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20)) + '_' + str(d) + '.csv'
    temp_file = os.path.join(temp_path, temp_file)

    temp_folder = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
    temp_path = os.path.join(temp_path, temp_folder)
    os.mkdir(temp_path)

    data_process.multiday_process(folder_path=folder_path, temp_path=temp_path, out_file=temp_file,
                                  start_date=d, end_date=d, tickers=tickers, verbose=False)
    df = get_multiday_df(temp_file)
    if os.path.exists(temp_file):
        os.remove(temp_file)
    shutil.rmtree(temp_path)
    return df
