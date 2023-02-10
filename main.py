from datetime import date
from data_manipulation.archive_process import process_archive_folder, process_split_ofi_archive_folder, FileFilter, \
    BucketOFIProps

from models.split_ofi import run_model

from models.split_ofi_current_return import SplitOFICurrentReturn

from constants import TICKERS


def process_archive():
    flt = FileFilter(start_date=date.fromisoformat('2016-03-01'),
                     end_date=date.fromisoformat('2016-04-01'),
                     tickers=['TFX', 'PSA'],
                     levels=10)
    bucket_ofi_props = BucketOFIProps(levels=10,
                                      bucket_size=1,
                                      rounding=True)
    process_archive_folder(folder_path='./lobster_sample/test_data',
                           temp_path='./lobster_sample/temp_data',
                           out_path='./lobster_sample/data2',
                           flt=flt,
                           bucket_ofi_props=bucket_ofi_props,
                           remove_after_process=True,
                           create_archive=True,
                           parallel_jobs=4)


def process_multiday():
    flt = FileFilter(start_date=date.fromisoformat('2016-03-01'),
                     end_date=date.fromisoformat('2016-03-10'),
                     tickers=['TFX', 'PSA'],
                     levels=10)
    bucket_ofi_props = BucketOFIProps(levels=10,
                                      bucket_size=30 * 60,
                                      rounding=True)
    process_split_ofi_archive_folder(folder_path='./lobster_sample/data2',
                                     out_file='./lobster_sample/combined2.csv',
                                     temp_path='./lobster_sample/temp_data',
                                     flt=flt,
                                     bucket_ofi_props=bucket_ofi_props,
                                     remove_after_process=True)


def run_lr():
    model = SplitOFICurrentReturn(train_data_file='./lobster_sample/combined2.csv', levels=1)
    model.fit_and_score()
    print(model.model.intercept_)


if __name__ == '__main__':
    run_lr()
