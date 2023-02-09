from datetime import date
from data_manipulation.archive_process import process_archive_folder, process_split_ofi_folder, FileFilter, BucketOFIProps

from models.split_ofi import run_model


def process_archive():
    flt = FileFilter(start_date=date.fromisoformat('2016-03-01'),
                     end_date=date.fromisoformat('2016-03-05'),
                     tickers=['TFX'],
                     levels=10)
    bucket_ofi_props = BucketOFIProps(levels=10,
                                      bucket_size=1,
                                      rounding=True)
    process_archive_folder(folder_path='./lobster_sample/test_data',
                           temp_path='./lobster_sample/temp_data',
                           out_path='./lobster_sample/data',
                           flt=flt,
                           bucket_ofi_props=bucket_ofi_props,
                           remove_after_process=True)


def process_multiday():
    flt = FileFilter(start_date=date.fromisoformat('2016-03-01'),
                     end_date=date.fromisoformat('2016-03-05'),
                     tickers=['TFX'],
                     levels=10)
    bucket_ofi_props = BucketOFIProps(levels=10,
                                      bucket_size=30 * 60,
                                      rounding=True)
    process_split_ofi_folder(folder_path='./lobster_sample/data',
                             out_file='./lobster_sample/combined2.csv',
                             flt=flt,
                             bucket_ofi_props=bucket_ofi_props)


def run_lr():
    run_model(train_data_path='./lobster_sample/combined2.csv',
              test_data_path='./lobster_sample/combined2.csv',
              levels=1)


if __name__ == '__main__':
    run_lr()
