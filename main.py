from datetime import date
from data_manipulation.archive_process import process_archive_folder, process_split_ofi_folder

from models.split_ofi import run_model


def process_archive():
    process_archive_folder(folder_path='./lobster_sample/test_data',
                           temp_path='./lobster_sample/temp_data',
                           out_path='./lobster_sample/data',
                           levels=10,
                           tickers=['TFX'],
                           date_interval=(date.fromisoformat('2016-03-01'), date.fromisoformat('2016-03-10')),
                           bucket_size=1,
                           remove_after_process=True)


def process_multiday():
    process_split_ofi_folder(folder_path='./lobster_sample/data',
                             out_file='./lobster_sample/combined2.csv',
                             levels=10,
                             tickers=['TFX', 'PSA'],
                             date_interval=(date.fromisoformat('2016-03-01'), date.fromisoformat('2016-03-10')),
                             bucket_size=30 * 60)


def run_lr():
    run_model(train_data_path='./lobster_sample/combined2.csv',
              test_data_path='./lobster_sample/combined2.csv',
              levels=1)


if __name__ == '__main__':
    run_lr()
