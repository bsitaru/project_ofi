import os
import shutil
import py7zr

from datetime import date
from functools import partial

from data_manipulation.bucket_ofi import compute_bucket_ofi_from_files, get_new_bucket_ofi_file_name, \
    get_bucket_size_from_file_name, get_bucket_ofi_df, compute_bucket_ofi_df_from_bucket_ofi, BucketOFIProps
from data_manipulation.multiday_bucket_ofi import prepare_df_for_multiday, MultidayProps


class FileFilter:
    def __init__(self, start_date: date = None, end_date: date = None, levels: int = None, tickers: list[str] = None):
        self.start_date = start_date
        self.end_date = end_date
        self.levels = levels
        self.tickers = tickers

    def filter(self, start_date: date = None, end_date: date = None, levels: int = None, ticker: str = None):
        if start_date is not None and self.end_date is not None and start_date > self.end_date:
            return False
        if end_date is not None and self.start_date is not None and end_date < self.start_date:
            return False
        if levels is not None and self.levels is not None and levels != self.levels:
            return False
        if ticker is not None and self.tickers is not None and ticker not in self.tickers:
            return False
        return True


def is_valid_archive(file_name: str, folder_path: str, flt: FileFilter) -> bool:
    file_path = os.path.join(folder_path, file_name)
    if not os.path.isfile(file_path):
        return False
    if not file_name.endswith('.7z'):
        return False
    s = file_name[:-3].split('_')
    if len(s) != 10:
        return False
    [ticker, start_date, end_date, levels] = s[6: 10]
    levels = int(levels) if levels.isdigit() else 0
    start_date = date.fromisoformat(start_date)
    end_date = date.fromisoformat(end_date)

    if not flt.filter(start_date=start_date, end_date=end_date, levels=levels, ticker=ticker):
        return False
    return True


def is_valid_data_file_name(file_name: str, flt: FileFilter):
    if not file_name.endswith('.csv'):
        return False
    s = file_name[:-4].split('_')
    if len(s) != 6:
        return False

    [_, d, _, _, file_type, _] = s
    d = date.fromisoformat(d)

    if not flt.filter(start_date=d, end_date=d):
        return False
    if file_type not in ['message', 'orderbook']:
        return False
    return True


def is_valid_split_ofi_file_name(file_name: str, flt: FileFilter):
    if not file_name.endswith('.csv'):
        return False
    s = file_name[:-4].split('_')
    if len(s) != 4:
        return False
    [ticker, d, file_type, lvl] = s
    d = date.fromisoformat(d)

    if not flt.filter(start_date=d, end_date=d, ticker=ticker):
        return False
    if not file_type.startswith('SplitOFIBucket'):
        return False
    return True


def filter_file_list(file_list: list[str], filter_fn):
    return list(filter(filter_fn, file_list))


def process_extracted_archive(folder_path: str, out_path: str, bucket_ofi_props: BucketOFIProps,
                              remove_after_process: bool = False) -> ():
    file_list = os.listdir(path=folder_path)

    for file_name in file_list:
        s = file_name.split('.')[0]
        [ticker, d, _, _, file_type, lvl] = s.split('_')
        if file_type == 'message':
            print(f"Processing {file_name}...")
            message_file_name = file_name
            orderbook_file_name = file_name.replace('message', 'orderbook')
            message_file_path = os.path.join(folder_path, message_file_name)
            orderbook_file_path = os.path.join(folder_path, orderbook_file_name)
            df = compute_bucket_ofi_from_files(message_file=message_file_path, orderbook_file=orderbook_file_path,
                                               props=bucket_ofi_props)
            new_file_name = get_new_bucket_ofi_file_name(ticker=ticker, d=d, props=bucket_ofi_props)
            new_file_path = os.path.join(out_path, new_file_name)
            df.to_csv(new_file_path, index=False)
            if remove_after_process:
                os.remove(message_file_path)
                os.remove(orderbook_file_path)


def process_archive_folder(folder_path: str, temp_path: str, out_path: str, flt: FileFilter,
                           bucket_ofi_props: BucketOFIProps, remove_after_process: bool = False) -> ():
    folder_path = os.path.abspath(folder_path)
    temp_path = os.path.abspath(temp_path)

    # Find valid archives
    file_list = os.listdir(path=folder_path)
    archives = filter_file_list(file_list, filter_fn=partial(is_valid_archive, folder_path=folder_path, flt=flt))

    for archive_name in archives:
        try:
            print(f"Processing Archive {archive_name}...")

            temp_folder = os.path.join(temp_path, archive_name)
            if os.path.exists(temp_folder):
                print(f"Archive already extracted {archive_name}")
            else:
                os.mkdir(temp_folder)
                archive_path = os.path.join(folder_path, archive_name)
                archive = py7zr.SevenZipFile(archive_path)
                file_list = archive.getnames()
                file_list = filter_file_list(file_list,
                                             filter_fn=partial(is_valid_data_file_name, flt=flt))
                archive.extract(path=temp_folder, targets=file_list)

            process_extracted_archive(folder_path=temp_folder, out_path=out_path, bucket_ofi_props=bucket_ofi_props,
                                      remove_after_process=remove_after_process)
            if remove_after_process:
                shutil.rmtree(temp_folder)
        except FileExistsError:
            print(f"Archive is already processing? {archive_name}")


def process_split_ofi_folder(folder_path: str, out_file: str, flt: FileFilter, bucket_ofi_props: BucketOFIProps):
    file_list = os.listdir(path=folder_path)
    files = filter_file_list(file_list, filter_fn=partial(is_valid_split_ofi_file_name, flt=flt))

    if os.path.exists(out_file):
        os.remove(out_file)

    for file_name in files:
        print(f"Processing {file_name}...")
        file_path = os.path.join(folder_path, file_name)
        bucket_ofi_props.prev_bucket_size = get_bucket_size_from_file_name(file_name)
        df = get_bucket_ofi_df(file_path)
        df = compute_bucket_ofi_df_from_bucket_ofi(df, props=bucket_ofi_props)
        df = prepare_df_for_multiday(df, file_name, props=MultidayProps(bucket_size=bucket_ofi_props.bucket_size))
        if os.path.exists(out_file):
            df.to_csv(out_file, mode='a', index=False, header=False)
        else:
            df.to_csv(out_file, index=False)
