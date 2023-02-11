import os
import shutil
import py7zr

from datetime import date
from functools import partial
from joblib import Parallel, delayed

from data_manipulation.bucket_ofi import compute_bucket_ofi_from_files, get_new_bucket_ofi_file_name, \
    get_bucket_size_from_file_name, get_bucket_ofi_df, compute_bucket_ofi_df_from_bucket_ofi, BucketOFIProps
from data_manipulation.multiday_bucket_ofi import prepare_df_for_multiday, MultidayProps


VERBOSE = True


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


def is_valid_split_ofi_archive(file_name: str, flt: FileFilter) -> bool:
    if not file_name.endswith('.7z'):
        return False
    s = file_name[:-3].split('_')
    if len(s) != 5:
        return False
    [ticker, start_date, end_date, file_type, levels] = s
    levels = int(levels)
    start_date = date.fromisoformat(start_date)
    end_date = date.fromisoformat(end_date)

    if not flt.filter(start_date=start_date, end_date=end_date, levels=levels, ticker=ticker):
        return False
    return True


def filter_file_list(file_list: list[str], filter_fn):
    return list(filter(filter_fn, file_list))


def process_extracted_archive(folder_path: str, out_path: str, bucket_ofi_props: BucketOFIProps,
                              remove_after_process: bool, parallel_jobs: int) -> ():
    file_list = os.listdir(path=folder_path)
    file_list = sorted(file_list)

    def process_file(file_name: str):
        [ticker, d, _, _, file_type, lvl] = file_name[:-4].split('_')
        if file_type == 'message':
            if VERBOSE:
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

    Parallel(n_jobs=parallel_jobs)(delayed(process_file)(f) for f in file_list)


def get_new_archive_name(archive_name: str, flt: FileFilter, bucket_ofi_props: BucketOFIProps):
    [ticker, start_date, end_date, levels] = archive_name[:-3].split('_')[6: 10]
    start_date = date.fromisoformat(start_date)
    end_date = date.fromisoformat(end_date)

    def intersect_dates(l1: date, r1: date, l2: date, r2: date):
        if l2 is not None and l1 < l2:
            l1 = l2
        if r2 is not None and r2 < r1:
            r1 = r2
        return l1, r1

    start_date, end_date = intersect_dates(start_date, end_date, flt.start_date, flt.end_date)
    levels = int(levels)

    name = f"{ticker}_{start_date}_{end_date}_SplitOFIBucket{bucket_ofi_props.bucket_size}_{levels}.7z"
    return name


def extract_files_from_archive(archive_path: str, out_path: str, filter_fn) -> ():
    archive = py7zr.SevenZipFile(archive_path)
    file_list = archive.getnames()
    file_list = filter_file_list(file_list, filter_fn=filter_fn)
    archive.extract(path=out_path, targets=file_list)


def process_archive_folder(folder_path: str, temp_path: str, out_path: str, flt: FileFilter,
                           bucket_ofi_props: BucketOFIProps, remove_after_process: bool, create_archive: bool,
                           parallel_jobs: int) -> ():
    folder_path = os.path.abspath(folder_path)
    temp_path = os.path.abspath(temp_path)

    # Find valid archives
    file_list = os.listdir(path=folder_path)
    archives = filter_file_list(file_list, filter_fn=partial(is_valid_archive, folder_path=folder_path, flt=flt))
    archives = sorted(archives)

    for archive_name in archives:
        try:
            if VERBOSE:
                print(f"Processing Archive {archive_name}...")

            temp_folder = os.path.join(temp_path, archive_name)
            if os.path.exists(temp_folder):
                if VERBOSE:
                    print(f"Archive already extracted {archive_name}")
            else:
                os.mkdir(temp_folder)
                archive_path = os.path.join(folder_path, archive_name)
                extract_files_from_archive(archive_path=archive_path, out_path=temp_folder,
                                           filter_fn=partial(is_valid_data_file_name, flt=flt))

            if create_archive:
                new_archive_name = get_new_archive_name(archive_name=archive_name, flt=flt,
                                                        bucket_ofi_props=bucket_ofi_props)
                new_out_path = os.path.join(out_path, new_archive_name[:-3])  # Without .7z extension
                if not os.path.exists(new_out_path):
                    os.mkdir(new_out_path)

            out_folder = new_out_path if create_archive else out_path

            process_extracted_archive(folder_path=temp_folder, out_path=out_folder, bucket_ofi_props=bucket_ofi_props,
                                      remove_after_process=remove_after_process, parallel_jobs=parallel_jobs)
            if remove_after_process:
                shutil.rmtree(temp_folder)

            if create_archive:
                if VERBOSE:
                    print(f"Creating archive {new_archive_name}...")
                new_archive_path = os.path.join(out_path, new_archive_name)
                archive = py7zr.SevenZipFile(new_archive_path, 'w')
                file_list = os.listdir(new_out_path)
                for file_name in file_list:
                    archive.write(os.path.join(new_out_path, file_name), arcname=file_name)
                archive.close()
                if remove_after_process:
                    shutil.rmtree(new_out_path)
        except FileExistsError:
            print(f"Archive is already processing? {archive_name}")


def process_split_ofi_folder(folder_path: str, out_file: str, bucket_ofi_props: BucketOFIProps):
    file_list = os.listdir(path=folder_path)
    file_list = sorted(file_list)
    for file_name in file_list:
        if VERBOSE:
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


def process_split_ofi_archive_folder(folder_path: str, out_file: str, temp_path: str, flt: FileFilter,
                                     bucket_ofi_props: BucketOFIProps, remove_after_process: bool):
    file_list = os.listdir(path=folder_path)
    archive_list = filter_file_list(file_list, filter_fn=partial(is_valid_split_ofi_archive, flt=flt))
    archive_list = sorted(archive_list)

    if os.path.exists(out_file):
        os.remove(out_file)

    for archive_name in archive_list:
        if VERBOSE:
            print(f"Processing Archive {archive_name}...")
        temp_folder = os.path.join(temp_path, archive_name)
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)
        archive_path = os.path.join(folder_path, archive_name)
        extract_files_from_archive(archive_path=archive_path, out_path=temp_folder,
                                   filter_fn=partial(is_valid_split_ofi_file_name, flt=flt))
        process_split_ofi_folder(folder_path=temp_folder, out_file=out_file, bucket_ofi_props=bucket_ofi_props)
        if remove_after_process:
            shutil.rmtree(temp_folder)
