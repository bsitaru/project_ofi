import os
import shutil

from datetime import date

from data_manipulation.bucket_ofi import compute_bucket_ofi_from_files, get_new_bucket_ofi_file_name, get_bucket_size_from_file_name, \
    get_bucket_ofi_df, compute_bucket_ofi_df_from_bucket_ofi
from data_manipulation.multiday_bucket_ofi import prepare_df_for_multiday


# Returns (ticker, start_date, end_date, levels) for the current archive.
# Returns None if the file is not an archive or is an invalid archive.
def is_valid_archive(folder_path: str, file_name: str, date_interval: (date, date) = None, levels: int = None,
                     tickers: list[str] = None, **kwargs) -> bool:
    file_path = os.path.join(folder_path, file_name)
    if not os.path.isfile(file_path):
        return False
    if not file_name.endswith('.7z'):
        return False
    s = file_name[:-3].split('_')
    if len(s) != 10:
        return False
    [ticker, start_date, end_date, lvl] = s[6: 10]
    lvl = int(lvl) if lvl.isdigit() else 0
    start_date = date.fromisoformat(start_date)
    end_date = date.fromisoformat(end_date)

    if date_interval is not None:
        left, right = date_interval
        if end_date < left or start_date > right:
            return False

    if tickers is not None:
        if ticker not in tickers:
            return False

    if levels is not None:
        if lvl != levels:
            return False

    return True


def extract_archive(folder_path: str, file_name: str, temp_path: str) -> str:
    file_path = os.path.join(folder_path, file_name)
    temp_folder = os.path.join(temp_path, file_name)
    os.mkdir(temp_folder)
    cmd = f"7za e '{file_path}' -o'{temp_folder}'"
    os.system(cmd)
    return temp_folder


def is_valid_data_file(folder_path: str, file_name: str, date_interval: (date, date) = None, **kwargs):
    file_path = os.path.join(folder_path, file_name)
    if not os.path.isfile(file_path):
        return False
    if not file_name.endswith('.csv'):
        return False
    s = file_name[:-4].split('_')
    if len(s) != 6:
        return False

    [_, d, _, _, file_type, _] = s
    d = date.fromisoformat(d)

    if date_interval is not None:
        left, right = date_interval
        if d < left or d > right:
            return False

    if file_type not in ['message', 'orderbook']:
        return False

    return True


def process_extracted_archive(folder_path: str, out_path: str, remove_after_process: bool = False, **kwargs) -> ():
    file_list = os.listdir(path=folder_path)
    data_files = []
    for file_name in file_list:
        if is_valid_data_file(folder_path=folder_path, file_name=file_name, **kwargs):
            data_files.append(file_name)
        else:
            # if remove_after_process:
            #     os.remove(os.path.join(folder_path, file_name))
            pass

    for file_name in data_files:
        s = file_name.split('.')[0]
        [ticker, d, _, _, file_type, lvl] = s.split('_')
        if file_type == 'message':
            print(f"Processing {file_name}...")
            message_file_name = file_name
            orderbook_file_name = file_name.replace('message', 'orderbook')
            message_file_path = os.path.join(folder_path, message_file_name)
            orderbook_file_path = os.path.join(folder_path, orderbook_file_name)
            df = compute_bucket_ofi_from_files(message_file=message_file_path, orderbook_file=orderbook_file_path,
                                               **kwargs)
            new_file_name = get_new_bucket_ofi_file_name(ticker=ticker, d=d, **kwargs)
            new_file_path = os.path.join(out_path, new_file_name)
            df.to_csv(new_file_path, index=False)
            if remove_after_process:
                os.remove(message_file_path)
                os.remove(orderbook_file_path)

    if remove_after_process:
        shutil.rmtree(folder_path)


def process_archive_folder(folder_path: str, temp_path: str, out_path: str, **kwargs) -> ():
    folder_path = os.path.abspath(folder_path)
    temp_path = os.path.abspath(temp_path)

    # Find valid archives
    file_list = os.listdir(path=folder_path)

    def filter_fn(file_name: str) -> bool:
        return is_valid_archive(folder_path=folder_path, file_name=file_name, **kwargs)

    archives = list(filter(filter_fn, file_list))

    for archive_name in archives:
        try:
            print(f"Extracting {archive_name}...")
            folder = extract_archive(folder_path=folder_path, file_name=archive_name, temp_path=temp_path)
            process_extracted_archive(folder_path=folder, out_path=out_path, **kwargs)
        except FileExistsError:
            print(f"Archive is already processing? {archive_name}")


def process_split_ofi_folder(folder_path: str, out_file: str, **kwargs):
    file_list = os.listdir(path=folder_path)

    def filter_fn(file_name: str, date_interval: (date, date) = None, tickers: list[str] = None) -> bool:
        file_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(file_path):
            return False
        if not file_name.endswith('.csv'):
            return False
        s = file_name[:-4].split('_')
        if len(s) != 4:
            return False
        [ticker, d, file_type, lvl] = s

        if date_interval is not None:
            left, right = date_interval
            if d < left or d > right:
                return False

        if tickers is not None:
            if ticker not in tickers:
                return False

        return True

    files = list(filter(filter_fn, file_list))

    if os.path.exists(out_file):
        os.remove(out_file)

    for file_name in files:
        print(f"Processing {file_name}...")
        file_path = os.path.join(folder_path, file_name)
        prev_bucket_size = get_bucket_size_from_file_name(file_name)
        df = get_bucket_ofi_df(file_path)
        df = compute_bucket_ofi_df_from_bucket_ofi(df, prev_bucket_size=prev_bucket_size, **kwargs)
        df = prepare_df_for_multiday(df, file_name, **kwargs)
        if os.path.exists(out_file):
            df.to_csv(out_file, mode='a', index=False, header=False)
        else:
            df.to_csv(out_file, index=False)
