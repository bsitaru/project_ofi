import os
import py7zr
from data_manipulation.file_filters import SplitOFIArchiveFilter
from data_manipulation.data_file import SplitOFICSVFile
from datetime import date


def fn_filter_dates(start_date: date = None, end_date: date = None):
    def filter_dates(d: date):
        if start_date is not None and d < start_date:
            return False
        if end_date is not None and d > end_date:
            return False
        return True

    return filter_dates


def get_dates_from_archive_files(folder_path: str, tickers: list[str] = None, start_date: date = None,
                                 end_date: date = None):
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

    dates = list(filter(fn_filter_dates(start_date, end_date), list(dates)))
    return sorted(dates)


def get_dates_from_folder(folder_path: str, tickers: list[str] = None, start_date: date = None, end_date: date = None):
    if tickers is None:
        tickers = os.listdir(folder_path)
        tickers = list(filter(lambda x: os.path.isdir(os.path.join(folder_path, x)), tickers))

    dates = set()
    for t in tickers:
        ticker_path = os.path.join(folder_path, t)
        if not os.path.exists(ticker_path):
            continue
        ticker_dates = os.listdir(ticker_path)
        for d in ticker_dates:
            if d.endswith('.csv'):
                try:
                    dates.add(date.fromisoformat(d[:-4]))
                except:
                    pass

    dates = list(filter(fn_filter_dates(start_date, end_date), list(dates)))
    return sorted(dates)


def get_dates_in_majority_from_folder(folder_path: str, tickers: list[str], start_date: date = None,
                                      end_date: date = None):
    dates = {}
    for t in tickers:
        t_dates = get_dates_from_folder(folder_path=folder_path, tickers=[t], start_date=start_date, end_date=end_date)
        for d in t_dates:
            if d not in dates:
                dates[d] = 0
            dates[d] += 1

    majority = len(tickers) // 2 + 1
    majority_dates = []
    for (d, num) in dates.items():
        if num >= majority:
            majority_dates.append(d)
    return sorted(majority_dates)
