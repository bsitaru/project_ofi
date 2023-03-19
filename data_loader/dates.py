import os
import py7zr
from data_manipulation.file_filters import SplitOFIArchiveFilter
from data_manipulation.data_file import SplitOFICSVFile
from datetime import date


def get_dates_from_archive_files(folder_path: str, tickers: list[str] = None, start_date: date = None, end_date: date = None):
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

    def filter_dates(d: date):
        if start_date is not None and d < start_date:
            return False
        if end_date is not None and d > end_date:
            return False
        return True

    dates = list(filter(filter_dates, list(dates)))
    return sorted(dates)
