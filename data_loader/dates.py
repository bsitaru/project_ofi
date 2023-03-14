import os
import py7zr
from data_manipulation.file_filters import SplitOFIArchiveFilter
from data_manipulation.data_file import SplitOFICSVFile


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
