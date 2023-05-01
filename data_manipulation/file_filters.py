from datetime import date
from abc import abstractmethod, ABC
import data_manipulation.data_file as data_file


def intersect_dates(l1: date, r1: date, l2: date, r2: date):
    if l2 is not None and l1 < l2:
        l1 = l2
    if r2 is not None and r2 < r1:
        r1 = r2
    return l1, r1


class FileFilter(ABC):
    def __init__(self, start_date: date = None, end_date: date = None, levels: int = None, tickers: list[str] = None):
        self.start_date = start_date
        self.end_date = end_date
        self.levels = levels
        self.tickers = tickers

    def _is_valid(self, start_date: date = None, end_date: date = None, levels: int = None, ticker: str = None):
        if start_date is not None and self.end_date is not None and start_date > self.end_date:
            return False
        if end_date is not None and self.start_date is not None and end_date < self.start_date:
            return False
        if levels is not None and self.levels is not None and levels != self.levels:
            return False
        if ticker is not None and self.tickers is not None and ticker not in self.tickers:
            return False
        return True

    def filter_list(self, l: list[str]):
        return sorted(list(filter(self.filter_file, l)))

    def filter_file(self, file_name: str):
        try:
            file = self.data_file_type(file_name)
            return self.is_valid(file)
        except ValueError as e:
            return False
        pass

    @abstractmethod
    def is_valid(self, file: data_file.DataFile):
        pass

    @property
    @abstractmethod
    def data_file_type(self):
        pass


class ArchiveFilter(FileFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_valid(self, file: data_file.ArchiveFile):
        return super()._is_valid(start_date=file.start_date, end_date=file.end_date, levels=file.levels,
                                 ticker=file.ticker)

    @property
    @abstractmethod
    def data_file_type(self):
        pass


class CSVFilter(FileFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_valid(self, file: data_file.CSVFile):
        return super()._is_valid(start_date=file.d, end_date=file.d, levels=file.levels, ticker=file.ticker)

    @property
    @abstractmethod
    def data_file_type(self):
        pass


class L3ArchiveFilter(ArchiveFilter):
    data_file_type = data_file.L3ArchiveFile

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class L3FileFilter(CSVFilter):
    data_file_type = data_file.L3CSVFile

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SplitOFIArchiveFilter(ArchiveFilter):
    data_file_type = data_file.SplitOFIArchiveFile

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SplitOFIFileFilter(CSVFilter):
    data_file_type = data_file.SplitOFICSVFile

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PricesArchiveFilter(ArchiveFilter):
    data_file_type = data_file.PricesArchiveFile

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PricesFileFilter(CSVFilter):
    data_file_type = data_file.PricesCSVFile

    def __init__(self, **kwargs):
        super().__init__(**kwargs)