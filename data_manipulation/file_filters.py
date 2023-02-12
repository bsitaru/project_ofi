from datetime import date
from abc import abstractmethod, ABC

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

    @abstractmethod
    def filter_file(self, file_name: str):
        pass


class L3ArchiveFilter(FileFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def filter_file(self, file_name: str):
        if not file_name.endswith('.7z'):
            return False
        s = file_name[:-3].split('_')
        if len(s) != 10:
            return False
        [ticker, start_date, end_date, levels] = s[6: 10]
        levels = int(levels) if levels.isdigit() else 0
        start_date = date.fromisoformat(start_date)
        end_date = date.fromisoformat(end_date)

        if not self._is_valid(start_date=start_date, end_date=end_date, levels=levels, ticker=ticker):
            return False
        return True


class L3FileFilter(FileFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def filter_file(self, file_name: str):
        if not file_name.endswith('.csv'):
            return False
        s = file_name[:-4].split('_')
        if len(s) != 6:
            return False

        [ticker, d, _, _, file_type, levels] = s
        levels = int(levels) if levels.isdigit() else 0
        d = date.fromisoformat(d)

        if not self._is_valid(start_date=d, end_date=d, levels=levels, ticker=ticker):
            return False
        if file_type not in ['message', 'orderbook']:
            return False
        return True


class SplitOFIArchiveFilter(FileFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def filter_file(self, file_name: str):
        if not file_name.endswith('.7z'):
            return False
        s = file_name[:-3].split('_')
        if len(s) != 5:
            return False
        [ticker, start_date, end_date, file_type, levels] = s
        levels = int(levels) if levels.isdigit() else 0
        start_date = date.fromisoformat(start_date)
        end_date = date.fromisoformat(end_date)

        if not self._is_valid(start_date=start_date, end_date=end_date, levels=levels, ticker=ticker):
            return False
        if not file_type.startswith('SplitOFIBucket'):
            return False
        return True


class SplitOFIFileFilter(FileFilter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def filter_file(self, file_name: str):
        if not file_name.endswith('.csv'):
            return False
        s = file_name[:-4].split('_')
        if len(s) != 4:
            return False
        [ticker, d, file_type, levels] = s
        levels = int(levels) if levels.isdigit() else 0
        d = date.fromisoformat(d)

        if not self._is_valid(start_date=d, end_date=d, ticker=ticker, levels=levels):
            return False
        if not file_type.startswith('SplitOFIBucket'):
            return False
        return True
