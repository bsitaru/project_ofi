from datetime import date


class DataFile:
    def __init__(self, file_name: str):
        self.file_name = file_name


class ArchiveFile(DataFile):
    def __init__(self, file_name: str):
        super().__init__(file_name)
        if not file_name.endswith('.7z'):
            raise ValueError(f'File is not an archive: {file_name}')
        self.archive_name = file_name[:-3]


class CSVFile(DataFile):
    def __init__(self, file_name: str):
        super().__init__(file_name)
        if not file_name.endswith('.csv'):
            raise ValueError(f'File is not a csv: {file_name}')
        self.csv_name = file_name[:-4]


class L3ArchiveFile(ArchiveFile):
    def __init__(self, file_name: str):
        super().__init__(file_name)
        s = self.archive_name.split('_')
        if len(s) != 10:
            raise ValueError(f"File is not an L3 archive: {file_name}")
        [ticker, start_date, end_date, levels] = s[6: 10]
        self.ticker = ticker
        self.levels = int(levels) if levels.isdigit() else 0
        self.start_date = date.fromisoformat(start_date)
        self.end_date = date.fromisoformat(end_date)


class L3CSVFile(CSVFile):
    def __init__(self, file_name):
        super().__init__(file_name)
        s = self.csv_name.split('_')
        if len(s) != 6:
            raise ValueError(f"File is not an L3 CSV: {file_name}")
        [ticker, d, _, _, file_type, levels] = s
        if file_type not in ['message', 'orderbook']:
            raise ValueError(f"File is not an L3 CSV: {file_name}")
        self.ticker = ticker
        self.levels = int(levels) if levels.isdigit() else 0
        self.d = date.fromisoformat(d)


class ProcessedArchiveFile(ArchiveFile):
    def __init__(self, file_name: str, arch_type: str):
        super().__init__(file_name)
        s = self.archive_name.split('_')
        if len(s) != 5:
            raise ValueError(f"File is not a {arch_type} archive: {file_name}")
        [ticker, start_date, end_date, file_type, levels] = s
        if not file_type.startswith(arch_type):
            raise ValueError(f"File is not a {arch_type} archive: {file_name}")
        self.ticker = ticker
        self.levels = int(levels) if levels.isdigit() else 0
        self.start_date = date.fromisoformat(start_date)
        self.end_date = date.fromisoformat(end_date)
        # self.bucket_size = int(file_type[14:]) if file_type[14:].isdigit() else 0


class SplitOFIArchiveFile(ProcessedArchiveFile):
    def __init__(self, file_name: str):
        super().__init__(file_name, arch_type='SplitOFIBucket')

class PricesArchiveFile(ProcessedArchiveFile):
    def __init__(self, file_name: str):
        super().__init__(file_name, arch_type='Prices')

class ProcessedCSVFile(CSVFile):
    def __init__(self, file_name, data_type):
        super().__init__(file_name)
        s = self.csv_name.split('_')
        if len(s) != 4:
            raise ValueError(f"File is not a {data_type} CSV: {file_name}")
        [ticker, d, file_type, levels] = s
        if not file_type.startswith(data_type):
            raise ValueError(f"File is not a {data_type} CSV: {file_name}")
        self.ticker = ticker
        self.levels = int(levels) if levels.isdigit() else 0
        self.d = date.fromisoformat(d)
        # self.bucket_size = int(file_type[14:]) if file_type[14:].isdigit() else 0

class SplitOFICSVFile(ProcessedCSVFile):
    def __init__(self, file_name):
        super().__init__(file_name, data_type='SplitOFIBucket')

class PricesCSVFile(ProcessedCSVFile):
    def __init__(self, file_name):
        super().__init__(file_name, data_type='Prices')
