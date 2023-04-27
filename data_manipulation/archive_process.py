import os
import sys
import shutil
import py7zr

from abc import ABC, abstractmethod
from datetime import date
from functools import partial
from joblib import Parallel, delayed

from data_manipulation.bucket_ofi import compute_bucket_ofi_from_files, get_new_bucket_ofi_file_name, \
    get_bucket_ofi_df, compute_bucket_ofi_df_from_bucket_ofi, BucketOFIProps
from data_manipulation.multiday_bucket_ofi import prepare_df_for_multiday
from data_manipulation.file_filters import FileFilter, L3ArchiveFilter, L3FileFilter, SplitOFIFileFilter, \
    SplitOFIArchiveFilter, intersect_dates
from data_manipulation.message import get_message_df
from data_manipulation.orderbook import get_orderbook_df
from data_manipulation.tick_ofi import compute_tick_ofi_df

VERBOSE = True


def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class ExtractedArchiveProcessor(ABC):
    def __init__(self, parallel_jobs: int = 1, verbose: bool = VERBOSE):
        self.parallel_jobs = parallel_jobs
        self.verbose = verbose
        self.file_filter = None

    def process_folder(self, folder_path: str, out_path: str):
        file_list = sorted(os.listdir(path=folder_path))
        if self.file_filter is not None:
            file_list = self.file_filter.filter_list(file_list)
        Parallel(n_jobs=self.parallel_jobs)(delayed(self.process_file)(f, folder_path, out_path) for f in file_list)

    @abstractmethod
    def process_file(self, file_name: str, folder_path: str, out_path: str):
        pass

    @abstractmethod
    def get_new_archive_name(self, old_archive_name: str):
        pass


class SplitOFIToExtractedProcessor(ExtractedArchiveProcessor):
    def __init__(self, file_filter: FileFilter, parallel_jobs: int = 1, verbose: bool = VERBOSE):
        super().__init__(parallel_jobs, verbose)
        self.file_filter = file_filter

    def process_file(self, file_name: str, folder_path: str, out_path: str):
        [ticker, d, _, _] = file_name[:-4].split('_')
        file_path = os.path.join(folder_path, file_name)
        dir_path = os.path.join(out_path, ticker)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        dest_path = os.path.join(dir_path, f"{d}.csv")
        shutil.copy(file_path, dest_path)

    def get_new_archive_name(self, old_archive_name: str):
        pass


class L3ToSplitOFIFileProcessor(ExtractedArchiveProcessor):
    def __init__(self, bucket_ofi_props: BucketOFIProps, file_filter: FileFilter, parallel_jobs: int = 1,
                 verbose: bool = VERBOSE):
        super().__init__(parallel_jobs, verbose)
        self.bucket_ofi_props = bucket_ofi_props
        self.file_filter = file_filter

    def process_file(self, file_name: str, folder_path: str, out_path: str):
        [ticker, d, _, _, file_type, lvl] = file_name[:-4].split('_')
        if file_type == 'message':
            if self.verbose:
                log(f"Processing {file_name}...")
            message_file_name = file_name
            orderbook_file_name = file_name.replace('message', 'orderbook')
            message_file_path = os.path.join(folder_path, message_file_name)
            orderbook_file_path = os.path.join(folder_path, orderbook_file_name)
            df = compute_bucket_ofi_from_files(message_file=message_file_path, orderbook_file=orderbook_file_path,
                                               props=self.bucket_ofi_props)
            if df.empty:  # Do not save file if empty
                return
            new_file_name = get_new_bucket_ofi_file_name(ticker=ticker, d=d, props=self.bucket_ofi_props)
            new_file_path = os.path.join(out_path, new_file_name)
            df.to_csv(new_file_path, index=False)

    def get_new_archive_name(self, old_archive_name: str):
        [ticker, start_date, end_date, levels] = old_archive_name[:-3].split('_')[6: 10]
        start_date = date.fromisoformat(start_date)
        end_date = date.fromisoformat(end_date)

        start_date, end_date = intersect_dates(start_date, end_date, self.file_filter.start_date,
                                               self.file_filter.end_date)
        levels = int(levels)

        name = f"{ticker}_{start_date}_{end_date}_SplitOFIBucket{self.bucket_ofi_props.bucket_size}_{levels}.7z"
        return name


class SplitOFIToSplitOFIFileProcessor(ExtractedArchiveProcessor):
    def __init__(self, bucket_ofi_props: BucketOFIProps, file_filter: FileFilter, parallel_jobs: int = 1,
                 verbose: bool = VERBOSE):
        super().__init__(parallel_jobs, verbose)
        self.bucket_ofi_props = bucket_ofi_props
        self.file_filter = file_filter

    def process_file(self, file_name: str, folder_path: str, out_path: str):
        [ticker, d, _, _] = file_name[:-4].split('_')
        if self.verbose:
            log(f"Processing {file_name}...")
        file_path = os.path.join(folder_path, file_name)
        df = get_bucket_ofi_df(file_path)
        df = compute_bucket_ofi_df_from_bucket_ofi(df, props=self.bucket_ofi_props)
        new_file_name = get_new_bucket_ofi_file_name(ticker=ticker, d=d, props=self.bucket_ofi_props)
        new_file_path = os.path.join(out_path, new_file_name)
        df.to_csv(new_file_path, index=False)

    def get_new_archive_name(self, old_archive_name: str):
        [ticker, start_date, end_date, _, levels] = old_archive_name[:-3].split('_')
        start_date = date.fromisoformat(start_date)
        end_date = date.fromisoformat(end_date)

        start_date, end_date = intersect_dates(start_date, end_date, self.file_filter.start_date,
                                               self.file_filter.end_date)
        levels = int(levels)

        rolling = '' if self.bucket_ofi_props.rolling_size == self.bucket_ofi_props.bucket_size else f"rol{self.bucket_ofi_props.rolling_size}"

        name = f"{ticker}_{start_date}_{end_date}_SplitOFIBucket{self.bucket_ofi_props.bucket_size}{rolling}_{levels}.7z"
        return name


class DataAssertFileProcessor(ExtractedArchiveProcessor):
    def __init__(self, file_filter: FileFilter, assert_fn, parallel_jobs: int = 1, verbose: bool = VERBOSE):
        super().__init__(parallel_jobs, verbose)
        self.file_filter = file_filter
        self.assert_fn = assert_fn

    def process_file(self, file_name: str, folder_path: str, out_path: str):
        [ticker, d, _, _, file_type, lvl] = file_name[:-4].split('_')
        if file_type == 'message':
            if self.verbose:
                log(f"Processing {file_name}...")
            message_file_name = file_name
            orderbook_file_name = file_name.replace('message', 'orderbook')
            message_file_path = os.path.join(folder_path, message_file_name)
            orderbook_file_path = os.path.join(folder_path, orderbook_file_name)
            message_df = get_message_df(message_file_path)
            orderbook_df = get_orderbook_df(orderbook_file_path, levels=self.file_filter.levels)
            tick_ofi_df = compute_tick_ofi_df(message_df=message_df, orderbook_df=orderbook_df,
                                              levels=self.file_filter.levels, trading_hours_only=True)
            # tick_ofi_df.to_csv(os.path.join(out_path, 'tick_temp.csv'), index=False)
            for _, row in tick_ofi_df.iterrows():
                self.assert_fn(row)

    def get_new_archive_name(self, old_archive_name: str):
        pass


class SplitOFIToMultidayFileProcessor(ExtractedArchiveProcessor):
    def __init__(self, verbose: bool = VERBOSE):
        # Cannot be done in parallel
        super().__init__(parallel_jobs=1, verbose=verbose)

    def process_file(self, file_name: str, folder_path: str, out_path: str):
        out_file = out_path
        if self.verbose:
            log(f"Processing {file_name}...")
        file_path = os.path.join(folder_path, file_name)
        df = get_bucket_ofi_df(file_path)
        df = prepare_df_for_multiday(df, file_name)
        if os.path.exists(out_path):
            df.to_csv(out_file, mode='a', index=False, header=False)
        else:
            df.to_csv(out_file, index=False)

    def get_new_archive_name(self, old_archive_name: str):
        # Not available for this type of processor
        pass


class ArchiveProcessor(ABC):
    def __init__(self, verbose: bool = VERBOSE):
        self.verbose = verbose
        self.archive_filter = None
        self.file_filter = None

    def _extract_files_from_archive(self, archive_name: str, folder_path: str, out_path: str, flt: FileFilter) -> ():
        if os.path.exists(out_path):
            if self.verbose:
                log(f"Archive already extracted {archive_name}")
            return

        os.mkdir(out_path)
        archive_path = os.path.join(folder_path, archive_name)
        archive = py7zr.SevenZipFile(archive_path)
        file_list = archive.getnames()
        file_list = flt.filter_list(file_list)
        archive.extract(path=out_path, targets=file_list)

    def _create_archive(self, archive_name: str, folder_path: str, out_path: str):
        if self.verbose:
            log(f"Creating archive {archive_name}...")
        new_archive_path = os.path.join(out_path, archive_name)
        archive = py7zr.SevenZipFile(new_archive_path, 'w')
        file_list = os.listdir(folder_path)
        for file_name in file_list:
            archive.write(os.path.join(folder_path, file_name), arcname=file_name)
        archive.close()

    def process_archive(self, folder_path: str, temp_path: str, out_path: str, remove_after_process: bool,
                        archive_output: bool, extracted_archive_processor: ExtractedArchiveProcessor, parallel_jobs: int):
        folder_path = os.path.abspath(folder_path)
        temp_path = os.path.abspath(temp_path)
        if out_path is not None:
            out_path = os.path.abspath(out_path)

        # Find valid archives
        file_list = os.listdir(path=folder_path)
        archive_list = self.archive_filter.filter_list(file_list)

        def process(archive_name):
            try:
                if self.verbose:
                    log(f"Processing Archive {archive_name}...")

                temp_folder = os.path.join(temp_path, archive_name)
                self._extract_files_from_archive(archive_name=archive_name, folder_path=folder_path,
                                                 out_path=temp_folder, flt=self.file_filter)
                # filter_fn=partial(is_valid_data_file_name, flt=flt))

                if archive_output:
                    new_archive_name = extracted_archive_processor.get_new_archive_name(old_archive_name=archive_name)
                    new_out_path = os.path.join(out_path, new_archive_name[:-3])  # Without .7z extension
                    if not os.path.exists(new_out_path):
                        os.mkdir(new_out_path)

                out_folder = new_out_path if archive_output else out_path

                if extracted_archive_processor is not None:
                    extracted_archive_processor.process_folder(folder_path=temp_folder, out_path=out_folder)

                if remove_after_process:
                    shutil.rmtree(temp_folder)

                if archive_output:
                    self._create_archive(archive_name=new_archive_name, folder_path=out_folder, out_path=out_path)
                    if remove_after_process:
                        shutil.rmtree(new_out_path)

            except FileExistsError:
                log(f"Archive is already processing? {archive_name}")

        # for archive_name in archive_list:
        #     process(archive_name)
        Parallel(n_jobs=parallel_jobs)(delayed(process)(archive_name) for archive_name in archive_list)


class L3ArchiveProcessor(ArchiveProcessor):
    def __init__(self, verbose: bool = VERBOSE, start_date: date = None, end_date: date = None, levels: int = None,
                 tickers: list[str] = None):
        super().__init__(verbose=verbose)
        self.archive_filter = L3ArchiveFilter(start_date=start_date, end_date=end_date, levels=levels, tickers=tickers)
        self.file_filter = L3FileFilter(start_date=start_date, end_date=end_date, levels=levels, tickers=tickers)


class SplitOFIArchiveProcessor(ArchiveProcessor):
    def __init__(self, verbose: bool = VERBOSE, start_date: date = None, end_date: date = None, levels: int = None,
                 tickers: list[str] = None):
        super().__init__(verbose=verbose)
        self.archive_filter = SplitOFIArchiveFilter(start_date=start_date, end_date=end_date, levels=levels,
                                                    tickers=tickers)
        self.file_filter = SplitOFIFileFilter(start_date=start_date, end_date=end_date, levels=levels, tickers=tickers)
