import typer

import pandas as pd

from datetime import date
from data_manipulation.archive_process import L3ArchiveProcessor, L3ToSplitOFIFileProcessor, FileFilter, BucketOFIProps, \
    SplitOFIArchiveProcessor, SplitOFIToSplitOFIFileProcessor, SplitOFIToMultidayFileProcessor, DataAssertFileProcessor, \
    SplitOFIToExtractedProcessor, L3ToPricesFileProcessor, PricesArchiveProcessor
from constants import TICKERS, LEVELS

main = typer.Typer()


def process_date(d: str) -> date:
    return date.fromisoformat(d) if d is not None else None


def process_tickers(tickers: str):
    if tickers is None:
        return None
    return tickers.split(' ')


def multiday_process(folder_path: str, temp_path: str, out_file: str, tickers: list[str] = None,
                     start_date: date = None, end_date: date = None,
                     remove_after_process: bool = True, verbose=True):
    # bucket_ofi_props = BucketOFIProps(levels=LEVELS, bucket_size=bucket_size, rounding=rounding,
    #                                   prev_bucket_size=bucket_size)

    archive_processor = SplitOFIArchiveProcessor(start_date=start_date, end_date=end_date, levels=LEVELS,
                                                 tickers=tickers, verbose=verbose)
    file_processor = SplitOFIToMultidayFileProcessor(verbose=verbose)

    archive_processor.process_archive(folder_path=folder_path, temp_path=temp_path, out_path=out_file,
                                      remove_after_process=remove_after_process, archive_output=False,
                                      extracted_archive_processor=file_processor)


def get_flt_and_bucket_ofi_props(tickers, start_date, end_date, bucket_size, rounding):
    if tickers is not None:
        tickers = tickers.split(' ')
    flt = FileFilter(start_date=date.fromisoformat(start_date) if start_date is not None else None,
                     end_date=date.fromisoformat(end_date) if end_date is not None else None,
                     tickers=tickers,
                     levels=LEVELS)
    bucket_ofi_props = BucketOFIProps(levels=LEVELS,
                                      bucket_size=bucket_size,
                                      rounding=rounding)
    return flt, bucket_ofi_props


@main.command()
def l3_to_split_ofi(folder_path: str, temp_path: str, out_path: str, bucket_size: int, tickers: str = None,
                    start_date: str = None, end_date: str = None, rounding: bool = True,
                    remove_after_process: bool = True, archive_output: bool = True, parallel_jobs: int = 1):
    tickers = process_tickers(tickers)
    start_date = process_date(start_date)
    end_date = process_date(end_date)
    bucket_ofi_props = BucketOFIProps(levels=LEVELS, bucket_size=bucket_size, rounding=rounding)

    archive_processor = L3ArchiveProcessor(start_date=start_date, end_date=end_date, levels=LEVELS, tickers=tickers)
    file_processor = L3ToSplitOFIFileProcessor(bucket_ofi_props=bucket_ofi_props,
                                               file_filter=archive_processor.file_filter, parallel_jobs=parallel_jobs)
    archive_processor.process_archive(folder_path=folder_path, temp_path=temp_path, out_path=out_path,
                                      remove_after_process=remove_after_process, archive_output=archive_output,
                                      extracted_archive_processor=file_processor, parallel_jobs=1)


@main.command()
def l3_to_prices(folder_path: str, temp_path: str, out_path: str, bucket_size: int, tickers: str = None,
                 start_date: str = None, end_date: str = None,
                 remove_after_process: bool = True, archive_output: bool = True, parallel_jobs: int = 1):
    tickers = process_tickers(tickers)
    start_date = process_date(start_date)
    end_date = process_date(end_date)
    bucket_ofi_props = BucketOFIProps(levels=LEVELS, bucket_size=bucket_size, rounding=False)

    archive_processor = L3ArchiveProcessor(start_date=start_date, end_date=end_date, levels=LEVELS, tickers=tickers)
    file_processor = L3ToPricesFileProcessor(bucket_ofi_props=bucket_ofi_props,
                                             file_filter=archive_processor.file_filter, parallel_jobs=parallel_jobs)
    archive_processor.process_archive(folder_path=folder_path, temp_path=temp_path, out_path=out_path,
                                      remove_after_process=remove_after_process, archive_output=archive_output,
                                      extracted_archive_processor=file_processor, parallel_jobs=1)


@main.command()
def split_ofi_to_split_ofi(folder_path: str, temp_path: str, out_path: str, bucket_size: int, prev_bucket_size: int,
                           tickers: str = None, start_date: str = None, end_date: str = None, rounding: bool = True,
                           rolling_size: int = None, remove_after_process: bool = True, archive_output: bool = True,
                           parallel_jobs: int = 1):
    tickers = process_tickers(tickers)
    start_date = process_date(start_date)
    end_date = process_date(end_date)
    bucket_ofi_props = BucketOFIProps(levels=LEVELS, bucket_size=bucket_size, rounding=rounding,
                                      prev_bucket_size=prev_bucket_size, rolling_size=rolling_size)

    archive_processor = SplitOFIArchiveProcessor(start_date=start_date, end_date=end_date, levels=LEVELS,
                                                 tickers=tickers)
    file_processor = SplitOFIToSplitOFIFileProcessor(bucket_ofi_props=bucket_ofi_props,
                                                     file_filter=archive_processor.file_filter,
                                                     parallel_jobs=parallel_jobs)

    archive_processor.process_archive(folder_path=folder_path, temp_path=temp_path, out_path=out_path,
                                      remove_after_process=remove_after_process, archive_output=archive_output,
                                      extracted_archive_processor=file_processor, parallel_jobs=1)


@main.command()
def multiday(folder_path: str, temp_path: str, out_file: str,
             tickers: str = None, start_date: str = None, end_date: str = None,
             remove_after_process: bool = True, verbose=True):
    tickers = process_tickers(tickers)
    start_date = process_date(start_date)
    end_date = process_date(end_date)

    multiday_process(folder_path=folder_path, temp_path=temp_path, out_file=out_file, tickers=tickers,
                     start_date=start_date, end_date=end_date, remove_after_process=remove_after_process,
                     verbose=verbose)


@main.command()
def iceberg_assert(folder_path: str, temp_path: str, tickers: str = None, start_date: str = None, end_date: str = None):
    tickers = process_tickers(tickers)
    start_date = process_date(start_date)
    end_date = process_date(end_date)
    archive_processor = L3ArchiveProcessor(start_date=start_date, end_date=end_date, levels=LEVELS, tickers=tickers)

    def assert_fn(row: pd.Series()) -> ():
        if row['event_type'] in [4, 5]:
            assert (abs(row['ofi_1']) > 1e-9)

    file_processor = DataAssertFileProcessor(file_filter=archive_processor.file_filter, assert_fn=assert_fn)
    archive_processor.process_archive(folder_path=folder_path, temp_path=temp_path, out_path=None,
                                      extracted_archive_processor=file_processor, remove_after_process=True,
                                      archive_output=False, parallel_jobs=1)


@main.command()
def extract_splitofi(folder_path: str, temp_path: str, out_path: str, tickers: str = None, start_date: str = None,
                     end_date: str = None):
    tickers = process_tickers(tickers)
    start_date = process_date(start_date)
    end_date = process_date(end_date)
    archive_processor = SplitOFIArchiveProcessor(start_date=start_date, end_date=end_date, levels=LEVELS,
                                                 tickers=tickers)
    file_processor = SplitOFIToExtractedProcessor(file_filter=archive_processor.file_filter)
    archive_processor.process_archive(folder_path=folder_path, temp_path=temp_path, out_path=out_path,
                                      extracted_archive_processor=file_processor, remove_after_process=True,
                                      archive_output=False, parallel_jobs=1)

@main.command()
def extract_prices(folder_path: str, temp_path: str, out_path: str, tickers: str = None, start_date: str = None,
                   end_date: str = None):
    tickers = process_tickers(tickers)
    start_date = process_date(start_date)
    end_date = process_date(end_date)
    archive_processor = PricesArchiveProcessor(start_date=start_date, end_date=end_date, levels=LEVELS,
                                               tickers=tickers)
    file_processor = None
    archive_processor.process_archive(folder_path=folder_path, temp_path=temp_path, out_path=out_path,
                                      extracted_archive_processor=file_processor, remove_after_process=True,
                                      archive_output=False, parallel_jobs=1)


if __name__ == '__main__':
    main()
