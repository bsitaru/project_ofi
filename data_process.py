import typer

from datetime import date
from data_manipulation.archive_process import L3ArchiveProcessor, L3ToSplitOFIFileProcessor, FileFilter, BucketOFIProps, \
    SplitOFIArchiveProcessor, SplitOFIToSplitOFIFileProcessor, SplitOFIToMultidayFileProcessor
from constants import TICKERS, LEVELS

main = typer.Typer()


def process_date(d: str) -> date:
    return date.fromisoformat(d) if d is not None else None


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
    start_date = process_date(start_date)
    end_date = process_date(end_date)
    bucket_ofi_props = BucketOFIProps(levels=LEVELS, bucket_size=bucket_size, rounding=rounding)

    archive_processor = L3ArchiveProcessor(start_date=start_date, end_date=end_date, levels=LEVELS, tickers=tickers)
    file_processor = L3ToSplitOFIFileProcessor(bucket_ofi_props=bucket_ofi_props,
                                               file_filter=archive_processor.file_filter, parallel_jobs=parallel_jobs)
    archive_processor.process_archive(folder_path=folder_path, temp_path=temp_path, out_path=out_path,
                                      remove_after_process=remove_after_process, archive_output=archive_output,
                                      extracted_archive_processor=file_processor)


@main.command()
def split_ofi_to_split_ofi(folder_path: str, temp_path: str, out_path: str, bucket_size: int, prev_bucket_size: int,
                           tickers: str = None, start_date: str = None, end_date: str = None, rounding: bool = True,
                           rolling_size: int = None, remove_after_process: bool = True, archive_output: bool = True,
                           parallel_jobs: int = 1):
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
                                      extracted_archive_processor=file_processor)


@main.command()
def multiday(folder_path: str, temp_path: str, out_file: str, bucket_size: int,
             tickers: str = None, start_date: str = None, end_date: str = None, rounding: bool = True,
             remove_after_process: bool = True):
    start_date = process_date(start_date)
    end_date = process_date(end_date)
    bucket_ofi_props = BucketOFIProps(levels=LEVELS, bucket_size=bucket_size, rounding=rounding,
                                      prev_bucket_size=bucket_size)

    archive_processor = SplitOFIArchiveProcessor(start_date=start_date, end_date=end_date, levels=LEVELS,
                                                 tickers=tickers)
    file_processor = SplitOFIToMultidayFileProcessor(bucket_ofi_props=bucket_ofi_props)

    archive_processor.process_archive(folder_path=folder_path, temp_path=temp_path, out_path=out_file,
                                      remove_after_process=remove_after_process, archive_output=False,
                                      extracted_archive_processor=file_processor)


if __name__ == '__main__':
    main()
