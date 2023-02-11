import typer

from datetime import date
from data_manipulation.archive_process import process_archive_folder, process_split_ofi_archive_folder, FileFilter, \
    BucketOFIProps

from constants import TICKERS, LEVELS

main = typer.Typer()


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
def archive(folder_path: str, temp_path: str, out_path: str, bucket_size: int, tickers: str = None,
            start_date: str = None, end_date: str = None, rounding: bool = True,
            remove_after_process: bool = True, create_archive: bool = True, parallel_jobs: int = 1):
    flt, bucket_ofi_props = get_flt_and_bucket_ofi_props(tickers, start_date, end_date, bucket_size, rounding)
    process_archive_folder(folder_path=folder_path,
                           temp_path=temp_path,
                           out_path=out_path,
                           flt=flt,
                           bucket_ofi_props=bucket_ofi_props,
                           remove_after_process=remove_after_process,
                           create_archive=create_archive,
                           parallel_jobs=parallel_jobs)


@main.command()
def multiday(folder_path: str, temp_path: str, out_file: str, bucket_size: int, tickers: str = None,
             start_date: str = None, end_date: str = None, rounding: bool = True, remove_after_process: bool = True):
    flt, bucket_ofi_props = get_flt_and_bucket_ofi_props(tickers, start_date, end_date, bucket_size, rounding)
    process_split_ofi_archive_folder(folder_path=folder_path,
                                     out_file=out_file,
                                     temp_path=temp_path,
                                     flt=flt,
                                     bucket_ofi_props=bucket_ofi_props,
                                     remove_after_process=remove_after_process)


if __name__ == '__main__':
    main()
