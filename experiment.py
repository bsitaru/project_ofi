import typer

import experiments.price_impact as pi_intraday
from data_process import process_tickers, process_date

main = typer.Typer()


@main.command()
def price_impact_intraday(folder_path: str, temp_path: str, results_path: str, in_sample_size: int, model_name: str,
                          os_size: int = None,
                          rolling: int = None, tickers: str = None, start_date: str = None, end_date: str = None,
                          parallel_jobs: int = 1):
    os_size = in_sample_size if os_size is None else os_size
    rolling = in_sample_size if rolling is None else rolling

    tickers = process_tickers(tickers)
    start_date = process_date(start_date)
    end_date = process_date(end_date)

    pi_intraday.run_experiment(folder_path=folder_path, temp_path=temp_path, results_path=results_path,
                               model_name=model_name,
                               in_sample_size=in_sample_size, os_size=os_size, rolling=rolling, start_date=start_date,
                               end_date=end_date, tickers=tickers, parallel_jobs=parallel_jobs)


if __name__ == '__main__':
    main()
