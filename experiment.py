import typer
import yaml

import constants
import experiments.price_impact as pi_intraday
from data_process import process_tickers, process_date
from ml_collections import ConfigDict

main = typer.Typer()


# in_sample_size: int,
#              model_name: str, os_size: int = None, rolling: int = None, tickers: str = None,
#              start_date: str = None, end_date: str = None,

def load_yaml(config_file: str):
    with open(config_file, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if cfg['start_date'] is not None:
        cfg['start_date'] = cfg['start_date']
    if cfg['end_date'] is not None:
        cfg['end_date'] = cfg['end_date']
    if cfg['tickers'] is None:
        cfg['tickers'] = constants.DEFAULT_TICKERS
    return cfg


@main.command()
def main_cmd(config_file: str, levels: int = None, parallel_jobs: int = None, use_all: bool = None):
    args = load_yaml(config_file)
    if levels is not None:
        args['selector']['levels'] = levels
    if parallel_jobs is not None:
        args['parallel_jobs'] = parallel_jobs
    if use_all:
        args['start_date'] = None
        args['end_date'] = None
        args['tickers'] = constants.DEFAULT_TICKERS
    args = ConfigDict(args)

    # os_size = in_sample_size if os_size is None else os_size
    # rolling = in_sample_size if rolling is None else rolling
    #
    # tickers = process_tickers(tickers)
    # start_date = process_date(start_date)
    # end_date = process_date(end_date)

    if args.experiment.name == 'individual_price_impact':
        pi_intraday.run_experiment_individual(args)
    elif args.experiment.name == 'universal_price_impact':
        pi_intraday.run_experiment_universal(args)
    else:
        raise ValueError(f'invalid experiment name {args.experiment.name}')


if __name__ == '__main__':
    main()
