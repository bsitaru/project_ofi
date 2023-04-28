import typer
import yaml

import constants
import experiments.price_impact as pi_intraday
from data_process import process_tickers, process_date
from ml_collections import ConfigDict

main = typer.Typer()


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
def main_cmd(config_file: str, levels: int = None, parallel_jobs: int = None, use_all: bool = None, folder_path: str = None, results_path: str = None):
    args = load_yaml(config_file)
    if levels is not None:
        args['selector']['levels'] = levels
    if parallel_jobs is not None:
        args['parallel_jobs'] = parallel_jobs
    if use_all:
        args['start_date'] = None
        args['end_date'] = None
        args['tickers'] = constants.DEFAULT_TICKERS
    if folder_path is not None:
        args['folder_path'] = folder_path
    if results_path is not None:
        args['results_path'] = results_path
    args = ConfigDict(args)

    if args.experiment.name in ['individual_price_impact', 'individual_future']:
        pi_intraday.run_experiment_individual(args)
    elif args.experiment.name == 'universal_price_impact':
        pi_intraday.run_experiment_universal(args)
    elif args.experiment.name in ['clustered_price_impact', 'neigh_price_impact']:
        pi_intraday.run_experiment_clustered(args)
    else:
        raise ValueError(f'invalid experiment name {args.experiment.name}')


if __name__ == '__main__':
    main()
