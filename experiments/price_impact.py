import os
import logging
import yaml
import experiments.runner as runner
from models.regression_results import AveragedRegressionResults
from experiments.naming import args_to_name
from logging_utils import get_logger, log, log_tickers

from joblib import Parallel, delayed


def experiment_init(args):
    print(f'Experiment: {args.experiment.name}')
    results_name = args_to_name(args)
    results_path = os.path.join(args.results_path, results_name)
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    logger = get_logger(results_path, 'logs')
    logger.info(f'Experiment: {args.experiment.name}')

    with open(os.path.join(results_path, 'config.yaml'), 'w') as outfile:
        yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    return logger, results_path


def run_experiment_individual(args):
    logger, results_path = experiment_init(args)

    parallel_jobs = args.parallel_jobs
    args.parallel_jobs = 1
    def run_experiment_for_ticker(ticker: str):
        logger_now = get_logger(results_path, ticker)
        logger_now.info('Starting...')
        logger = get_logger(results_path, 'logs')

        results = runner.experiment(tickers=[ticker], args=args, logger_name=ticker)

        if results.average.size > 0:
            results_text = f'{ticker} --- INS : {results.average[0]} --- OOS : {results.average[1]}'
            log(results_text, logger=logger)

            results.save_pickle(os.path.join(results_path, f'{ticker}.pickle'))

    Parallel(n_jobs=parallel_jobs)(delayed(run_experiment_for_ticker)(t) for t in args.tickers)

    results = AveragedRegressionResults.from_directory(results_path)
    results.log(logger)


def run_experiment_universal(args):
    logger, results_path = experiment_init(args)

    results = runner.experiment(args, tickers=args.tickers, logger_name='logs')
    results_text = f'{log_tickers(args.tickers)} --- INS : {results.average[0]} --- OOS : {results.average[1]}'
    log(results_text, logger=logger)
    results.save_pickle(os.path.join(results_path, 'universal.pickle'))

def run_experiment_clustered(args):
    logger, results_path = experiment_init(args)

    results = runner.experiment(args, tickers=args.tickers, logger_name='logs')
    results_text = f'{log_tickers(args.tickers)} --- INS : {results.average[0]} --- OOS : {results.average[1]}'
    log(results_text, logger=logger)
    results.save_pickle(os.path.join(results_path, 'clustered.pickle'))
