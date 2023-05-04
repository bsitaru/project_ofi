import os
from datetime import date

import constants
import data_manipulation.prediction as pred
import typer
import strategy.sign as stsign
import strategy.portfolio as stportfolio
from strategy.strategy_results import StrategyResults

from logging_utils import get_logger, log

main = typer.Typer()

quarters = [('2017-01-01', '2017-03-31'), ('2017-04-01', '2017-06-30'), ('2017-07-01', '2017-09-30'),
            ('2017-10-01', '2017-12-31'),
            ('2018-01-01', '2018-03-31'), ('2018-04-01', '2018-06-30'), ('2018-07-01', '2018-09-30'),
            ('2018-10-01', '2018-12-31'),
            ('2019-01-01', '2019-03-31'), ('2019-04-01', '2019-06-30'), ('2019-07-01', '2019-09-30'),
            ('2019-10-01', '2019-12-31')]


def print_pnl(stats_mean, stats_prod, logger):
    log('Mean PnL: ', logger=logger)
    stats_mean.print(logger)
    log('Product PnL: ', logger=logger)
    stats_prod.print(logger)


@main.command()
def sign(folder_path: str):
    logger = get_logger(folder_path, 'pred_signs')
    files_list = os.listdir(folder_path)
    all_correct, all_tot = 0, 0
    for file_name in files_list:
        if 'predict' not in file_name:
            continue
        df = pred.get_prediction_df(os.path.join(folder_path, file_name))
        ratio, correct, tot_num = stsign.get_heat_ratio(df)
        text = f"{file_name} --- ratio: {ratio} --- correct: {correct} --- total: {tot_num}"
        print(text)
        logger.info(text)
        all_correct += correct
        all_tot += tot_num

    ratio = all_correct / all_tot
    text = f"TOTAL --- ratio: {ratio} --- correct: {all_correct} --- total: {all_tot}"
    print(text)
    logger.info(text)


@main.command()
def portfolio(folder_path: str, start_date=None, end_date=None):
    if start_date is not None:
        start_date = date.fromisoformat(start_date)
    if end_date is not None:
        end_date = date.fromisoformat(end_date)
    logger = get_logger(folder_path, 'portfolio')
    df = pred.get_all_predictions(folder_path)
    logger.info("Predictions loaded...")
    stats_mean, stats_prod = stportfolio.make_strategy_portfolio(df, logger, start_date=start_date, end_date=end_date)

    print_pnl(stats_mean, stats_prod, logger)


@main.command()
def portfolio_quarters(folder_path: str):
    logger = get_logger(folder_path, 'portfolio')
    df = pred.get_all_predictions(folder_path)
    logger.info("Predictions loaded...")

    mn, prd = [], []
    for (start_date, end_date) in quarters:
        log(f"quarter {start_date} {end_date}", logger=logger)
        stats_mean, stats_prod = stportfolio.make_strategy_portfolio(df, logger,
                                                                     start_date=date.fromisoformat(start_date),
                                                                     end_date=date.fromisoformat(end_date))

        log(f"quarter {start_date} {end_date} results", logger=logger)
        print_pnl(stats_mean, stats_prod, logger)

        mn.append(stats_mean)
        prd.append(stats_prod)

    log(f"overall results", logger=logger)
    stats_mean = StrategyResults.from_strategy_results_list(mn)
    stats_prod = StrategyResults.from_strategy_results_list(prd)
    print_pnl(stats_mean, stats_prod, logger)


if __name__ == '__main__':
    main()
