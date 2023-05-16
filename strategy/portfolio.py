from datetime import date

import pandas as pd
import numpy as np

import constants
import data_manipulation.prices as prices
from logging_utils import log
from data_loader.day_prices import DayPrices
from strategy.strategy_results import StrategyResults, DayResults

HORIZONT = 60





def make_strategy_portfolio(df: pd.DataFrame, logger=None, start_date=None, end_date=None, tickers=None):
    date_dfs = df.groupby('date')
    dates = np.array(list(date_dfs.groups.keys()))
    if start_date is not None:
        dates = dates[start_date <= dates]
    if end_date is not None:
        dates = dates[dates <= end_date]

    def solve_date(d: date, date_df):
        # date_df = df[df['date'] == d]
        times = np.unique(date_df['time_now'])
        day_tickers = np.unique(date_df['ticker'])

        prices_dfs = {}
        for t in day_tickers:
            if tickers is not None and t not in tickers:
                continue
            prices_dfs[t] = DayPrices(t, d)

        all_rets = []
        for now in times:
            if len(all_rets) == 151:
                money = 1.1
            df_now = date_df[date_df['time_now'] == now].values.tolist()
            weights = {}
            tot_weights = 0
            for row in df_now:
                # row = df_now.loc[df_now['ticker'] == t]
                [_, _, t, y_true, y_pred, y_train_std] = row
                if tickers is not None and t not in tickers:
                    continue
                # y_pred = row['y_pred'].iloc[0]
                # y_train_std = row['y_train_std'].iloc[0]
                ret = np.exp(y_pred) - 1.0

                pr_df = prices_dfs[t]
                price_row = pr_df.get_price_row(now)
                # print(f"time {now}, date {d}, ticker {t} --- {price_row}")
                rel_spread = price_row[4]
                if np.abs(ret) <= rel_spread or y_train_std == 0:
                    continue

                w = ret / y_train_std
                tot_weights += np.abs(w)
                weights[t] = w

            if len(weights) == 0:
                continue

            rets = []
            for (t, w) in weights.items():
                w /= tot_weights

                pr_df = prices_dfs[t]
                price_now = pr_df.get_price_row(now)
                price_next = pr_df.get_price_row(now + HORIZONT)

                ret = 0.0
                if w < 0:  # sell now, buy next
                    ret = price_now[3] / price_next[3]
                    # ret = price_now[2] / price_next[1]
                else:  # buy now, sell next
                    ret = price_next[3] / price_now[3]
                    # ret = price_next[2] / price_now[1]

                rets.append(np.abs(w) * ret)
            tot_ret = sum(rets)
            all_rets.append(tot_ret)

        all_rets = np.array(all_rets) - 1.0
        day_results = DayResults(all_rets)
        return day_results

    all_res = []
    for d in dates:
        if d in constants.EARLY_CLOSING_DAYS:
            continue
        date_df = date_dfs.get_group(d)
        res = solve_date(d, date_df)
        log(f'date {d} pnl --- {res.pnl} --- ppd --- {res.ppd}')
        all_res.append(res)

    stats_mean = StrategyResults(all_res)
    return stats_mean
