import pandas as pd
import numpy as np

import data_manipulation.prices as prices
from logging_utils import log

HORIZONT = 60
DAYS_PER_YEAR = 241
def make_strategy_portfolio(df: pd.DataFrame, logger=None):
    dates = np.unique(df['date'])

    def solve_date(d: str):
        date_df = df[df['date'] == d]
        times = np.unique(date_df['time_now'])
        tickers = np.unique(date_df['ticker'])

        prices_dfs = {}
        for t in tickers:
            pr_df = prices.get_prices_df_for_ticker_date(ticker=t, d=d)
            pr_df = prices.compute_additional(pr_df)
            prices_dfs[t] = pr_df

        money = 1.0
        for now in times:
            df_now = date_df[date_df['time_now'] == now]
            tickers = np.unique(df_now['ticker'])
            weights = {}
            tot_weights = 0
            for t in tickers:
                row = df_now.loc[df_now['ticker'] == t]
                y_pred = row['y_pred'].iloc[0]
                y_train_std = row['y_train_std'].iloc[0]
                ret = np.exp(y_pred) - 1.0

                pr_df = prices_dfs[t]
                price_row = pr_df.loc[pr_df['time'] <= now].iloc[-1]
                # print(f"time {now}, date {d}, ticker {t} --- {price_row}")
                rel_spread = price_row['rel_spread']
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
                price_now = pr_df.loc[pr_df['time'] <= now].iloc[-1]
                price_next = pr_df.loc[pr_df['time'] <= (now + HORIZONT)].iloc[-1]

                ret = 0.0
                if w < 0:
                    ret = price_next['ask_price'] / price_now['bid_price']
                else:
                    ret = price_next['bid_price'] / price_now['ask_price']

                rets.append(np.abs(w) * ret)
            tot_ret = sum(rets)
            money *= tot_ret

        money -= 1.0
        return money

    all_profits = []
    for d in dates:
        money = solve_date(d)
        log(f'date {d} pnl --- {money}', logger=logger)
        all_profits.append(money)

    all_profits = np.array(all_profits)
    avg_profit = np.mean(all_profits)
    sum_profit = np.sum(all_profits)
    anual = avg_profit * DAYS_PER_YEAR
    log(f"average profit --- {avg_profit}", logger=logger)
    log(f"sum profit --- {sum_profit}", logger=logger)
    log(f"annualized profit --- {anual}", logger=logger)

