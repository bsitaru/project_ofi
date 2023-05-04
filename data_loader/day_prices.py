import numpy as np
import data_manipulation.prices as prices


class DayPrices:
    def __init__(self, t, d):
        pr_df = prices.get_prices_df_for_ticker_date(ticker=t, d=d)
        pr_df = prices.compute_additional(pr_df)
        self.index = pr_df['time'].to_numpy()
        self.data = pr_df.to_numpy()

    def get_price_row(self, time):
        idx = np.searchsorted(self.index, time, side='right')
        idx -= 1
        return self.data[idx, :]
