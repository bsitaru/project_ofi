import os
from datetime import date, timedelta

import constants
import data_loader.dates as dates_loader
from data_loader.day_prices import DayPrices
from data_manipulation.prices import get_prices_df

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

if __name__ == '__main__':
    path = './lobster_sample/prices_extracted'
    all_dates = set()
    for ticker in constants.TICKERS[:100]:
        dates = dates_loader.get_dates_from_folder(path, tickers=[ticker])
        # dates = [date.fromisoformat('2017-07-03')]
        for d in dates:
            dayprices = DayPrices(ticker, d)

            rows = [dayprices.get_price_row(time) for time in [57599, 57500, 57400, 57300]]
            has_early_closing = False
            for row in rows:
                if row[2] < 0.0001 or row[1] > 499999.0:
                    has_early_closing = True
                    break

            if has_early_closing:
                print(f"early closing --- {d} --- {ticker}")
                all_dates.add(d)

    print(f"all --- {all_dates}")

