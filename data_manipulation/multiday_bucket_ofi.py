import pandas as pd

from datetime import date
from constants import START_TRADE, END_TRADE, VOLATILE_TIMEFRAME


def prepare_df_for_multiday(df: pd.DataFrame, file_name: str, bucket_size: int, **kwargs) -> pd.DataFrame:
    # Remove volatile timeframe
    left = START_TRADE + VOLATILE_TIMEFRAME
    right = END_TRADE - VOLATILE_TIMEFRAME - bucket_size + 1
    df.drop(df[(left > df['start_time']) | (df['start_time'] > right)].index, inplace=True)

    # Add ticker and date
    [ticker, d, _, _] = file_name[:-4].split('_')
    df['ticker'] = ticker
    df['date'] = str(d)

    return df
