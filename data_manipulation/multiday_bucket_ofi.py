import pandas as pd

from datetime import date
from constants import START_TRADE, END_TRADE, VOLATILE_TIMEFRAME
import data_manipulation.bucket_ofi as bucket_ofi

c_dtype = bucket_ofi.c_dtype
c_dtype['date'] = str
c_dtype['ticker'] = str


class MultidayProps:
    def __init__(self, bucket_size: int):
        self.bucket_size = bucket_size


def prepare_df_for_multiday(df: pd.DataFrame, file_name: str, props: MultidayProps) -> pd.DataFrame:
    # Remove volatile timeframe
    left = START_TRADE + VOLATILE_TIMEFRAME
    right = END_TRADE - VOLATILE_TIMEFRAME - props.bucket_size + 1
    df.drop(df[(left > df['start_time']) | (df['start_time'] > right)].index, inplace=True)

    # Add ticker and date
    [ticker, d, _, _] = file_name[:-4].split('_')
    df['ticker'] = ticker
    df['date'] = str(d)

    return df


def get_multiday_df(file_path: str):
    df = pd.read_csv(file_path, dtype=c_dtype)
    return df
