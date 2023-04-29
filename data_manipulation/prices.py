from datetime import date

import numpy as np
import pandas as pd

from data_manipulation.bucket_ofi import BucketOFIProps
from data_manipulation.orderbook import get_orderbook_df
from data_manipulation.message import get_message_df

c_dtype = {}
c_dtype['time'] = int
c_dtype['ask_price'] = int
c_dtype['bid_price'] = int


def empty_df():
    return pd.DataFrame(columns=list(c_dtype.keys()))


def compute_prices_df(orderbook_df: pd.DataFrame, message_df: pd.DataFrame, props):
    df = orderbook_df
    df['time'] = message_df['time']
    df.drop(columns=df.columns[~df.columns.isin(['time', 'ask_price_1', 'bid_price_1'])], inplace=True)
    df.drop(df[(df['time'] <= props.start_time) | (df['time'] > props.end_time)].index, inplace=True)

    if df.empty:
        return pd.DataFrame()

    # Compute end time of bucket
    df['time'] = ((df['time'] + props.bucket_size - 1) // props.bucket_size).astype(int) * props.bucket_size
    group_df = df.groupby('time')

    last_element = lambda s: s.iloc[-1]

    df = group_df.agg({c: last_element for c in ['time', 'ask_price_1', 'bid_price_1']})
    df.rename(columns={'ask_price_1': 'ask_price', 'bid_price_1': 'bid_price'}, inplace=True)
    df['ask_price'] /= 10000.0
    df['bid_price'] /= 10000.0
    return df


def compute_prices_df_from_files(message_file: str, orderbook_file: str, props: BucketOFIProps) -> pd.DataFrame:
    message_df = get_message_df(message_file)
    orderbook_df = get_orderbook_df(orderbook_file, props.levels)
    df = compute_prices_df(message_df=message_df, orderbook_df=orderbook_df, props=props)
    return df


def get_new_file_name(ticker: str, d: date, props: BucketOFIProps):
    FILE_TYPE = 'Prices'
    name = '_'.join([ticker, d, f'{FILE_TYPE}{props.bucket_size}', str(props.levels)])
    return name + '.csv'


def get_prices_df(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, dtype=c_dtype)
        return df
    except:
        return empty_df()
