# Bucket OFI - aggregate OFI into buckets
import pandas as pd
import numpy as np

from datetime import date

from constants import OFI_TYPES, OFI_NAMES, OFI_COLS, VOLUME_COLS, START_TRADE, END_TRADE

from data_manipulation.message import get_message_df
from data_manipulation.orderbook import get_orderbook_df
from data_manipulation.tick_ofi import compute_tick_ofi_df

bucket_ofi_cols = ['start_time', 'event_count', 'start_price', 'end_price', 'return_now'] + OFI_COLS + VOLUME_COLS


def compute_return_now(start_price: pd.Series, end_price: pd.Series):
    return np.log(end_price / start_price)


def compute_start_price_in_bucket(start_price: pd.Series, end_price: np.ndarray) -> np.ndarray:
    # Set the start price in a bucket as the previous end price.
    ret = np.roll(end_price, 1)
    ret[:1] = start_price.iloc[0]
    return ret


def compute_bucket_ofi_df_from_tick_ofi(df: pd.DataFrame, levels: int, bucket_size: int) -> pd.DataFrame:
    # Compute start time of bucket
    df['start_time'] = (df['time'] // bucket_size).astype(int) * bucket_size

    # Divide OFI between types
    for i in range(1, levels + 1):
        for t in OFI_TYPES:
            df[f"ofi_{t}_{i}"] = np.where(df["ofi_type"] == t, df[f"ofi_{i}"], 0)

    group_df = df.groupby('start_time')
    df = group_df[OFI_COLS].agg('sum')
    df[VOLUME_COLS] = group_df[VOLUME_COLS].agg('mean')
    df['start_price'] = group_df.apply(lambda df: df['price'].tolist()[0])
    df['end_price'] = group_df.apply(lambda df: df['price'].tolist()[-1])
    df['start_price'] = compute_start_price_in_bucket(df['start_price'], df['end_price'])
    df['event_count'] = group_df.agg('size')
    df['return_now'] = compute_return_now(df['start_price'], df['end_price'])

    # Compute normalized OFI
    # !!! There should always be at least an order on the market during the whole day.
    for i in range(1, levels + 1):
        for name in OFI_NAMES:
            df[f"{name}_{i}"] /= df[f"volume_{i}"]

    # Find missing buckets and set values for them
    all_indices = pd.Index(range(START_TRADE, END_TRADE, bucket_size)).astype(int)
    now_indices = df.index
    missing_indices = all_indices.difference(now_indices)
    prev_index = now_indices[np.searchsorted(now_indices, missing_indices) - 1]

    df = df.reindex(all_indices, fill_value=0, copy=False)
    df['start_time'] = df.index
    # Set start and end price for missing buckets as the previous end price
    for (idx, prv) in zip(missing_indices, prev_index):
        df.loc[idx, 'start_price'] = df.loc[prv, 'end_price']
        df.loc[idx, 'end_price'] = df.loc[prv, 'end_price']

    return df[bucket_ofi_cols]


def compute_bucket_ofi_df_from_bucket_ofi(df: pd.DataFrame, levels: int, prev_bucket_size: int,
                                          bucket_size: int, **kwargs) -> pd.DataFrame:
    if bucket_size % prev_bucket_size != 0:
        raise ValueError(
            f"New bucket size is not a multiple of previous bucket size! {prev_bucket_size} ; {bucket_size}")

    # Compute new start time
    df['start_time'] = (df['start_time'] // bucket_size).astype(int) * bucket_size

    # Recompute sum of OFIs and sum of volumes
    for i in range(1, levels + 1):
        for name in OFI_NAMES:
            df[f"{name}_{i}"] *= df[f"volume_{i}"]
        df[f"volume_{i}"] *= df['event_count']

    sum_cols = OFI_COLS + VOLUME_COLS + ['event_count']
    group_df = df.groupby('start_time')
    df = group_df[sum_cols].agg('sum')
    df['start_time'] = df.index
    df['start_price'] = group_df.apply(lambda df: df['start_price'].tolist()[0])
    df['end_price'] = group_df.apply(lambda df: df['end_price'].tolist()[-1])
    df['start_price'] = compute_start_price_in_bucket(df['start_price'], df['end_price'])
    df['return_now'] = compute_return_now(df['start_price'], df['end_price'])

    for i in range(1, levels + 1):
        volume_col = f"volume_{i}"
        df[volume_col] = np.divide(df[volume_col], df["event_count"], out=np.zeros_like(df[volume_col]),
                                   where=df["event_count"] != 0)
        for name in OFI_NAMES:
            ofi_col = f"{name}_{i}"
            df[ofi_col] = np.divide(df[ofi_col], df[volume_col], out=np.zeros_like(df[ofi_col]),
                                    where=df[volume_col] != 0)

    return df[bucket_ofi_cols]


FILE_TYPE = 'SplitOFIBucket'


def get_new_bucket_ofi_file_name(ticker: str, d: date, bucket_size: int, levels: int, **kwargs):
    name = '_'.join([ticker, d, f'{FILE_TYPE}{bucket_size}', str(levels)])
    return name + '.csv'


def get_bucket_size_from_file_name(file_name: str) -> int:
    file_type = file_name.split('_')[2]
    if not file_type.startswith(FILE_TYPE):
        return None
    return int(file_type[len(FILE_TYPE):])


def compute_bucket_ofi_from_files(message_file: str, orderbook_file: str, levels: int, bucket_size: int,
                                  **kwargs) -> pd.DataFrame:
    message_df = get_message_df(message_file)
    orderbook_df = get_orderbook_df(orderbook_file, levels)
    tick_ofi_df = compute_tick_ofi_df(message_df, orderbook_df, levels)
    bucket_ofi_df = compute_bucket_ofi_df_from_tick_ofi(tick_ofi_df, levels, bucket_size)
    return bucket_ofi_df


def get_bucket_ofi_df(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False)
    return df
