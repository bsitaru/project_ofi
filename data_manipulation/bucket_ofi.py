import pandas as pd
import numpy as np

from constants import *

from data_manipulation.message import get_message_df
from data_manipulation.orderbook import get_orderbook_df
from data_manipulation.tick_ofi import compute_tick_ofi_df


def compute_bucket_ofi_df(df: pd.DataFrame, bucket_size: int = BUCKET_SIZE) -> pd.DataFrame:
    # Compute start time of bucket
    df['start_time'] = (df['time'] // bucket_size).astype(int) * bucket_size

    # Divide OFI between types
    for i in range(1, LEVELS + 1):
        for t in OFI_TYPES:
            df[f"ofi_{t}_{i}"] = np.where(df["ofi_type"] == t, df[f"ofi_{i}"], 0)

    group_df = df.groupby('start_time')
    df = group_df[OFI_COLS].agg('sum')
    df[VOLUME_COLS] = group_df[VOLUME_COLS].agg('mean')
    df['start_price'] = group_df.apply(lambda df: df['price'].tolist()[0])
    df['end_price'] = group_df.apply(lambda df: df['price'].tolist()[-1])

    # Compute normalized OFI
    # !!! There should always be at least an order on the market during the whole day.
    for i in range(1, LEVELS + 1):
        for name in OFI_NAMES:
            df[f"{name}_{i}"] /= df[f"volume_{i}"]

    # Find missing buckets and set values for them
    all_indices = pd.Index(range(START_TRADE, END_TRADE, bucket_size)).astype(int)
    now_indices = df.index
    missing_indices = all_indices.difference(now_indices)
    prev_index = now_indices[np.searchsorted(now_indices, missing_indices) - 1]

    df = df.reindex(all_indices, fill_value=0, copy=False)
    df['start_time'] = df.index
    for (idx, prv) in zip(missing_indices, prev_index):
        df.loc[idx, 'start_price'] = df.loc[prv, 'end_price']
        df.loc[idx, 'end_price'] = df.loc[prv, 'end_price']

    cols = ['start_time', 'start_price', 'end_price'] + OFI_COLS
    return df[cols]


def generate_bucket_ofi_file(message_file: str, orderbook_file: str, bucket_ofi_file: str) -> None:
    message_df = get_message_df(message_file)
    orderbook_df = get_orderbook_df(orderbook_file)
    tick_ofi_df = compute_tick_ofi_df(message_df, orderbook_df)
    bucket_ofi_df = compute_bucket_ofi_df(tick_ofi_df)
    bucket_ofi_df.to_csv(bucket_ofi_file, index=False)
