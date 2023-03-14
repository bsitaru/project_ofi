# Bucket OFI - aggregate OFI into buckets
import pandas as pd
import numpy as np

from datetime import date

from constants import OFI_TYPES, OFI_NAMES, OFI_COLS, VOLUME_COLS, ROUNDING, ROUNDING_RET, \
    START_TIME, END_TIME

from data_manipulation.message import get_message_df
from data_manipulation.orderbook import get_orderbook_df
from data_manipulation.tick_ofi import compute_tick_ofi_df

bucket_ofi_cols = ['start_time', 'event_count', 'start_price', 'end_price', 'return_now',
                   'return_future'] + OFI_COLS + VOLUME_COLS
sum_cols = OFI_COLS + VOLUME_COLS + ['event_count']

c_dtype = {c: float for c in bucket_ofi_cols}
c_dtype['start_time'] = int
c_dtype['event_count'] = int


class BucketOFIProps:
    def __init__(self, levels: int, bucket_size: int, rounding: bool, prev_bucket_size: int = None,
                 rolling_size: int = None, start_time: int = START_TIME, end_time: int = END_TIME):
        self.levels = levels
        self.bucket_size = bucket_size
        self.prev_bucket_size = prev_bucket_size
        self.rounding = rounding
        self.rolling_size = rolling_size if rolling_size is not None else bucket_size
        self.start_time = start_time
        self.end_time = end_time


def compute_return_now(start_price: pd.Series, end_price: pd.Series):
    return np.log(end_price / start_price)


def compute_start_price_in_bucket(start_price: pd.Series, end_price: np.ndarray) -> np.ndarray:
    # Set the start price in a bucket as the previous end price.
    ret = np.roll(end_price, 1)
    ret[:1] = start_price.iloc[0]
    return ret


def compute_return_future(return_now: pd.Series, roll: int = 1):
    ret = np.roll(return_now, -roll)
    ret[-roll:] = 0  # 0 for the last buckets
    return ret


def round_df(df: pd.DataFrame) -> ():
    for col in bucket_ofi_cols:
        if col in ['return_now', 'return_future']:
            df[col] = np.around(df[col], decimals=ROUNDING_RET)
        elif df.dtypes[col] == float:
            df[col] = np.around(df[col], decimals=ROUNDING)


def compute_bucket_ofi_df_from_tick_ofi(df: pd.DataFrame, props: BucketOFIProps) -> pd.DataFrame:
    # Compute start time of bucket
    df['start_time'] = (df['time'] // props.bucket_size).astype(int) * props.bucket_size

    # Divide OFI between types
    for i in range(1, props.levels + 1):
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
    df['return_future'] = compute_return_future(df['return_now'])

    # Compute normalized OFI
    # !!! There should always be at least an order on the market during the whole day.
    for i in range(1, props.levels + 1):
        for name in OFI_NAMES:
            df[f"{name}_{i}"] = np.divide(df[f"{name}_{i}"], df[f"volume_{i}"], out=np.ones_like(df[f"{name}_{i}"]),
                                          where=df[f"volume_{i}"] != 0)

    # Find missing buckets and set values for them
    all_indices = pd.Index(range(props.start_time, props.end_time, props.bucket_size)).astype(int)
    now_indices = df.index
    missing_indices = all_indices.difference(now_indices)
    prev_index = now_indices[np.searchsorted(now_indices, missing_indices) - 1]

    df = df.reindex(all_indices, fill_value=0, copy=False)
    df['start_time'] = df.index
    # Set start and end price for missing buckets as the previous end price
    for (idx, prv) in zip(missing_indices, prev_index):
        df.loc[idx, 'start_price'] = df.loc[prv, 'end_price']
        df.loc[idx, 'end_price'] = df.loc[prv, 'end_price']

    if props.rounding:
        round_df(df)

    return df[bucket_ofi_cols]


def compute_bucket_ofi_df_from_bucket_ofi(df: pd.DataFrame, props: BucketOFIProps) -> pd.DataFrame:
    if props.bucket_size % props.prev_bucket_size != 0:
        raise ValueError(
            f"New bucket size is not a multiple of previous bucket size! {props.prev_bucket_size} ; {props.bucket_size}")

    # Recompute sum of OFIs and sum of volumes
    for i in range(1, props.levels + 1):
        for name in OFI_NAMES:
            df[f"{name}_{i}"] *= df[f"volume_{i}"]
        df[f"volume_{i}"] *= df['event_count']

    # Compute new start time
    # df['start_time'] = (df['start_time'] // props.bucket_size).astype(int) * props.bucket_size

    # group_df = df.groupby('start_time')
    # df = group_df[sum_cols].agg('sum')
    window_size = props.bucket_size // props.prev_bucket_size
    rolling_df = df.rolling(window_size)

    first_element = lambda s: s.iloc[0]
    last_element = lambda s: s.iloc[-1]

    df = rolling_df.agg({'start_time': first_element,
                         'start_price': first_element,
                         'end_price': last_element,
                         **{c: 'sum' for c in sum_cols}})
    df['return_now'] = compute_return_now(df['start_price'], df['end_price'])
    df['return_future'] = compute_return_future(df['return_now'], roll=window_size)

    pd.DataFrame.dropna(df, inplace=True)

    df['start_time'] = df['start_time'].astype(int)
    df['event_count'] = df['event_count'].astype(int)

    df.drop(df[df['start_time'] % props.rolling_size != 0].index, inplace=True)

    for i in range(1, props.levels + 1):
        volume_col = f"volume_{i}"
        df[volume_col] = np.divide(df[volume_col], df["event_count"], out=np.zeros_like(df[volume_col]),
                                   where=df["event_count"] != 0)
        for name in OFI_NAMES:
            ofi_col = f"{name}_{i}"
            df[ofi_col] = np.divide(df[ofi_col], df[volume_col], out=np.zeros_like(df[ofi_col]),
                                    where=df[volume_col] != 0)

    if props.rounding:
        round_df(df)

    return df[bucket_ofi_cols]


FILE_TYPE = 'SplitOFIBucket'


def get_new_bucket_ofi_file_name(ticker: str, d: date, props: BucketOFIProps):
    rolling = '' if props.rolling_size == props.bucket_size else f'roll{props.rolling_size}'
    name = '_'.join([ticker, d, f'{FILE_TYPE}{props.bucket_size}{rolling}', str(props.levels)])
    return name + '.csv'


def compute_bucket_ofi_from_files(message_file: str, orderbook_file: str, props: BucketOFIProps) -> pd.DataFrame:
    message_df = get_message_df(message_file)
    orderbook_df = get_orderbook_df(orderbook_file, props.levels)
    tick_ofi_df = compute_tick_ofi_df(message_df, orderbook_df, props.levels)
    bucket_ofi_df = compute_bucket_ofi_df_from_tick_ofi(tick_ofi_df, props)
    return bucket_ofi_df


def get_bucket_ofi_df(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, dtype=c_dtype)
        return df
    except:
        return pd.DataFrame(columns=list(c_dtype.keys()))
