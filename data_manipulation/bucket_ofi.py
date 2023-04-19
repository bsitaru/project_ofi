# Bucket OFI - aggregate OFI into buckets
# Computes split ofi (not normalised) and average volume for each level in bucket
import pandas as pd
import numpy as np

from datetime import date

import constants
from constants import OFI_TYPES, SPLIT_OFI_NAMES, SPLIT_OFI_COLS, VOLUME_COLS, ROUNDING, ROUNDING_RET, \
    START_TIME, END_TIME

from data_manipulation.message import get_message_df
from data_manipulation.orderbook import get_orderbook_df
from data_manipulation.tick_ofi import compute_tick_ofi_df

bucket_ofi_cols = ['start_time', 'event_count', 'return', 'start_price', 'end_price'] + SPLIT_OFI_COLS + VOLUME_COLS
sum_cols = SPLIT_OFI_COLS + VOLUME_COLS + ['event_count']

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


def empty_df():
    return pd.DataFrame(columns=list(c_dtype.keys()))


def compute_return(start_price: pd.Series, end_price: pd.Series):
    return np.log(end_price / start_price, out=np.zeros_like(start_price, dtype=float), where=start_price != 0)


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


def is_valid_df(df: pd.DataFrame) -> bool:
    if df.isna().values.any() or df.isnull().values.any():
        return False
    if (df['start_price'] < 0).values.any() or (df['start_price'] > 499999).values.any():
        return False
    return True


def is_valid_trading_sample(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    if (df['event_count'] == 0).values.sum() > 2:
        return False
    return True


def compute_normalized_ofi(df: pd.DataFrame, levels: int = constants.LEVELS) -> pd.DataFrame:
    if not is_valid_df(df):
        return empty_df()
    average_vol_size = df[VOLUME_COLS[0]]
    for c in VOLUME_COLS[1:levels]:
        average_vol_size += df[c]
    average_vol_size = average_vol_size.astype(float)
    for c in SPLIT_OFI_COLS:
        df[c] = df[c].astype(float)
        df[c] = np.divide(df[c].astype(float), average_vol_size, out=np.zeros_like(df[c].astype(float), dtype=float), where=average_vol_size != 0)
    return df


def compute_ofi_df_from_split(df: pd.DataFrame) -> pd.DataFrame:
    if not is_valid_df(df):
        return empty_df()
    for i in range(1, constants.LEVELS + 1):
        cols = [f"{s}_{i}" for s in SPLIT_OFI_NAMES]
        ofi_col = f"ofi_{i}"
        df[ofi_col] = df[cols[0]]
        for c in cols[1:]:
            df[ofi_col] += df[c]
    return df


def compute_otofi_df_from_split(df: pd.DataFrame) -> pd.DataFrame:
    if not is_valid_df(df):
        return empty_df()
    for i in range(1, constants.LEVELS + 1):
        df[f'ofi_order_{i}'] = df[f'ofi_add_{i}'] + df[f'ofi_cancel_{i}']
    return df


def compute_bucket_ofi_df_from_tick_ofi(df: pd.DataFrame, props: BucketOFIProps) -> pd.DataFrame:
    # Remove data outside range
    df.drop(df[(df['time'] < props.start_time) | (df['time'] >= props.end_time)].index, inplace=True)

    if df.empty:
        return empty_df()

    # Compute start time of bucket
    df['start_time'] = (df['time'] // props.bucket_size).astype(int) * props.bucket_size

    # Divide OFI between types
    for i in range(1, props.levels + 1):
        for t in OFI_TYPES:
            df[f"ofi_{t}_{i}"] = np.where(df["ofi_type"] == t, df[f"ofi_{i}"], 0)

    group_df = df.groupby('start_time')
    df = group_df[SPLIT_OFI_COLS].agg('sum')
    df[VOLUME_COLS] = group_df[VOLUME_COLS].agg('mean')
    df['start_price'] = group_df.apply(lambda df: df['start_price'].tolist()[0])
    df['end_price'] = group_df.apply(lambda df: df['end_price'].tolist()[-1])
    df['event_count'] = group_df.agg('size')
    df['return'] = compute_return(df['start_price'], df['end_price'])
    # df['return_future'] = compute_return_future(df['return_now'])

    # Compute normalized OFI
    # !!! There should always be at least an order on the market during the whole day.
    # for i in range(1, props.levels + 1):
    #     for name in SPLIT_OFI_NAMES:
    #         df[f"{name}_{i}"] = np.divide(df[f"{name}_{i}"], df[f"volume_{i}"],
    #                                       out=np.ones_like(df[f"{name}_{i}"], dtype=float),
    #                                       where=df[f"volume_{i}"] != 0)

    # Find missing buckets and set values for them
    all_indices = pd.Index(range(props.start_time, props.end_time, props.bucket_size)).astype(int)
    now_indices = df.index
    first_idx = props.end_time + 1 if len(now_indices) == 0 else now_indices[0]
    missing_indices = all_indices.difference(now_indices)
    missing_indices = missing_indices[missing_indices >= first_idx]
    prev_index = now_indices[np.searchsorted(now_indices, missing_indices) - 1]
    prev_prices = df.loc[prev_index, 'end_price']

    df = df.reindex(all_indices, fill_value=0, copy=False)
    df['start_time'] = df.index
    # Set start and end price for missing buckets as the previous end price
    for (idx, prv) in zip(missing_indices, prev_prices):
        df.loc[idx, 'start_price'] = prv
        df.loc[idx, 'end_price'] = prv

    if props.rounding:
        round_df(df)

    return df[bucket_ofi_cols]


def compute_bucket_ofi_df_from_bucket_ofi(df: pd.DataFrame, props: BucketOFIProps) -> pd.DataFrame:
    if props.bucket_size % props.prev_bucket_size != 0:
        raise ValueError(
            f"New bucket size is not a multiple of previous bucket size! {props.prev_bucket_size} ; {props.bucket_size}")

    # Recompute sum of OFIs and sum of volumes
    # for i in range(1, props.levels + 1):
    #     for name in SPLIT_OFI_NAMES:
    #         df[f"{name}_{i}"] *= df[f"volume_{i}"]
    #     df[f"volume_{i}"] *= df['event_count']

    for i in range(1, props.levels + 1):
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
    df['return'] = compute_return(df['start_price'], df['end_price'])
    # df['return_future'] = compute_return_future(df['return_now'], roll=window_size)

    pd.DataFrame.dropna(df, inplace=True)

    for c in SPLIT_OFI_COLS + ['start_time', 'event_count']:
        df[c] = df[c].astype(int)
    # df['start_time'] = df['start_time'].astype(int)
    # df['event_count'] = df['event_count'].astype(int)

    df.drop(df[df['start_time'] % props.rolling_size != 0].index, inplace=True)

    # Compute normalised OFI
    # for i in range(1, props.levels + 1):
    #     volume_col = f"volume_{i}"
    #     df[volume_col] = np.divide(df[volume_col], df["event_count"], out=np.zeros_like(df[volume_col]),
    #                                where=df["event_count"] != 0)
    #     for name in SPLIT_OFI_NAMES:
    #         ofi_col = f"{name}_{i}"
    #         df[ofi_col] = np.divide(df[ofi_col], df[volume_col], out=np.zeros_like(df[ofi_col]),
    #                                 where=df[volume_col] != 0)

    for i in range(1, props.levels + 1):
        c = f"volume_{i}"
        df[c] = np.divide(df[c], df['event_count'], out=np.zeros_like(df[c], dtype=float), where=df['event_count'] != 0)

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
        return empty_df()
