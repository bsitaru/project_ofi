from typing import List

import numpy as np
import pandas as pd

from constants import levels_list
from data_manipulation.bucket_ofi import compute_normalized_ofi, compute_otofi_df_from_split, compute_ofi_df_from_split, \
    is_valid_trading_sample, compute_bucket_ofi_df_from_bucket_ofi, BucketOFIProps

from abc import ABC, abstractmethod


class Selector(ABC):

    def __init__(self):
        self.column_names = []

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def select_interval_df(self, df: pd.DataFrame, left: int, right: int) -> np.ndarray:
        # interval (left, right]
        df = df[(df['time'] > left) & (df['time'] <= right)]
        if not is_valid_trading_sample(df):
            df = pd.DataFrame().to_numpy()
        else:
            df = df[self.column_names]
        ret = df.to_numpy()
        return ret


class DataSelector(Selector):
    def __init__(self, volume_normalize: bool):
        super().__init__()
        self.volume_normalize = volume_normalize

    def pre_process(self, df: pd.DataFrame):
        if self.volume_normalize:
            df = compute_normalized_ofi(df)
        return df

    def process(self, df: pd.DataFrame):
        df = self.pre_process(df)
        df = self.custom_processor(df)
        return df

    def custom_processor(self, df: pd.DataFrame):
        return df


class SplitOFISelector(DataSelector):
    def __init__(self, levels, **kwargs):
        super(SplitOFISelector, self).__init__(**kwargs)
        self.name = f"SplitOFI_{levels}"
        self.column_names = self.column_names + levels_list('ofi_add', levels) + levels_list('ofi_cancel', levels) \
                            + levels_list('ofi_trade', levels)

class AddOFISelector(DataSelector):
    def __init__(self, levels, **kwargs):
        super(AddOFISelector, self).__init__(**kwargs)
        self.name = f"AddOFI_{levels}"
        self.column_names = self.column_names + levels_list('ofi_add', levels)

class OTOFISelector(DataSelector):
    def __init__(self, levels, **kwargs):
        super(OTOFISelector, self).__init__(**kwargs)
        self.name = f"OTOFI_{levels}"
        self.column_names = self.column_names + levels_list('ofi_order', levels) + levels_list('ofi_trade', levels)

    def custom_processor(self, df: pd.DataFrame):
        return compute_otofi_df_from_split(df)


class OFISelector(DataSelector):
    def __init__(self, levels, **kwargs):
        super(OFISelector, self).__init__(**kwargs)
        self.name = f"OFI_{levels}"
        self.column_names = self.column_names + levels_list('ofi', levels)

    def custom_processor(self, df: pd.DataFrame):
        return compute_ofi_df_from_split(df)


class ReturnSelector(DataSelector):
    def __init__(self, y_lag: int):
        super(ReturnSelector, self).__init__(volume_normalize=False)
        self.y_lag = y_lag
        self.column_names = 'return'

    def select_interval_df(self, df: pd.DataFrame, left: int, right: int) -> np.ndarray:
        left += self.y_lag
        right += self.y_lag
        return super(ReturnSelector, self).select_interval_df(df, left, right)


class MultiHorizontSelector(Selector):
    def __init__(self, selector: Selector, horizonts: List[int], bucket_size: int):
        super().__init__()
        self.selector = selector
        self.horizonts = horizonts
        self.bucket_size = bucket_size
        self.column_names = sum([[f"{n}_{h}" for n in selector.column_names] for h in horizonts], [])

    def process(self, df: pd.DataFrame):
        initial_df = df
        # add column suffix for horizont 1
        df_list = []
        one_columns = ['return', 'event_count', 'start_price', 'end_price']
        for i, h in enumerate(self.horizonts):
            if type(initial_df) == list:  # already loaded from file
                df = initial_df[i]
            else:
                df = initial_df.copy()
                if h != 1:
                    df = initial_df.copy()
                    props = BucketOFIProps(bucket_size=h * self.bucket_size, prev_bucket_size=self.bucket_size,
                                           rolling_size=self.bucket_size, rounding=True)
                    df = compute_bucket_ofi_df_from_bucket_ofi(df, props)

            df = self.selector.process(df)
            remap_dict = {f"{n}": f"{n}_{h}" for n in self.selector.column_names}
            df.rename(columns=remap_dict, inplace=True)

            cols = np.array(df.columns)
            cols = np.setdiff1d(cols, np.array(list(remap_dict.values()) + one_columns + ['time']))
            cols = cols.tolist()
            if h != 1:
                cols = cols + one_columns
            df.drop(columns=cols, inplace=True)
            df_list.append(df)

        df_list = [df.set_index('time') for df in df_list]
        df = df_list[0].join(df_list[1:], how='inner')
        df['time'] = df.index
        return df


def factory(all_args):
    args = all_args.selector
    classes = {'OFI': OFISelector, 'SplitOFI': SplitOFISelector, 'OTOFI': OTOFISelector, 'Return': ReturnSelector, 'AddOFI': AddOFISelector}
    if args.type not in classes.keys():
        raise ValueError(f"invalid selector type {args.type}")
    constructor = classes[args.type]
    if args.type in ['OFI', 'SplitOFI', 'OTOFI', 'AddOFI']:
        selector = constructor(volume_normalize=args.volume_normalize, levels=args.levels)
    else:
        raise ValueError(f'invalid selector {args.type}')
    if 'multi_horizonts' in args:
        selector = MultiHorizontSelector(selector=selector, horizonts=args.multi_horizonts,
                                         bucket_size=all_args.horizont)
    return selector


def return_factory(y_lag: int = 0):
    return ReturnSelector(y_lag)
