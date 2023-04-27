import numpy as np
import pandas as pd

from constants import levels_list
from data_manipulation.bucket_ofi import compute_normalized_ofi, compute_otofi_df_from_split, compute_ofi_df_from_split, \
    is_valid_trading_sample


class DataSelector:
    def __init__(self, volume_normalize: bool):
        self.volume_normalize = volume_normalize
        self.column_names = []

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

    def select_interval_df(self, df: pd.DataFrame, left: int, right: int) -> np.ndarray:
        df = df[(df['time'] > left) & (df['time'] <= right)]
        if not is_valid_trading_sample(df):
            return pd.DataFrame().to_numpy()
        return df[self.column_names].to_numpy()


class SplitOFISelector(DataSelector):
    def __init__(self, levels, **kwargs):
        super(SplitOFISelector, self).__init__(**kwargs)
        self.name = f"SplitOFI_{levels}"
        self.column_names = self.column_names + levels_list('ofi_add', levels) + levels_list('ofi_cancel',
                                                                                             levels) + levels_list(
            'ofi_trade', levels)


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
    def __init__(self):
        super(ReturnSelector, self).__init__(volume_normalize=False)
        self.column_names = 'return'


def factory(args):
    classes = {'OFI': OFISelector, 'SplitOFI': SplitOFISelector, 'OTOFI': OTOFISelector, 'Return': ReturnSelector}
    if args.type not in classes.keys():
        raise ValueError(f"invalid selector type {args.type}")
    constructor = classes[args.type]
    if args.type in ['OFI', 'SplitOFI', 'OTOFI']:
        return constructor(volume_normalize=args.volume_normalize, levels=args.levels)
    else:
        raise ValueError(f'invalid selector {args.type}')


def return_factory():
    return ReturnSelector()
