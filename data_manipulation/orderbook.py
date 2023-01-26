import pandas as pd

from constants import LEVELS


def get_orderbook_columns(levels: int = LEVELS):
    cols = [[f"bid_price_{i}", f"bid_size_{i}", f"ask_price_{i}", f"ask_size_{i}"] for i in range(1, levels + 1)]
    cols = sum(cols, [])
    return cols


def get_orderbook_df(orderbook_file: str, levels: int = LEVELS) -> pd.DataFrame:
    cols = get_orderbook_columns(levels)
    df = pd.read_csv(orderbook_file, header=None, names=cols)
    return df


def get_empty_orderbook_line_df(levels: int = LEVELS) -> pd.DataFrame:
    dct = {}
    for i in range(1, levels + 1):
        dct[f"bid_price_{i}"] = 9999999999
        dct[f"bid_size_{i}"] = 0
        dct[f"ask_price_{i}"] = -9999999999
        dct[f"ask_size_{i}"] = 0
    return pd.DataFrame(dct, index=[0])


def compute_two_line_orderbook_df(orderbook_df: pd.DataFrame) -> pd.DataFrame:
    df = orderbook_df.copy()
    empty_line = get_empty_orderbook_line_df(LEVELS)
    df = pd.concat([empty_line, df]).reset_index(drop=True)
    df.drop(df.tail(1).index, inplace=True)
    df = df.add_prefix('prev_')

    df = pd.concat([orderbook_df, df], axis=1)
    return df
