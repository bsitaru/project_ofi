# Tick OFI - OFI for each event
import pandas as pd
import numpy as np

from constants import OFI_TYPES, START_TRADE, END_TRADE
from data_manipulation.message import get_message_df
from data_manipulation.orderbook import get_orderbook_df, compute_two_line_orderbook_df


def compute_ofi_levels(orderbook_df: pd.DataFrame, levels: int) -> pd.DataFrame:
    # Compute dataframe with line and previous line.
    df = compute_two_line_orderbook_df(orderbook_df, levels)

    for i in range(1, levels + 1):
        # Compute OFI
        df[f"ofi_{i}"] = \
            (df[f"bid_price_{i}"] >= df[f"prev_bid_price_{i}"]) * df[f"bid_size_{i}"] \
            - (df[f"bid_price_{i}"] <= df[f"prev_bid_price_{i}"]) * df[f"prev_bid_size_{i}"] \
            - (df[f"ask_price_{i}"] <= df[f"prev_ask_price_{i}"]) * df[f"ask_size_{i}"] \
            + (df[f"ask_price_{i}"] >= df[f"prev_ask_price_{i}"]) * df[f"prev_ask_size_{i}"]

        # Compute average volume at this level.
        df[f"volume_{i}"] = (df[f"bid_size_{i}"] + df[f"ask_size_{i}"]) / 2.0

    # Represent price as mean of bid and ask price. Divide by 10000 because LOBSTER data is stored in this way.
    df["price"] = (df['bid_price_1'] + df['ask_price_1']) / 20000.0

    return df


def compute_ofi_type(message_df: pd.DataFrame) -> np.ndarray:
    ofi_type_conditions = [
        (message_df['event_type'].isin([1])),
        (message_df['event_type'].isin([2, 3])),
        (message_df['event_type'].isin([4, 5, 6, 7]))
    ]
    ans = np.select(ofi_type_conditions, OFI_TYPES)
    return ans


def compute_tick_ofi_df(message_df: pd.DataFrame, orderbook_df: pd.DataFrame, levels: int,
                        trading_hours_only: bool = True) -> pd.DataFrame:
    df = message_df[['time']].copy()
    df['ofi_type'] = compute_ofi_type(message_df)

    ofi_df = compute_ofi_levels(orderbook_df, levels)

    df = pd.concat([df, ofi_df], axis=1)

    cols = ['time', 'price', 'ofi_type'] + \
           [f'ofi_{i}' for i in range(1, levels + 1)] + \
           [f'volume_{i}' for i in range(1, levels + 1)]

    if trading_hours_only:
        df.drop(df[(START_TRADE > df['time']) | (df['time'] > END_TRADE)].index, inplace=True)

    return df[cols]
