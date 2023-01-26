# Tick OFI - OFI for each event
import pandas as pd
import numpy as np

from constants import LEVELS, OFI_TYPES, START_TRADE, END_TRADE
from data_manipulation.message import get_message_df
from data_manipulation.orderbook import get_orderbook_df, compute_two_line_orderbook_df

def compute_ofi_levels(orderbook_df: pd.DataFrame) -> pd.DataFrame:
    # Compute dataframe with line and previous line.
    df = compute_two_line_orderbook_df(orderbook_df)

    for i in range(1, LEVELS + 1):
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


def compute_tick_ofi_df(message_df: pd.DataFrame, orderbook_df: pd.DataFrame, drop_outside_trading_frame: bool = True) -> pd.DataFrame:
    df = message_df[['time']].copy()
    df['ofi_type'] = compute_ofi_type(message_df)

    ofi_df = compute_ofi_levels(orderbook_df)

    df = pd.concat([df, ofi_df], axis=1)

    cols = ['time', 'price', 'ofi_type'] + \
           [f'ofi_{i}' for i in range(1, LEVELS + 1)] + \
           [f'volume_{i}' for i in range(1, LEVELS + 1)]

    if drop_outside_trading_frame:
        df.drop(df[(START_TRADE > df['time']) | (df['time'] > END_TRADE)].index, inplace=True)

    return df[cols]


def generate_tick_ofi_file(message_file: str, orderbook_file: str, tick_ofi_file: str) -> None:
    message_df = get_message_df(message_file)
    orderbook_df = get_orderbook_df(orderbook_file)
    tick_ofi_df = compute_tick_ofi_df(message_df, orderbook_df)
    tick_ofi_df.to_csv(tick_ofi_file, index=False)
