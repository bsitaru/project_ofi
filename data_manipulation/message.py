import pandas as pd

cols = ['time', 'event_type', 'order_id', 'size', 'price', 'direction', '_']
c_dtype = {'time': float, 'event_type': int, 'order_id': int, 'size': int, 'price': int, 'direction': int, '_': str}


def get_message_df(message_file: str) -> pd.DataFrame:
    df = pd.read_csv(message_file, header=None, names=cols, dtype=c_dtype)
    return df
