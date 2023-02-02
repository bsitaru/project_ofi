import pandas as pd


def get_message_df(message_file: str) -> pd.DataFrame:
    cols = ['time', 'event_type', 'order_id', 'size', 'price', 'direction', '_']
    df = pd.read_csv(message_file, header=None, names=cols, low_memory=False)
    return df
