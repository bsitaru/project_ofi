import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, df: pd.DataFrame, column_names, horizont, roll_y: bool = False, start_time=None, end_time=None):
        if start_time is not None:
            df = df[df['time'] > start_time]
        if end_time is not None:
            df = df[df['time'] <= end_time]

        df = self.check_prices(df)

        if df.empty:
            self.x, self.y = None, None
            return

        x = df[column_names].to_numpy()
        y = df['return'].to_numpy()
        event_count = df['event_count'].to_numpy()

        left_time = df['time'].iloc[0]
        right_time = df['time'].iloc[-1]

        if roll_y:  # future prediction, so roll y, remove last
            if np.shape(x)[0] == 1:
                self.x, self.y = None, None
                return
            x = x[:-1]
            y = y[1:]
            event_count = event_count[:-1]
            right_time = df['time'].iloc[-2]

        self.x, self.y, self.event_count = x, y, event_count
        self.left_time, self.right_time = left_time, right_time
        self.horizont = horizont
        self.size = np.shape(y)[0]

    def check_prices(self, df):
        bad_prices = (df['start_price'] <= 0.0001) | (df['end_price'] <= 0.0001) | (df['start_price'] > 499998.0) | (df['end_price'] > 499998.0)
        if bad_prices.values.any():
            max_time = df[bad_prices].iloc[0]['time']
            df = df[df['time'] < max_time]
        return df

    def select_interval(self, left, right) -> (np.ndarray, np.ndarray):
        idx_left = (left - self.left_time - 1) // self.horizont + 1
        idx_right = (right - self.left_time) // self.horizont + 1
        if idx_right > self.size or idx_left < 0:
            return None, None
        if (self.event_count[idx_left:idx_right] == 0).sum() > 1:
            return None, None
        return self.x[idx_left:idx_right], self.y[idx_left:idx_right]
