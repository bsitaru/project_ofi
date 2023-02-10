import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


def read_csv(file: str, process_df) -> (np.ndarray, np.ndarray):
    df = pd.read_csv(file)
    return process_df(df)


class LinRegModel:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.model = LinearRegression()

    def fit(self, data: (np.ndarray, np.ndarray)) -> ():
        x, y = data
        self.model.fit(x, y)

    def score(self, name: str, data: (np.ndarray, np.ndarray)) -> ():
        x, y = data
        s = self.model.score(x, y)
        print(f"{name} score: {s}")

    def fit_and_score(self):
        self.fit(self.train_data)
        self.score(name=f"{self.name}_train", data=self.train_data)
        if self.test_data:
            self.score(name=f"{self.name}_test", data=self.test_data)
