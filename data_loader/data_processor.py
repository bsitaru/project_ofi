import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    def __init__(self):
        pass

    def process(self, x: np.ndarray):
        return x

    def fit(self, x):
        return x


class Decorator(DataProcessor):
    _data_processor: DataProcessor = None

    def __init__(self, data_processor: DataProcessor):
        self._data_processor = data_processor

    @property
    def data_processor(self):
        return self._data_processor

    def process(self, x: np.ndarray):
        return self._data_processor.process(x)

    def fit(self, x: np.ndarray):
        return self._data_processor.fit(x)


class Normalize(Decorator):
    def __init__(self, data_processor: DataProcessor):
        super().__init__(data_processor)
        self.normalizer = MinMaxScaler()

    def fit(self, x: np.ndarray):
        x = self.data_processor.fit(x)
        self.normalizer.fit(x)
        return self.normalizer.transform(x)

    def process(self, x: np.ndarray):
        x = self.data_processor.process(x)
        return self.normalizer.transform(x)


def factory(args):
    processor = DataProcessor()
    if args.normalize:
        processor = Normalize(processor)
    return processor
