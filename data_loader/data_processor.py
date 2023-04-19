import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


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


class SkDecorator(Decorator):
    def __init__(self, data_processor: DataProcessor, sk_component):
        super().__init__(data_processor)
        self.sk_component = sk_component

    def fit(self, x: np.ndarray):
        x = self.data_processor.fit(x)
        return self.sk_component.fit_transform(x)

    def process(self, x: np.ndarray):
        x = self.data_processor.process(x)
        return self.sk_component.transform(x)


class Normalize(SkDecorator):
    def __init__(self, data_processor: DataProcessor):
        super().__init__(data_processor, MinMaxScaler())


class PCAProcessor(SkDecorator):
    def __init__(self, data_processor: DataProcessor, n_components: int = None):
        super().__init__(data_processor, PCA(n_components=n_components))


def factory(args):
    processor = DataProcessor()
    if args.normalize:
        processor = Normalize(processor)
    if 'pca' in args:
        processor = PCAProcessor(processor, n_components=args.pca)
    return processor
