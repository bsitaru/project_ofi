import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

class Standardizer(SkDecorator):
    def __init__(self, data_processor: DataProcessor):
        super().__init__(data_processor, StandardScaler())


class PCAProcessor(SkDecorator):
    def __init__(self, data_processor: DataProcessor, n_components: int = None):
        super().__init__(data_processor, PCA(n_components=n_components))

    def explained_variance_ratio(self):
        ans = self.sk_component.explained_variance_ratio_
        return np.cumsum(ans)


class MultiPCA(Decorator):
    def __init__(self, data_processor: DataProcessor, n_groups, n_components: int = None):
        super().__init__(data_processor)
        self.n_groups = n_groups
        self.pcas = [PCA(n_components) for i in range(n_groups)]

    def fit(self, x: np.ndarray):
        x = self.data_processor.fit(x)
        xs = np.hsplit(x, self.n_groups)
        xs = [pca.fit_transform(x) for (x, pca) in zip(xs, self.pcas)]
        return np.column_stack(xs)

    def process(self, x: np.ndarray):
        x = self.data_processor.process(x)
        xs = np.hsplit(x, self.n_groups)
        xs = [pca.transform(x) for (x, pca) in zip(xs, self.pcas)]
        return np.column_stack(xs)


def factory_individual(args):
    processor = DataProcessor()
    if args.normalize:
        processor = Normalize(processor)
    return processor


def factory_group(args):
    processor = DataProcessor()
    if 'pca' in args:
        processor = PCAProcessor(processor, n_components=args.pca)
    if 'multipca' in args:
        processor = MultiPCA(processor, n_groups=args.multipca.groups, n_components=args.multipca.components)
    if args.standardize:
        processor = Standardizer(processor)
    return processor
