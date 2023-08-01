import numpy as np
from typing import Tuple
from abc import abstractmethod


class BaseDataLoader:
    """
    Base class for all data loaders
    """

    @abstractmethod
    def split_data(self):
        raise NotImplementedError

    @abstractmethod
    def preprocessing(self):
        raise NotImplementedError

    @abstractmethod
    def feature_engineering(self):
        raise NotImplementedError

    @abstractmethod
    def load_data_from_file(self):
        raise NotImplementedError
