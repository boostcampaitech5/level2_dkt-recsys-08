import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """

    @abstractmethod
    def train(self):
        """
        Full training logic
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        """
        Full training logic
        """
        raise NotImplementedError
