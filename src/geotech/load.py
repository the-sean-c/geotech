import logging
from abc import ABC, abstractmethod

import numpy as np

from geotech.config import Config
from geotech.distributions import Constant, ParameterDistribution

logger = logging.getLogger(__name__)
config = Config()


class Load(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, x: float, y: float, elevations: float):
        pass

    @abstractmethod
    def __repr__(self) -> str:
        return f"{self.load}"


class PointLoad(Load):
    def __init__(self, load, x_loc, y_loc):
        self.load = load

    def sample(self, x: float, y: float, elevations: float):
        return self.load

    def __repr__(self) -> str:
        return f"{self.load}"
