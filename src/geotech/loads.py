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
    def sample_vertical_pressure(self, x: float, y: float, elevations: float):
        pass

    @abstractmethod
    def __repr__(self) -> str:
        return f"{self.load}"


class PointLoad(Load):
    def __init__(
        self,
        load: ParameterDistribution,
        elevation_load: ParameterDistribution,
        x_load: ParameterDistribution,
        y_load: ParameterDistribution,
    ):
        self.load = load  # kN
        self.elevation_load = elevation_load  # m
        self.x_load = x_load  # m
        self.y_load = y_load  # m

        for attr_name, attr_value in self.__dict__.items():
            if not isinstance(attr_value, ParameterDistribution):
                attr_value = Constant(attr_value)

    def sample_vertical_pressure(
        self, x: float, y: float, elevations: np.array
    ) -> np.array:
        """Calculate the vertical pressure at a given point per Boussinesq

        Returns:
            vertical pressure in kPa
        """
        x_load = self.x_load.sample()
        y_load = self.y_load.sample()
        elevation_load = self.elevation_load.sample()
        r = np.sqrt((x - x_load) ** 2 + (y - y_load) ** 2)
        R = np.sqrt(r**2 + elevations.T**2)
        Q = self.load.sample()
        z = elevation_load - elevations.T

        return (3 * Q) / (2 * np.pi * z**2) / (1 + (r / z) ** 2) ** (5 / 2)

    def __repr__(self) -> str:
        return f"{self.load}"
