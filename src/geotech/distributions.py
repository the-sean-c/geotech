import logging
from abc import ABC, abstractmethod

import numpy as np

from geotech.config import Config

logger = logging.getLogger(__name__)
config = Config()


class ParameterDistribution(ABC):
    """Abstract class for parameter distributions.

    Soil and water parameters can be defined by an arbitrary distribution. Desired
    distributions should be implemented as subclasses of this class.
    """

    @abstractmethod
    def __init__(self):
        """Initialize the distribution parameters"""
        pass

    @abstractmethod
    def sample(self):
        """Return a sample from the distribution.

        The sample should be an array of size config.iterations.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class Uniform(ParameterDistribution):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def sample(self):
        return config.rng.uniform(self.lower, self.upper, size=config.iterations)

    def __repr__(self) -> str:
        return f"{self.lower} to {self.upper}"


class Normal(ParameterDistribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return config.rng.normal(self.mean, self.std, size=config.iterations)

    def __repr__(self) -> str:
        return f"{self.mean} +- {self.std}"


class LogNormal(ParameterDistribution):
    def __init__(self, underlying_mean, underlying_std):
        self.mu = underlying_mean  # Of underlying normal distribution
        self.sigma = underlying_std  # Of underlying normal distribution

    def sample(self):
        return config.rng.lognormal(self.mu, self.sigma, size=config.iterations)

    def __repr__(self) -> str:
        # TODO, figure out a way to do this so it looks right.
        raise NotImplementedError


class Constant(ParameterDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return np.ones(config.iterations) * self.value

    def __repr__(self) -> str:
        return f"{self.value}"
