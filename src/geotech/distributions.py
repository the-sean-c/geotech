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
        """Initialize the distribution parameters

        Randomized sampling should be done here to maintain consistency between
        different elevations and locations in the same iteration.
        """
        self.sample_values = None

    def sample(
        self,
        northings: float | list[float] = 0.0,
        eastings: float | list[float] = 0.0,
        elevations: float | list[float] = 0.0,
    ):
        """Return a sample from the distribution."""
        for param in [northings, eastings, elevations]:
            if isinstance(param, (int, float)):
                param = [param]

        return np.broadcast_to(
            self.sample_values,
            (len(northings), len(eastings), len(elevations), config.iterations),
        ).copy()

    @abstractmethod
    def __repr__(self) -> str:
        pass


class Uniform(ParameterDistribution):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.sample_values = config.rng.uniform(
            self.lower, self.upper, size=config.iterations
        )

    def __repr__(self) -> str:
        return f"{self.lower} to {self.upper}"


class Normal(ParameterDistribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.sample_values = config.rng.normal(
            self.mean, self.std, size=config.iterations
        )

    def __repr__(self) -> str:
        return f"{self.mean} +- {self.std}"


class LogNormal(ParameterDistribution):
    def __init__(self, underlying_mean, underlying_std):
        self.mu = underlying_mean  # Of underlying normal distribution
        self.sigma = underlying_std  # Of underlying normal distribution
        self.sample_values = config.rng.lognormal(
            self.mu, self.sigma, size=config.iterations
        )

    def __repr__(self) -> str:
        # TODO, figure out a way to do this so it looks right.
        raise NotImplementedError


class Constant(ParameterDistribution):
    def __init__(self, value):
        self.value = value
        self.sample_values = np.ones(config.iterations) * self.value

    def __repr__(self) -> str:
        return f"{self.value}"


class Bootstrap(ParameterDistribution):
    def __init__(self, samples):
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
