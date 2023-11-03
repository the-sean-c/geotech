from abc import ABC, abstractmethod

import numpy as np


class ParameterDistribution(ABC):
    """Abstract class for parameter distributions.

    Soil and water parameters can be defined by an arbitrary distribution. Desired
    distributions should be implemented as subclasses of this class.
    """

    @abstractmethod
    def __init__(self):
        """Must initialize the distribution parameters"""
        pass

    @abstractmethod
    def sample(self):
        """Must return a sample from the distribution"""
        pass


class Uniform(ParameterDistribution):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def sample(self):
        return np.random.uniform(self.lower, self.upper)


class Normal(ParameterDistribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return np.random.normal(self.mean, self.std)


class LogNormal(ParameterDistribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return np.random.lognormal(self.mean, self.std)


class Constant(ParameterDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value


class PoreWaterPressure(ABC):
    """Abstract class for pore water pressure.

    Porewater pressure can be defined by an arbitrary function. Desired functions
    should be implemented as subclasses of this class.
    """

    water_unit_weight = 9.81  # kN/m3

    @abstractmethod
    def __init__(self):
        """Must initialize the pore water pressure function parameters"""
        pass

    @abstractmethod
    def sample(self, elevation):
        """Must return a sample from the PWP distribution at a given elevation"""
        pass


class WaterTable(PoreWaterPressure):
    def __init__(
        self,
        groundwater_elevation: ParameterDistribution,
        gradient: ParameterDistribution = None,
    ):
        """Initialize pore water pressure

        Args:
            groundwater_elevation: Elevation of the groundwater table
            gradient: Pore water pressure gradient in m/m. Positive indicates
                artesian conditions. Defaults to None.
        """
        self.groundwater_elevation = groundwater_elevation
        self.gradient = gradient

    def sample(self, elevation):
        return self.groundwater_elevation.sample() + self.gradient.sample()


class SoilLayer:
    def __init__(
        self,
        name: str,
        depth_top: ParameterDistribution,
        depth_bottom: ParameterDistribution,
        wet_density: ParameterDistribution,
        dry_density: ParameterDistribution,
        cohesion: ParameterDistribution,
        angle_of_internal_friction: ParameterDistribution,
        compression_index: ParameterDistribution,
        recompression_index: ParameterDistribution,
        initial_void_ratio: ParameterDistribution,
    ):
        self.name = name
        self.depth_top = depth_top
        self.depth_bottom = depth_bottom
        self.wet_density = wet_density
        self.dry_density = dry_density
        self.cohesion = cohesion
        self.angle_of_internal_friction = angle_of_internal_friction
        self.compression_index = compression_index
        self.recompression_index = recompression_index
        self.initial_void_ratio = initial_void_ratio

    def is_wet(self, groundwater_depth):
        return self.depth_top >= groundwater_depth

    def get_density(self, groundwater_depth):
        if self.is_wet(groundwater_depth):
            return self.wet_density
        else:
            return self.dry_density

    def __repr__(self):
        return f"{self.name} ({self.depth_top}m to {self.depth_bottom}m)"


class SoilProfile:
    def __init__(self):
        pass
