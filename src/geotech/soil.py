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

    @abstractmethod
    def __repr__(self) -> str:
        pass


class Uniform(ParameterDistribution):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def sample(self):
        return np.random.uniform(self.lower, self.upper)

    def __repr__(self) -> str:
        return f"{self.lower} to {self.upper}"


class Normal(ParameterDistribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return np.random.normal(self.mean, self.std)

    def __repr__(self) -> str:
        return f"{self.mean} +- {self.std}"


class LogNormal(ParameterDistribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return np.random.lognormal(self.mean, self.std)

    def __repr__(self) -> str:
        return f"{self.mean} */ {self.std}"


class Constant(ParameterDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def __repr__(self) -> str:
        return f"{self.value}"


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

    @abstractmethod
    def __repr__(self) -> str:
        pass


class WaterTable(PoreWaterPressure):
    def __init__(
        self,
        water_table_elevation: ParameterDistribution,
        gradient: ParameterDistribution = Constant(0.0),
    ):
        """Initialize pore water pressure

        Args:
            water_table_elevation: Elevation of the groundwater table
            gradient: Pore water pressure gradient in m/m. Positive indicates
                artesian conditions. Defaults to 0.0.
        """
        self.water_table_elevation = water_table_elevation  # m
        self.gradient = gradient  # m/m

    def sample(self, elevation):
        water_table_elevation = self.water_table_elevation.sample()
        hydrostatic = (water_table_elevation - elevation) * self.water_unit_weight
        artesian = (
            (water_table_elevation - elevation)
            * self.water_unit_weight
            * self.gradient.sample()
        )
        return hydrostatic + artesian

    def __repr__(self):
        return f"Water table at {self.water_table_elevation} m"


class MeasuredPoreWaterPressure(PoreWaterPressure):
    def __init__(
        self,
        pore_water_pressure_measurements: list[float, ParameterDistribution],
    ):
        """Initialize pore water pressure

        Args:
            pore_water_pressure_measurements: List of pore water pressure measurements
                as tuples of elevation (m) and pore water pressure (kPa).
        """
        self.pore_water_pressure_measurements = sorted(
            pore_water_pressure_measurements, reverse=True
        )

    def sample(self, elevation):
        return self.groundwater_elevation.sample() + self.gradient.sample()

    def __repr__(self):
        return f"Measured PWP"


class SoilLayer:
    def __init__(
        self,
        name: str,
        elevation_top: ParameterDistribution,
        elevation_bottom: ParameterDistribution,
        wet_density: ParameterDistribution,
        dry_density: ParameterDistribution,
        cohesion: ParameterDistribution,
        angle_of_internal_friction: ParameterDistribution,
        compression_index: ParameterDistribution,
        recompression_index: ParameterDistribution,
        initial_void_ratio: ParameterDistribution,
    ):
        self.name = name
        self.elevation_top = elevation_top  # m
        self.elevation_bottom = elevation_bottom  # m
        self.wet_density = wet_density  # kN/m3
        self.dry_density = dry_density  # kN/m3
        self.cohesion = cohesion  # kPa
        self.angle_of_internal_friction = angle_of_internal_friction  # degrees
        self.compression_index = compression_index  # m2/kN
        self.recompression_index = recompression_index  # m2/kN
        self.initial_void_ratio = initial_void_ratio  # unitless

    def is_wet(self, groundwater_depth):
        return self.elevation_top >= groundwater_depth

    def get_density(self, groundwater_depth):
        if self.is_wet(groundwater_depth):
            return self.wet_density
        else:
            return self.dry_density

    def __repr__(self):
        return f"{self.name} ({self.elevation_top}m to {self.elevation_bottom}m)"


class SoilProfile:
    def __init__(
        self,
        layers: list[float, SoilLayer],
        pore_water_pressure: PoreWaterPressure = Constant(0.0),
    ):
        self.layers = []
        self.porewater_pressure = None
