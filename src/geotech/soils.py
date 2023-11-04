import logging
from abc import ABC, abstractmethod

import numpy as np

from geotech.config import Config
from geotech.distributions import Constant, ParameterDistribution

logger = logging.getLogger(__name__)
config = Config()


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

        for attr_name, attr_value in self.__dict__.items():
            if not isinstance(attr_value, ParameterDistribution):
                attr_value = Constant(attr_value)

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
        # TODO, format the measurements nicely.
        raise NotImplementedError


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

        for attr_name, attr_value in self.__dict__.items():
            if not isinstance(attr_value, ParameterDistribution):
                attr_value = Constant(attr_value)

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
        layers: list[float, SoilLayer] = [],
        pore_water_pressure: PoreWaterPressure = Constant(0.0),
    ):
        self.layers = layers
        self.porewater_pressure = pore_water_pressure

        if not isinstance(self.porewater_pressure, ParameterDistribution):
            self.porewater_pressure = Constant(self.porewater_pressure)

    def add_layer(self, layer: SoilLayer):
        self.layers.append(layer)
        self.layers.sort(key=lambda x: x.elevation_top, reverse=True)

    def get_samples(self):
        samples = {}
        for layer in self.layers:
            samples[layer.name] = {
                "elevation_top": layer.elevation_top.sample(),
                "elevation_bottom": layer.elevation_bottom.sample(),
                "wet_density": layer.wet_density.sample(),
                "dry_density": layer.dry_density.sample(),
                "cohesion": layer.cohesion.sample(),
                "angle_of_internal_friction": layer.angle_of_internal_friction.sample(),
                "compression_index": layer.compression_index.sample(),
                "recompression_index": layer.recompression_index.sample(),
                "initial_void_ratio": layer.initial_void_ratio.sample(),
            }
        return samples
