import logging

import numpy as np

from geotech.config import Config
from geotech.loads import Load
from geotech.soils import SoilProfile

logger = logging.getLogger(__name__)
config = Config()


def calculate_settlements(profile: SoilProfile, load: Load, time: float) -> np.ndarray:
    """Calculate the settlements for a soil profile.

    Parameters
    ----------
    profile : SoilProfile
        The soil profile to calculate the settlements for.
    load : float
        The load applied to the soil profile.
    time : float
        The time for which to calculate the settlements.

    Returns
    -------
    settlements : np.ndarray
        The settlements for the soil profile.
    """
    # settlements = np.zeros(config.iterations)
    # for layer in profile.layers:
    #     settlements += layer.calculate_settlements(load, time)
    # return settlements
    pass
