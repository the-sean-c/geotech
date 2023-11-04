import logging

import numpy as np

logger = logging.getLogger(__name__)


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.iterations = 1000
            cls._instance.rng = np.random.default_rng(seed=42)
        return cls._instance
