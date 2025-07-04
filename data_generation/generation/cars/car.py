from typing import Tuple
from enum import Enum

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

df = pd.read_csv("data_generation/data/dimensions.csv")

mean = np.mean(df, axis=0)
cov = np.cov(df, rowvar=0) #type: ignore

y = multivariate_normal(mean=mean, cov=cov)

class CarType(Enum):
    CAR = 0
    PICKUP = 1
    VAN = 1

class CarGenerator:
    def __init__(self) -> None:
        self.dimensions = y.rvs()
        self.curvature = 0
        self.color = (0, 0, 0)

        self.type = CarType(0)
        self.front_windshield = {}
        self.rear_windshield = {}

        self.effects = {
            "shine": False,
            "roof_squares": [],
            "roof_lines": [],
            "transparency": 1.0,
            "clutter_pixels": [],
            "saturation": 0,
            "shadow": None
        }
    