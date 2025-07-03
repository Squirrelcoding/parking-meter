from typing import Tuple
from enum import Enum

class CarType(Enum):
    CAR = 0
    PICKUP = 1
    VAN = 1

class Car:
    def __init__(self) -> None:
        self.dimensions = (0, 0)
        self.curvature = 0
        self.color = (0, 0, 0)
        self.type = CarType(0)
        self.front_windshield = None
        self.rear_windshield = None

        self.random_noise = []
