from typing import Tuple

class Car:
    def __init__(self) -> None:
        self.dimensions = (0, 0)
        self.front_windshield = None
        self.rear_windshield = None
        self.color = None
        self.random_noise = []
        pass
    def set_dimensions(self, dimensions: Tuple[float, float]):
        self.dimensions = dimensions
    def add_noise(self, noise):
        pass
    def build(self):
        pass
