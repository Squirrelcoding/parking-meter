from typing import Tuple

class Car:
    def __init__(self) -> None:
        self.dimensions = (0, 0)
        self.front_windshield = None
        self.rear_windshield = None
        self.color = None
        self.random_noise = []
    def set_dimensions(self, dimensions: Tuple[float, float]):
        self.dimensions = dimensions
    def set_color(self, color):
        pass
    def add_noise(self, noise):
        pass
    def build(self):
        pass
