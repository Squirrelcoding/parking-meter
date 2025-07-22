import json

import pandas as pd
import numpy as np

from numpy import random
from scipy.stats import multivariate_normal

df = pd.read_csv("data_generation/data/dimensions.csv")

mean = np.mean(df, axis=0)
cov = np.cov(df, rowvar=0)  # type: ignore

y = multivariate_normal(mean=mean, cov=cov)

color_distribution = {
    "white": 0.239,
    "black": 0.232,
    "gray": 0.155,
    "silver": 0.145,
    "red": 0.103,
    "blue": 0.09,
    "brown": 0.014,
    "green": 0.007,
    "beige": 0.004,
    "orange": 0.004
}

class CarGenerator:
    def __init__(self) -> None:
        # Sample dimensions from the multivariate distribution
        self.dimensions = list(y.rvs())

        # Set a random corner radius
        self.corner_radius = random.uniform(20, 30)

        # Choose a color. Source: https://tempuslogix.com/most-popular-car-colors/
        self.color = random.choice(
            ["white", "black", "gray", "silver", "red",
                "blue", "brown", "green", "beige", "orange", "OTHER"],
            p=[0.239, 0.232, 0.155, 0.145, 0.103,
                0.09, 0.014, 0.007, 0.004, 0.004, 0.007]
        )

        # Generate the features of the front windshield
        self.front_windshield = {
            "color": (random.randint(20, 60), random.randint(30, 80), random.randint(60, 120)),
            "windshield_width": random.uniform(0.85, 1.0) * self.dimensions[1],
            "windshield_length": random.uniform(0.2, 0.25) * self.dimensions[0],
            "hood_length": random.uniform(0.0, 0.2) * self.dimensions[0]
        }

        # Generate the features of the rear windshield.
        self.rear_windshield = {
            "color": (random.randint(20, 60), random.randint(30, 80), random.randint(60, 120)),
            "windshield_width": random.uniform(0.85, 1.0) * self.dimensions[1],
            "windshield_length": random.uniform(0, 0.2) * self.dimensions[0],
            "trunk_length": random.uniform(0, 0.2) * self.dimensions[0]
        }

        self.effects = {
            "shine": None,
            "roof_squares": [],
            "transparency": 1.0,
            "clutter_pixels": [],
            "shadow": None,
            "blur": 0
        }
        
    def add_shine(self):
        x_coordinate = random.uniform(0, self.dimensions[0])
        y_coordinate = random.uniform(0, self.dimensions[1])

        radius = random.uniform(0, self.dimensions[0] / 2)

        blur_level = random.uniform(3, 5)
        self.effects["shine"] = {
            "x": x_coordinate,
            "y": y_coordinate,
            "radius": radius,
            "level": blur_level
        }

    def add_rectangle(self):
        width = random.uniform(0, self.dimensions[0])
        height = random.uniform(0, self.dimensions[1])
        x_coordinate = random.uniform(0, self.dimensions[0])
        y_coordinate = random.uniform(0, self.dimensions[1])
        color = None
        rotation = 0
        self.effects["roof_squares"].append({
            width, height, x_coordinate, y_coordinate, color, rotation
        })
        
    def set_transparency(self):
        self.effects["transparency"] = random.uniform(0, 12)
         
    def copy_region(self):
        """
        Randomly selects a rectangular region of the car, creates a copy, and shifts the copy to mimic satellite imaging errors.
        """
        x_coordinate = random.uniform(0, self.dimensions[0])
        y_coordinate = random.uniform(0, self.dimensions[1])

        radius = random.uniform(0, self.dimensions[0] / 2)

        blur_level = random.uniform(3, 5)
        self.effects["shine"] = {
            "x": x_coordinate,
            "y": y_coordinate,
            "radius": radius,
            "level": blur_level
        }
        
    def pixelate_region(self):
        """
        Randomly selects a rectangular region of the car and pixelates it.
        """
        x_coordinate = random.uniform(0, self.dimensions[0])
        y_coordinate = random.uniform(0, self.dimensions[1])

        radius = random.uniform(0, self.dimensions[0] / 2)

        blur_level = random.uniform(3, 5)
        self.effects["shine"] = {
            "x": x_coordinate,
            "y": y_coordinate,
            "radius": radius,
            "level": blur_level
        }      
        
    def add_static_noise(self):
        """
        Adds static noise to the car.
        """
        pass 
    
    def add_shadow(self):
        """
        Adds a shadow to the car.
        """
        pass 

    def build(self):
        return {
            "dimensions": self.dimensions,
            "corner_radius": self.corner_radius,
            "color": self.color,
            "front_windshield": self.front_windshield,
            "rear_windshield": self.rear_windshield,
            "effects": self.effects
        }
        
def generate_cars(output: str, n: int):
    results = {"cars": [CarGenerator().build() for _ in range(n)]}
    with open(output, 'w') as f:
        json.dump(results, f)
        
generate_cars("data_generation/data/cars.json", 1)