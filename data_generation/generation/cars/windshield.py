"""
Generates the placement of windshields of a car (location, width, curvature, etc.)
"""
from numpy import random

def front_windshield(width: float, length: float):
    windshield_width = random.uniform(0.85, 1.0) * width
    windshield_length = random.uniform(0.2, 0.25) * length

    hood_length = random.uniform(0.25, 0.3) * length

    r = random.randint(20, 60)
    g = random.randint(30, 80)
    b = random.randint(60, 120)
    color = (r, g, b)

    return (windshield_width, windshield_length, hood_length, color)

def back_windshield(width: float, length: float):
    windshield_width = random.uniform(0.85, 1.0) * width
    windshield_length = random.uniform(0, 0.2) * length

    trunk_length = random.uniform(0, 0.2) * length

    r = random.randint(20, 60)
    g = random.randint(30, 80)
    b = random.randint(60, 120)
    color = (r, g, b)

    return (windshield_width, windshield_length, trunk_length, color)
