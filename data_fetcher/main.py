"""
Extracts cars from the boxes in the `parking_lots` folder and applies transformations for randomization.
"""
import json
from PIL import Image

count = 0

with open("./data/parking_lots/via_project_24Jun2025_22h44m.json") as json_file:
    json_data = json.load(json_file)
    for file in json_data["_via_img_metadata"]:
        filename = json_data["_via_img_metadata"][file]["filename"]
        regions = json_data["_via_img_metadata"][file]["regions"]
        for region in regions:
            im = Image.open(f"./data/parking_lots/{filename}")
            x1 = region["shape_attributes"]["x"]
            y1 = region["shape_attributes"]["y"]
            x2 = x1 + region["shape_attributes"]["width"]
            y2 = y1 + region["shape_attributes"]["height"]
            print(x1, y1, x2, y2)
            im1 = im.crop((x1, y1, x2, y2))
            im1.save(f"./data/cars/car{count}.png")
            count += 1