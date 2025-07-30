"""
Given an existing dataset with head annotations, generate more data by cropping random images from it and applying a series of transformations.
"""

import json
import pathlib

dataset_path = pathlib.Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/dataset"
annotation_file = dataset_path / "annotations.json"

with open(annotation_file, 'r') as f:
	data = json.load(f)
	print(data)