# Parking meter

A geospatial analysis project to analyze the use of parking lots using various satellite image datasets (i.e, NAIP, Sentinel-2, etc.) in the US. The goal is to observe how cars are distributed across parking lots throughout the various times of the day and the year.

The project will synthetically generate data using statistics and p5.js and use that data to train a model to identify real cars on parking lots from satellite images from which further inferences will be made. Parking lots will be found using OpenStreetMaps.

## Data collection

The CarPK dataset is a dataset of over 1500 images of cars with bounding boxes in parking lots. We will preprocess and augment the dataset to ensure that it works best for our cars.

### Dimension generation

The website [carsized.com](https://www.carsized.com/en/) contains information about the dimensions of cars. We are especially interested in their widths and lengths since they are the only dimensions that are significant in a satellite image.

Under `data_generation/data_collection/dimensions` we have two Python files that collect the width and length of every car found in the index of the website into `data_generation/data/dimensions.csv`.

## Progress

- [ ] The current plan for right now is to transform the CARPK and COWC datasets and suit them for training the CSRNet model. This includes applying transformations and more importantly converting them into a head-annotation format suitable for density detection.