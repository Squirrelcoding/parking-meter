# Parking meter

A geospatial analysis project to analyze the use of parking lots using various satellite image datasets (i.e, NAIP, Sentinel-2, etc.) in the US. The goal is to observe how cars are distributed across parking lots throughout the various times of the day and the year.

The project will synthetically generate data using statistics and p5.js and use that data to train a model to identify real cars on parking lots from satellite images from which further inferences will be made. Parking lots will be found using OpenStreetMaps.

## Data collection

The data to be used to train the model to identify cars in a satellite image of a parking lot is synthetically generated due to the scarcity existing datasets. Labelling up to hundreds of cars in a single training image is extremely tedious. Furthermore, parking lots come in many shapes and sizes, meaning that thousands of cars would have to be labelled by hand in a reasonable amount of time - which is not possible. 

Because of this, we are instead choosing to synthetically generate satellite images of parking lots using Python and p5.js.

## Synthetic data generation of cars

Vehicles can widely vary in terms of their dimensions, colors, features, materials, etc. In order to synthetically generate cars such that the model can still generalize with accuracy, it is vital to use real-world data to generate new cars with heavy variation.

### Dimension generation

The website [carsized.com](https://www.carsized.com/en/) contains information about the dimensions of cars. We are especially interested in their widths and lengths since they are the only dimensions that are significant in a satellite image.

Under `data_generation/data_collection/dimensions` we have two Python files that collect the width and length of every car found in the index of the website into `data_generation/data/dimensions.csv`.

## Progress

- [ ] Synthetic data generation (SGD)
    - [ ] SDG for Cars
        - [X] Collect data for dimensions and create a multivariate normal distribution to sample from.
        - [ ] Collect data for colors and create a uniform distribution to sample from.
        - [ ] Use a distribution of the types of cars (SUV, pickup, van, etc.) and create a uniform distribution to sample from.
        - [ ] Determine a way to accurately generate the placement, size, and orientation of smaller features of the cars such as the windshields, mirrors, and windows.
            - [ ] Windshields
            - [ ] Mirrors
            - [ ] Windows 
        - [ ] Implement a variety of "random noise" to add to the car AFTER all of the basic features have been added.
            - [ ] Random squares
            - [ ] Random lines
            - [ ] Reflections
            - [ ] Slight transparency
            - [ ] Shadows
            - [ ] Saturation
- [ ] Model development
    - TO BE ADDED.
- [ ] Inference
    - TO BE ADDED.