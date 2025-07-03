import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

df = pd.read_csv("data_generation/data/dimensions.csv")

mean = np.mean(df, axis=0)
cov = np.cov(df, rowvar=0) #type: ignore

y = multivariate_normal(mean=mean, cov=cov)
print(y.rvs())
