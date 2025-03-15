#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "./cwdata.csv"
df = pd.read_csv(file_path)

# Display basic dataset information
print("Dataset Overview:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
