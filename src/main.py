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

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill missing values
df[numerical_cols] = df[numerical_cols].apply(lambda col: col.fillna(col.median()))
df[categorical_cols] = df[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]))

# Confirm missing values have been handled
print("\nMissing values after cleaning:\n", df.isnull().sum())
