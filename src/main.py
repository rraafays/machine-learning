#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
file_path = "./cwdata.csv"
df = pd.read_csv(file_path)

# Drop unnecessary unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Fill missing values
df[numerical_cols] = df[numerical_cols].apply(lambda col: col.fillna(col.median()))
df[categorical_cols] = df[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]))

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Normalize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display the first few rows of the cleaned dataset
print("Cleaned Dataset Sample:")
print(df.head())
