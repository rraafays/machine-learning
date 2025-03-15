#!/usr/bin/env python3

# Step 1: Load, Clean & Normalize the Data

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Load the dataset
file_path = "./cwdata.csv"  # Provided dataset
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print(df.head())

# Step 1: Data Cleaning

# Drop irrelevant or unnamed columns
df = df.drop(columns=["Unnamed: 10"], errors="ignore")

# Standardize categorical values (making them consistent)
df["Employed"] = df["Employed"].str.strip().str.lower().map({"y": 1, "yes": 1, "n": 0, "no": 0, np.nan: 0})
df["Home Owner"] = df["Home Owner"].str.strip().str.lower().map({"y": 1, "yes": 1, "n": 0, "no": 0, np.nan: 0})
df["Fraud"] = df["Fraud"].str.strip().str.lower().map({"y": 1, "yes": 1, "n": 0, "no": 0, "1": 1, "0": 0, np.nan: 0})

# Encode categorical features (Gender, Area, Education, Colour)
label_encoders = {}
for col in ["Gender", "Area", "Education", "Colour"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Save encoder for reference

# Handle missing or non-numeric values before scaling
df[["Income", "Balance", "Age"]] = df[["Income", "Balance", "Age"]].replace(" ", np.nan).astype(float)

# Normalize numerical values (Income, Balance, Age)
scaler = MinMaxScaler()
df[["Income", "Balance", "Age"]] = scaler.fit_transform(df[["Income", "Balance", "Age"]])

# Show the cleaned dataset
print(df.head())

# Ensure numerical columns have proper numeric values
df["Income"] = pd.to_numeric(df["Income"], errors='coerce')  # Convert errors to NaN
df["Balance"] = pd.to_numeric(df["Balance"], errors='coerce')
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')

# Fill missing numerical values with the median (better than mean for robustness)
df["Income"] = df["Income"].fillna(df["Income"].median())
df["Balance"] = df["Balance"].fillna(df["Balance"].median())
df["Age"] = df["Age"].fillna(df["Age"].median())

# Normalize numerical values after fixing them
scaler = MinMaxScaler()
df[["Income", "Balance", "Age"]] = scaler.fit_transform(df[["Income", "Balance", "Age"]])

# Show cleaned dataset
df.head()


import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Data Visualization & Feature Analysis

# Distribution of numerical features (Income, Balance, Age)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df["Income"], bins=30, ax=axes[0], kde=True).set(title="Income Distribution")
sns.histplot(df["Balance"], bins=30, ax=axes[1], kde=True).set(title="Balance Distribution")
sns.histplot(df["Age"], bins=30, ax=axes[2], kde=True).set(title="Age Distribution")

plt.show()

# Box plots for outliers in numerical features
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(y=df["Income"], ax=axes[0]).set(title="Income Box Plot")
sns.boxplot(y=df["Balance"], ax=axes[1]).set(title="Balance Box Plot")
sns.boxplot(y=df["Age"], ax=axes[2]).set(title="Age Box Plot")

plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
