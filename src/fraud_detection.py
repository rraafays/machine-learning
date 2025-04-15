#!/usr/bin/env python3

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    LabelEncoder
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error
)

file_path = "./cwdata.csv"
df = pd.read_csv(file_path)

print(df.head())

df = df.drop(columns=["Unnamed: 10"], errors="ignore")

df["Employed"] = df["Employed"].str.strip().str.lower().map(
    {"y": 1, "yes": 1, "n": 0, "no": 0, np.nan: 0})
df["Home Owner"] = df["Home Owner"].str.strip().str.lower().map(
    {"y": 1, "yes": 1, "n": 0, "no": 0, np.nan: 0})
df["Fraud"] = df["Fraud"].str.strip().str.lower().map(
    {"y": 1, "yes": 1, "n": 0, "no": 0, "1": 1, "0": 0, np.nan: 0})

label_encoders = {}
for col in ["Gender", "Area", "Education", "Colour"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

df[["Income", "Balance", "Age"]] = df[["Income", "Balance", "Age"]
                                      ].replace(" ", np.nan).astype(float)

scaler = MinMaxScaler()
df[["Income", "Balance", "Age"]] = scaler.fit_transform(
    df[["Income", "Balance", "Age"]])

print(df.head())

df["Income"] = pd.to_numeric(
    df["Income"], errors='coerce')
df["Balance"] = pd.to_numeric(df["Balance"], errors='coerce')
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')

df["Income"] = df["Income"].fillna(df["Income"].median())
df["Balance"] = df["Balance"].fillna(df["Balance"].median())
df["Age"] = df["Age"].fillna(df["Age"].median())

scaler = MinMaxScaler()
df[["Income", "Balance", "Age"]] = scaler.fit_transform(
    df[["Income", "Balance", "Age"]])

print(df.head())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df["Income"], bins=30, ax=axes[0],
             kde=True).set(title="Income Distribution")
sns.histplot(df["Balance"], bins=30, ax=axes[1],
             kde=True).set(title="Balance Distribution")
sns.histplot(df["Age"], bins=30, ax=axes[2],
             kde=True).set(title="Age Distribution")

plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(y=df["Income"], ax=axes[0]).set(title="Income Box Plot")
sns.boxplot(y=df["Balance"], ax=axes[1]).set(title="Balance Box Plot")
sns.boxplot(y=df["Age"], ax=axes[2]).set(title="Age Box Plot")

plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

X_unsupervised = df.drop(columns=["Fraud"])
y_classification = df["Fraud"]

X_train, X_test, y_train_class, y_test_class = train_test_split(
    X_unsupervised, y_classification, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train_class)

y_pred_class = clf.predict(X_test)

classification_acc = accuracy_score(y_test_class, y_pred_class)
classification_report_text = classification_report(y_test_class, y_pred_class)

y_regression = df["Balance"]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_unsupervised, y_regression, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_reg, y_train_reg)

y_pred_reg = regressor.predict(X_test_reg)

regression_mse = mean_squared_error(y_test_reg, y_pred_reg)

print(classification_acc, classification_report_text, regression_mse)

imputer = SimpleImputer(strategy='mean')
X_unsupervised_imputed = imputer.fit_transform(X_unsupervised)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_unsupervised_imputed)

silhouette_avg = silhouette_score(X_unsupervised_imputed, df["Cluster"])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_unsupervised_imputed)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=df["Cluster"], palette="coolwarm", alpha=0.7)
plt.title("K-Means Clustering Visualization (PCA Reduced)")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.show()

silhouette_avg, df["Cluster"].value_counts()
