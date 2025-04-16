#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import platform
import seaborn as sns
import time

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    LabelEncoder,
    StandardScaler
)

from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset
)

# ----------------------------------------------------------------------------
# NEURAL NETWORK MODEL DEFINITION
# ----------------------------------------------------------------------------
# This neural network uses a simple but effective approach for classification:
# - Multiple dense layers with progressively decreasing sizes to extract
#   hierarchical features
# - BatchNorm layers to stabilize and accelerate training by normalizing
#   activations
# - ReLU activations to introduce non-linearity without vanishing gradient
#   issues
# - Dropout for regularization to prevent overfitting
# - Kaiming initialization to set appropriate initial weights for ReLU
#   activations
#
# Alternatives considered:
# 1. More complex architectures (e.g., residual connections) - avoided for
#    simplicity and to prevent overfitting on this relatively small dataset
# 2. Different activation functions (e.g., LeakyReLU, SELU) - standard ReLU
#    chosen for its effectiveness and computational efficiency
# 3. No hidden layers - would limit the model's ability to learn complex
#    patterns
# 4. More hidden layers - could lead to overfitting and diminishing returns
# 5. Different weight initialization schemes - K aiming chosen as it's optimal
#    for ReLU networks


class FraudClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.3):
        super(FraudClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

# ----------------------------------------------------------------------------
# CUSTOM DATASET CLASS
# ----------------------------------------------------------------------------
# This dataset class provides a clean interface for PyTorch to access the data.
# It converts data to tensors and handles the optional target variable for
# unsupervised scenarios.
#
# Benefits of custom Dataset class:
# 1. Enables efficient batch processing through DataLoader
# 2. Abstracts data access logic from model training code
# 3. Handles both supervised (labeled) and unsupervised (unlabeled)
#    use cases
# 4. Ensures proper tensor formatting required by PyTorch
#
# Alternatives considered:
# 1. Using TensorDataset directly - less flexible for customization
# 2. Using NumPy arrays directly - would require conversion at each batch
# 3. Using pandas DataFrames directly - inefficient for deep learning


class FraudDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# ----------------------------------------------------------------------------
# DEVICE DETECTION AND SETUP
# ----------------------------------------------------------------------------
# This function automatically detects and configures the optimal computational
# device
# for model training based on the user's hardware (Apple Silicon, NVIDIA GPU,
# or CPU).
#
# Benefits:
# 1. Cross-platform compatibility for maximum performance
# 2. Hardware-specific optimizations (MPS for Apple, CUDA for NVIDIA)
# 3. Graceful fallback to CPU when no GPU is available
# 4. Enables acceleration without code changes when hardware changes
#
# Alternatives considered:
# 1. Hardcoded device selection - would limit portability
# 2. Manual device selection via arguments - adds complexity for users
# 3. CPU-only implementation - would be significantly slower
# 4. No explicit detection - would miss optimizations on Apple Silicon


def setup_device():
    system = platform.system()
    print(f"Detected operating system: {system}")

    if (system == "Darwin" and
            hasattr(torch.backends, 'mps') and
            torch.backends.mps.is_available()):
        print("Apple Silicon detected - using MPS backend")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return torch.device("mps"), "apple_silicon"

    elif torch.cuda.is_available():
        cuda_device_count = torch.cuda.device_count()
        cuda_device_name = torch.cuda.get_device_name(0)
        print(
            f"NVIDIA GPU detected - using CUDA "
            f"({cuda_device_name}, {cuda_device_count} device(s))")
        return torch.device("cuda"), "nvidia"

    else:
        print("No GPU acceleration available - using CPU")
        return torch.device("cpu"), "cpu"

# ----------------------------------------------------------------------------
# DATA CLEANING AND PREPROCESSING
# ----------------------------------------------------------------------------
# This function performs essential data preprocessing steps including:
# - Handling missing values through imputation
# - Converting categorical variables to numerical using encoding
# - Standardizing variable formats and ranges
#
# Benefits:
# 1. Ensures data quality for model training through systematic cleanup
# 2. Uses domain-appropriate handling for different data types
# 3. Provides consistent numerical representation for ML algorithms
# 4. Returns encoders for later use in inference/prediction
#
# Alternatives considered:
# 1. More complex imputation (e.g., KNN, iterative) - simpler median imputation
#    chosen for balance of effectiveness and efficiency
# 2. One-hot encoding - label encoding chosen for dimensionality control
# 3. Dropping missing values - would reduce dataset size unnecessarily
# 4. Feature engineering with interaction terms - avoided for initial modeling
# 5. Non-parametric transformations - not needed for this dataset's
#    distributions


def clean_data(df):
    print("Cleaning and preprocessing the data...")

    df = df.drop(columns=["Unnamed: 10"], errors="ignore")

    print("Converting binary fields to numerical values...")
    df["Employed"] = (df["Employed"].astype(str).str.strip().str.lower()
                      .map({"y": 1, "yes": 1, "n": 0, "no": 0, "1": 1,
                            "0": 0, "": 0, "nan": 0}))
    df["Home Owner"] = (df["Home Owner"].astype(str).str.strip().str.lower()
                        .map({"y": 1, "yes": 1, "n": 0, "no": 0, "1": 1,
                              "0": 0, "": 0, "nan": 0}))
    df["Fraud"] = (df["Fraud"].astype(str).str.strip().str.lower()
                   .map({"y": 1, "yes": 1, "n": 0, "no": 0, "1": 1,
                         "0": 0, "": 0, "nan": 0}))

    print("Converting numerical columns...")
    df["Income"] = pd.to_numeric(df["Income"], errors='coerce')
    df["Balance"] = pd.to_numeric(df["Balance"], errors='coerce')
    df["Age"] = pd.to_numeric(df["Age"], errors='coerce')

    print("Handling missing values...")
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Balance"] = df["Balance"].fillna(df["Balance"].median())
    df["Age"] = df["Age"].fillna(df["Age"].median())

    print("Encoding categorical variables...")
    label_encoders = {}
    for col in ["Gender", "Area", "Education", "Colour"]:
        le = LabelEncoder()
        df[col] = df[col].astype(str).str.strip()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

# ----------------------------------------------------------------------------
# FEATURE NORMALIZATION
# ----------------------------------------------------------------------------
# Applies MinMaxScaler to numerical features to normalize them to [0,1] range.
#
# Benefits:
# 1. Puts all numerical features on the same scale to prevent dominance
# 2. Improves convergence speed for gradient-based optimization
# 3. Necessary for neural networks to work effectively
# 4. Preserves the shape of the original distribution
#
# Alternatives considered:
# 1. StandardScaler (mean=0, std=1) - MinMaxScaler preferred for bounded [0,1]
#    output
# 2. RobustScaler - less sensitive to outliers but not necessary here
# 3. No scaling - would harm model performance, especially neural networks
# 4. Log transformation - not needed as distributions are not heavily skewed


def normalize_features(df):
    print("Normalizing numerical features...")
    scaler = MinMaxScaler()
    df[["Income", "Balance", "Age"]] = scaler.fit_transform(
        df[["Income", "Balance", "Age"]])
    return df, scaler

# ----------------------------------------------------------------------------
# DATA VISUALIZATION: DISTRIBUTIONS
# ----------------------------------------------------------------------------
# Creates histograms of the main numerical features to understand their
# distributions.
#
# Benefits:
# 1. Provides visual insight into data characteristics and distributions
# 2. Helps identify potential issues (outliers, skewness) before modeling
# 3. Supports feature engineering decisions
# 4. Creates documentation artifacts for assessment and presentation
#
# Alternatives considered:
# 1. Box plots - histograms chosen for better distribution visualization
# 2. Violin plots - more complex and less intuitive for basic distribution
#    analysis
# 3. Interactive visualizations - static plots chosen for simplicity and
#    compatibility
# 4. No visualization - would miss important data insights


def plot_distributions(df, output_dir):
    print("Generating distribution plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df["Income"], bins=30, ax=axes[0],
                 kde=True).set(title="Income Distribution")
    sns.histplot(df["Balance"], bins=30, ax=axes[1],
                 kde=True).set(title="Balance Distribution")
    sns.histplot(df["Age"], bins=30, ax=axes[2],
                 kde=True).set(title="Age Distribution")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "feature_distributions.png")
    plt.savefig(output_path, dpi=300)
    print(f"Feature distributions saved to {output_path}")

# ----------------------------------------------------------------------------
# DATA VISUALIZATION: CORRELATION MATRIX
# ----------------------------------------------------------------------------
# Generates a heatmap of feature correlations to understand relationships
# between variables.
#
# Benefits:
# 1. Identifies multicollinearity between features that could affect model
#    performance
# 2. Reveals potential predictive relationships with the target variable
# 3. Guides feature selection decisions
# 4. Helps understand underlying data structure
#
# Alternatives considered:
# 1. Pairwise scatter plots - too many for large feature sets
# 2. Point-biserial correlation for categorical vars - simpler approach chosen
# 3. VIF analysis - correlation matrix provides similar insights with better
#    visualization
# 4. No correlation analysis - would miss important feature relationships


def plot_correlation_matrix(df, output_dir):
    print("Generating correlation matrix...")

    plt.figure(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title("Feature Correlation Matrix")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(output_path, dpi=300)
    print(f"Correlation matrix saved to {output_path}")

# ----------------------------------------------------------------------------
# NEURAL NETWORK TRAINING LOOP
# ----------------------------------------------------------------------------
# Single epoch training function for the neural network model with progress
# tracking.
#
# Benefits:
# 1. Encapsulates the training loop for cleaner code organization
# 2. Provides real-time progress tracking and metrics with tqdm
# 3. Calculates and returns key performance metrics
# 4. Handles device-agnostic training (CPU/GPU/MPS)
#
# Alternatives considered:
# 1. Using higher-level training frameworks (e.g., PyTorch Lightning) - custom
#    implementation provides more control and educational value
# 2. Integrating validation within training loop - separated for cleaner code
# 3. Custom training schedule - simple epoch-based approach chosen for clarity
# 4. Gradient accumulation - not needed for this dataset size


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        current_acc = 100 * correct / total
        pbar.set_postfix({"loss": f"{running_loss/(pbar.n+1):.4f}",
                          "acc": f"{current_acc:.2f}%"})

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# ----------------------------------------------------------------------------
# NEURAL NETWORK VALIDATION
# ----------------------------------------------------------------------------
# Evaluates model performance on validation data without updating weights.
#
# Benefits:
# 1. Provides unbiased evaluation of model performance during training
# 2. Collects predictions for detailed performance analysis
# 3. Uses torch.no_grad() for memory efficiency during inference
# 4. Calculates metrics consistent with the training loop
#
# Alternatives considered:
# 1. Simplified metrics - comprehensive metrics chosen for better evaluation
# 2. Separate evaluation script - integrated for streamlined workflow
# 3. Only final evaluation - progressive evaluation provides training insights
# 4. Different threshold values - 0.5 is standard for binary classification


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_acc = 100 * correct / total
            pbar.set_postfix(
                {
                    "loss": f"{running_loss/(pbar.n+1):.4f}",
                    "acc": f"{current_acc:.2f}%",
                })

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy, all_preds, all_labels

# ----------------------------------------------------------------------------
# TRADITIONAL ML MODEL TRAINING (RANDOM FOREST)
# ----------------------------------------------------------------------------
# Trains a Random Forest classifier and evaluates its performance.
#
# Benefits:
# 1. Provides a strong baseline model with different strengths than neural
#    networks
# 2. Random Forests handle non-linear relationships well without feature
#    scaling
# 3. Built-in feature importance analysis for model interpretability
# 4. Typically requires less hyperparameter tuning than neural networks
# 5. Robust to overfitting through ensemble approach
#
# Alternatives considered:
# 1. Logistic Regression - too simple for potentially complex relationships
# 2. SVM - less interpretable and generally slower for large datasets
# 3. Gradient Boosting (XGBoost, LightGBM) - Random Forest chosen for balance
#    of performance and simplicity
# 4. Decision Tree - Random Forest provides better generalization
# 5. Ensemble of different models - avoided for simplicity


def train_sklearn_models(X_train, X_test, y_train, y_test, output_dir):
    print("Training Random Forest Classifier...")

    rf_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }

    clf = RandomForestClassifier(**rf_params)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    feature_imp = pd.Series(clf.feature_importances_,
                            index=X_train.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_imp.values, y=feature_imp.index)
    plt.title('Feature Importance in Random Forest Model')
    plt.tight_layout()
    output_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(output_path, dpi=300)
    print(f"Feature importance plot saved to {output_path}")

    return clf, accuracy, report

# ----------------------------------------------------------------------------
# NEURAL NETWORK TRAINING PIPELINE
# ----------------------------------------------------------------------------
# Complete training pipeline for the neural network model with early stopping.
#
# Benefits:
# 1. Comprehensive pipeline handling data preparation, training, validation,
#    and evaluation
# 2. Early stopping to prevent overfitting
# 3. Automatic learning rate adjustment with scheduler
# 4. Hardware-specific optimizations for different platforms
# 5. Progress tracking and visualization
#
# Alternatives considered:
# 1. Fixed epochs without early stopping - less efficient and could overfit
# 2. Different optimizers (SGD, RMSprop) - Adam chosen for adaptive learning
#    rate
# 3. Cross-validation - single validation split chosen for efficiency
# 4. Different loss functions - BCE appropriate for binary classification
# 5. Fixed learning rate - scheduler allows adaptive rate based on performance


def train_neural_model(X_train, X_test,
                       y_train, y_test,
                       device, platform_type, output_dir):
    print("\nTraining Neural Network Model...")

    torch.manual_seed(42)
    np.random.seed(42)

    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    if platform_type == "apple_silicon":
        batch_size = 64
        num_workers = 0
    elif platform_type == "nvidia":
        batch_size = 32
        num_workers = min(8, multiprocessing.cpu_count())
    else:
        batch_size = 16
        num_workers = min(4, multiprocessing.cpu_count())

    print(f"Using batch size: {batch_size}, worker processes: {num_workers}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(platform_type != "cpu")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(platform_type != "cpu")
    )

    input_dim = X_train.shape[1]
    model = FraudClassifier(input_dim).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5)

    epochs = 30
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc, _, _ = validate(
            model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        print(
            f"Time: {epoch_time:.1f}s - "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% - "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = os.path.join(output_dir, "best_fraud_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model with validation accuracy: {
                  val_acc:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")

    model_path = os.path.join(output_dir, "best_fraud_model.pth")
    model.load_state_dict(torch.load(model_path))

    print("Performing final evaluation...")
    _, final_acc, predictions, actual_labels = validate(
        model, test_loader, criterion, device)
    print(f"Final Validation Accuracy: {final_acc:.2f}%")

    plot_training_results(train_losses, val_losses,
                          train_accs, val_accs, predictions,
                          actual_labels, output_dir)

    return model, final_acc

# ----------------------------------------------------------------------------
# UNSUPERVISED LEARNING: K-MEANS CLUSTERING
# ----------------------------------------------------------------------------
# Performs K-means clustering for unsupervised pattern discovery in the data.
#
# Benefits:
# 1. Discovers natural groupings in data without relying on labels
# 2. Uses silhouette score to determine optimal number of clusters objectively
# 3. Applies standard scaling for distance-based algorithm
# 4. Visualizes clusters using PCA for dimensionality reduction
#
# Alternatives considered:
# 1. Hierarchical Clustering - K-means chosen for scalability with larger
#    datasets
# 2. DBSCAN - K-means chosen for simplicity and interpretability
# 3. Gaussian Mixture Models - K-means provides clearer cluster boundaries
# 4. Spectral Clustering - K-means more efficient for this dataset size
# 5. Self-organizing maps - K-means more widely understood and interpretable


def perform_clustering(X, output_dir):
    print("\nPerforming K-means clustering...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = []
    cluster_range = range(2, 6)

    for n_clusters in tqdm(cluster_range, desc="Finding optimal clusters"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Silhouette score for {
              n_clusters} clusters: {silhouette_avg:.4f}")

    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_clusters}")

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                          cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title(
        f'K-means Clustering (k={optimal_clusters}) with PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    output_path = os.path.join(output_dir, "clustering_results.png")
    plt.savefig(output_path, dpi=300)
    print(f"Clustering visualization saved to {output_path}")

    return cluster_labels

# ----------------------------------------------------------------------------
# VISUALIZATION OF MODEL RESULTS
# ----------------------------------------------------------------------------
# Comprehensive visualization of model training and evaluation metrics.
#
# Benefits:
# 1. Provides visual evaluation of model convergence and performance
# 2. Includes multiple complementary evaluation metrics (loss, accuracy,
#    confusion matrix, ROC)
# 3. Creates professional-quality visualizations for documentation
# 4. Helps identify potential issues (overfitting, poor convergence)
#
# Alternatives considered:
# 1. Interactive visualizations (Plotly) - static matplotlib chosen for
#    compatibility
# 2. Separate plots - combined subplot layout for comprehensive overview
# 3. Minimal metrics - comprehensive metrics chosen for thorough evaluation
# 4. Different visualization styles - seaborn/matplotlib provide clean,
#    publication-ready plots


def plot_training_results(train_losses, val_losses, train_accs,
                          val_accs, predictions,
                          actual_labels, output_dir):
    print("Generating training plots...")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1),
             val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")

    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, label="Train Accuracy")
    plt.plot(range(1, len(val_accs) + 1),
             val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training vs Validation Accuracy")

    plt.subplot(2, 2, 3)
    cm = confusion_matrix(actual_labels, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.subplot(2, 2, 4)
    fpr, tpr, _ = roc_curve(actual_labels, predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "training_results.png")
    plt.savefig(output_path, dpi=300)
    print(f"Training results saved to {output_path}")

# ----------------------------------------------------------------------------
# COMMAND LINE ARGUMENT PARSING
# ----------------------------------------------------------------------------
# Robust command-line interface for both training and prediction modes.
#
# Benefits:
# 1. Provides a user-friendly interface for both training and prediction
# 2. Clearly documents required parameters with help text
# 3. Separates prediction options into a logical group
# 4. Includes usage examples for better usability
#
# Alternatives considered:
# 1. Configuration file - CLI more appropriate for simple parameters
# 2. Interactive prompts - CLI better for automation and scripting
# 3. Environment variables - CLI provides better documentation and validation
# 4. Web interface - CLI simpler for this use case


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Fraud Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train model
  ./fraud_detection.py

  # Make a prediction using an existing model
  ./fraud_detection.py --predict \\
                          --income 50000 \\
                          --balance 10000 \\
                          --age 35 \\
                          --gender M \\
                          --area "Urban" \\
                          --education "University" \\
                          --colour "Blue" \\
                          --employed Y \\
                          --homeowner N
        ''')

    parser.add_argument('--predict', action='store_true',
                        help='Run in prediction mode using trained model')

    pred_group = parser.add_argument_group('Prediction options')
    pred_group.add_argument('--income', type=float,
                            help='Income value (numeric, e.g. 50000)')
    pred_group.add_argument('--balance', type=float,
                            help='Account balance (numeric, e.g. 10000)')
    pred_group.add_argument('--age', type=int,
                            help='Age (integer, e.g. 35)')
    pred_group.add_argument('--gender', type=str,
                            help='Gender (M/F)')
    pred_group.add_argument('--area', type=str,
                            help='Area/Region (e.g. Urban, Rural, Suburban)')
    pred_group.add_argument('--education', type=str,
                            help='Education level (e.g. University, School)')
    pred_group.add_argument('--colour', type=str,
                            help='Color preference (e.g. Blue, Red, Green)')
    pred_group.add_argument('--employed', type=str,
                            help='Employment status (Y/N)')
    pred_group.add_argument('--homeowner', type=str,
                            help='Home ownership status (Y/N)')

    return parser.parse_args()

# ----------------------------------------------------------------------------
# PREDICTION FUNCTION
# ----------------------------------------------------------------------------
# Applies the trained model to make predictions on new data.
#
# Benefits:
# 1. Provides a clean interface for real-world fraud prediction
# 2. Handles all necessary preprocessing of input data
# 3. Returns both binary prediction and confidence score
# 4. Uses the same preprocessing steps as training for consistency
#
# Alternatives considered:
# 1. Separate preprocessing - integrated for consistency
# 2. Different threshold values - 0.5 standard for binary classification
# 3. Returning only binary prediction - confidence score adds valuable
#    information
# 4. Batch prediction - single-instance focus for this interactive use case


def predict_fraud(args, model_path, label_encoders, scaler, device):
    input_dim = 9
    model = FraudClassifier(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data = {
        'Income': [args.income],
        'Balance': [args.balance],
        'Age': [args.age],
        'Gender': [args.gender],
        'Area': [args.area],
        'Education': [args.education],
        'Colour': [args.colour],
        'Employed': [1 if args.employed.lower() in ['y', 'yes', '1'] else 0],
        'Home Owner': [1 if args.homeowner.lower() in ['y', 'yes', '1'] else 0]
    }

    input_df = pd.DataFrame(data)

    for col, encoder in label_encoders.items():
        input_df[col] = encoder.transform(
            input_df[col].astype(str).str.strip())

    input_df[["Income", "Balance", "Age"]] = scaler.transform(
        input_df[["Income", "Balance", "Age"]])

    input_tensor = torch.FloatTensor(input_df.values).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    probability = output.item() * 100
    is_fraud = output.item() > 0.5

    return is_fraud, probability

# ----------------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------------
# Orchestrates the overall workflow of the fraud detection system.
#
# Benefits:
# 1. Separates training and prediction workflows for clarity
# 2. Implements a complete ML pipeline from data loading to evaluation
# 3. Handles both supervised (fraud detection) and unsupervised (clustering)
#    learning
# 4. Creates comprehensive output directory with all results and visualizations
#
# Alternatives considered:
# 1. Separate scripts for different tasks - integrated for workflow continuity
# 2. Class-based architecture - procedural approach chosen for simplicity
# 3. Pipeline framework (e.g., scikit-learn Pipeline) - custom implementation
#    for flexibility
# 4. Database integration - file-based approach simpler for this use case


def main():
    args = parse_arguments()

    torch.manual_seed(42)
    np.random.seed(42)

    output_dir = os.path.join("results", "fraud")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    device, platform_type = setup_device()

    if args.predict:
        required_args = ['income', 'balance', 'age', 'gender', 'area',
                         'education', 'colour', 'employed', 'homeowner']
        missing_args = [
            arg for arg in required_args if getattr(args, arg) is None]

        if missing_args:
            print(f"Error: Missing required arguments: {
                  ', '.join(missing_args)}")
            return

        print("\nLoading dataset for encoders and scalers...")
        file_path = "./cwdata.csv"
        df = pd.read_csv(file_path)
        df, label_encoders = clean_data(df)
        df, scaler = normalize_features(df)

        model_path = os.path.join(output_dir, "best_fraud_model.pth")
        if not os.path.exists(model_path):
            print(f"Error: Model file {
                  model_path} not found. Train the model first.")
            return

        print("\nMaking fraud prediction...")
        is_fraud, probability = predict_fraud(
            args, model_path, label_encoders, scaler, device)

        print("\n=== Fraud Prediction Results ===")
        print(f"Fraud detected: {'Yes' if is_fraud else 'No'}")
        if is_fraud:
            print(f"Confidence: {probability:.2f}%")
        else:
            print(f"Confidence: {100-probability:.2f}%")

    else:
        if platform_type == "nvidia":
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        print("\nLoading dataset...")
        file_path = "./cwdata.csv"
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {
              df.shape[0]} rows and {df.shape[1]} columns")
        print("First few rows of the dataset:")
        print(df.head())

        df, label_encoders = clean_data(df)
        df, scaler = normalize_features(df)

        print("\nProcessed dataset:")
        print(df.head())

        plot_distributions(df, output_dir)
        plot_correlation_matrix(df, output_dir)

        print("\nPreparing data for modeling...")
        X = df.drop(columns=["Fraud"])
        y = df["Fraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")

        rf_model, rf_accuracy, rf_report = train_sklearn_models(
            X_train, X_test, y_train, y_test, output_dir)

        nn_model, nn_accuracy = train_neural_model(
            X_train, X_test, y_train, y_test,
            device, platform_type, output_dir)

        cluster_labels = perform_clustering(X, output_dir)
        df['Cluster'] = cluster_labels

        print("\n=== Final Results ===")
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        print(f"Neural Network Accuracy: {nn_accuracy:.2f}%")
        print("\nTraining visualizations saved to:")
        print(f" {os.path.join(output_dir, 'feature_distributions.png')}")
        print(f" {os.path.join(output_dir, 'correlation_matrix.png')}")
        print(f" {os.path.join(output_dir, 'feature_importance.png')}")
        print(f" {os.path.join(output_dir, 'training_results.png')}")
        print(f" {os.path.join(output_dir, 'clustering_results.png')}")
        print(f"\nBest model saved to {os.path.join(
            output_dir, 'best_fraud_model.pth')}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
