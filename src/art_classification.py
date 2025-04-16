#!/usr/bin/env python3

import argparse
import kagglehub
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import platform
import seaborn as sns
import time
import torchvision

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

# ----------------------------------------------------------------------------
# CUSTOM IMAGE FOLDER FOR ROBUST DATASET HANDLING
# ----------------------------------------------------------------------------
# A custom implementation of torchvision's ImageFolder that gracefully handles
# corrupted or problematic images that might otherwise crash training.
#
# Benefits:
# 1. Prevents training failures due to corrupted image files
# 2. Handles images with transparency layers automatically
# 3. Improves training robustness by skipping problematic files
# 4. Maintains dataset integrity without preprocessing steps
#
# Alternatives considered:
# 1. Preprocessing dataset to remove problematic images - adds complexity and
#    still might miss some issues
# 2. Using standard ImageFolder with try-except in training - less efficient
#    and harder to debug
# 3. Using OpenCV instead of PIL - adds additional dependencies and doesn't
#    handle some image formats as well
# 4. Creating separate datasets for each image type - unnecessarily complex
#    and harder to maintain


class SafeImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            image = self.loader(path)
            if image.mode == 'P' and 'transparency' in image.info:
                image = image.convert('RGBA')
            elif image.mode == 'L' and 'transparency' in image.info:
                image = image.convert('RGBA')
            if self.transform is not None:
                image = self.transform(image)
            return image, target
        except (IOError, OSError, RuntimeError):
            return self.__getitem__((index + 1) % len(self.samples))

# ----------------------------------------------------------------------------
# CONVOLUTIONAL NEURAL NETWORK FOR ART CLASSIFICATION
# ----------------------------------------------------------------------------
# A custom CNN architecture specifically designed for art classification with
# optimized depth, width, and regularization techniques for artistic features.
#
# Benefits:
# 1. Progressive filter expansion (32→64→128→256) captures increasingly complex
#    artistic features and patterns
# 2. Double convolution blocks with shared filter sizes enhance feature
#    extraction
# 3. BatchNorm layers stabilize training and improve convergence
# 4. Strategic dropout prevents overfitting to specific artistic styles
# 5. Proper weight initialization ensures stable gradient flow
#
# Alternatives considered:
# 1. Pre-trained models (ResNet, VGG) - not specialized for artistic features
#    and would require significant fine-tuning
# 2. Deeper architectures - could lead to overfitting with limited art datasets
# 3. Attention mechanisms - would add complexity with uncertain benefits for
#    this task
# 4. MobileNet-style separable convolutions - would reduce parameters but might
#    miss important textural details in artwork
# 5. Vision Transformers - require significantly more training data and
#    computational resources than available


class OptimizedArtCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(OptimizedArtCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        feature_size = 256 * (224 // 16) * (224 // 16)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ----------------------------------------------------------------------------
# TRAINING FUNCTION WITH MIXED PRECISION AND GRADIENT ACCUMULATION
# ----------------------------------------------------------------------------
# Optimized training function for a single epoch that supports hardware
# acceleration techniques like mixed precision and effective batch size
# expansion through gradient accumulation.
#
# Benefits:
# 1. Mixed precision training dramatically accelerates computation on
#    supported hardware
# 2. Gradient accumulation enables effective training with larger batch sizes
#    without increased memory requirements
# 3. Real-time progress tracking with performance metrics improves monitoring
# 4. Platform-agnostic implementation works efficiently across different
#    compute devices
#
# Alternatives considered:
# 1. Standard training without acceleration techniques - simpler but
#    significantly slower
# 2. Distributed training across multiple GPUs - more complex and not necessary
#    for this scale
# 3. Using higher-level frameworks like PyTorch Lightning - less control over
#    implementation details
# 4. Custom loss scaling without autocast - more prone to numerical instability
# 5. Fixed batch sizes without accumulation - would limit effective batch size
#    on memory-constrained devices


def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None,
                grad_accum_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / grad_accum_steps

            loss.backward()

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        running_loss += loss.item() * grad_accum_steps
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        current_acc = 100 * correct / total
        pbar.set_postfix({"loss": f"{running_loss/(i+1):.4f}",
                         "acc": f"{current_acc:.2f}%"})

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# ----------------------------------------------------------------------------
# MODEL VALIDATION AND EVALUATION
# ----------------------------------------------------------------------------
# Comprehensive validation function that evaluates model performance on
# unseen data and collects metrics for detailed analysis without affecting
# weights.
#
# Benefits:
# 1. Efficient evaluation using torch.no_grad() to save memory and computation
# 2. Collects comprehensive metrics including loss, accuracy, and predictions
# 3. Real-time progress tracking with performance indicators
# 4. Returns predictions for confusion matrix and further analysis
#
# Alternatives considered:
# 1. Simplified evaluation with only accuracy measurement - less informative
# 2. Including validation as part of training loop - reduces code clarity
# 3. Using additional metrics like F1-score or precision/recall
#    added unnecesary complexity for balanced datasets
# 4. Test-time augmentation - would improve results but increases validation
#    time
# 5. K-fold cross validation - more statistically robust but prohibitively
#    compute-intensive for large image datasets


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
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
# CROSS-PLATFORM DEVICE DETECTION AND SETUP
# ----------------------------------------------------------------------------
# Automatic detection and configuration of the optimal computing device based
# on the user's hardware, with support for latest accelerators.
#
# Benefits:
# 1. Cross-platform compatibility for maximum performance on any system
# 2. Explicit support for Apple Silicon MPS acceleration
# 3. Automatic CUDA detection for NVIDIA GPUs
# 4. Graceful fallback to CPU when no GPU is available
# 5. Provides clear feedback about the hardware being used
#
# Alternatives considered:
# 1. Hardcoded device selection - would limit portability across environments
# 2. Manual selection via command line arguments - adds complexity for users
# 3. Only supporting CUDA - would miss optimizations on Apple Silicon
# 4. Using higher-level abstractions like PyTorch Lightning - less control
#    over hardware-specific optimizations


def setup_device():
    """Set up the appropriate device based on the platform"""
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
# SINGLE IMAGE CLASSIFICATION AND VISUALIZATION
# ----------------------------------------------------------------------------
# Function to classify a single image using a trained model and visualize
# the results with confidence scores and probabilities.
#
# Benefits:
# 1. Handles various image formats including transparency channels
# 2. Provides confidence scores and top-k predictions for multi-class problems
# 3. Creates visual output of the classification for easy interpretation
# 4. Reports detailed probability distribution across classes
#
# Alternatives considered:
# 1. Command-line output only - less intuitive and user-friendly
# 2. Web service implementation - unnecessarily complex for simple
#    classification
# 3. Using raw logits instead of softmax probabilities - less interpretable
# 4. Batch classification - single image focus is more appropriate for
#    interactive use
# 5. Multiple model ensemble - would increase robustness but at significant
#    performance cost


def classify_image(image_path, model_path, device, classes):
    """Classify a single image using a trained model"""
    print(f"Classifying image: {image_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file {
              model_path} not found. Please train the model first.")
        return

    print("Loading model...")
    num_classes = len(classes)
    model = OptimizedArtCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path)

        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        elif image.mode == 'L' and 'transparency' in image.info:
            image = image.convert('RGBA')

        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = classes[predicted_idx.item()]
            confidence = probabilities[predicted_idx.item()].item() * 100

        print(f"\nPrediction: {
              predicted_class} (Confidence: {confidence:.2f}%)")

        if len(classes) > 2:
            top_k = min(3, len(classes))
            top_probs, top_idx = torch.topk(probabilities, top_k)
            print("\nTop predictions:")
            for i in range(top_k):
                print(f"  {classes[top_idx[i].item()]}: {
                      top_probs[i].item()*100:.2f}%")

        plt.figure(figsize=(6, 6))
        plt.imshow(np.array(image))
        plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing image: {str(e)}")

# ----------------------------------------------------------------------------
# MAIN FUNCTION: TRAINING AND EVALUATION PIPELINE
# ----------------------------------------------------------------------------
# Complete end-to-end pipeline for art classification including data
# preparation, model training, evaluation, and result visualization.
#
# Benefits:
# 1. Command-line interface with documentation for easy use
# 2. Automatic dataset downloading and management
# 3. Comprehensive training with early stopping to prevent overfitting
# 4. Platform-specific optimizations for maximum performance
# 5. Result visualization with training curves and confusion matrix
# 6. Model persistence for later use in classification
#
# Alternatives considered:
# 1. Configuration file based setup - more complex but less flexible
# 2. Splitting into multiple scripts - better separation of concerns but more
#    complex project structure
# 3. Web or GUI interface - more user-friendly but unnecessarily complex
# 4. Using an ML framework like PyTorch Lightning - more abstracted but less
#    control over implementation details
# 5. Fixed hyperparameters without early stopping - simpler but less effective


def main():
    parser = argparse.ArgumentParser(
        description='Art Classification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train model
  ./art_classification.py

  # Classify an image using an existing model
  ./art_classification.py --image ~/path/to/image
        ''')
    parser.add_argument('--image', type=str,
                        help='Path to an image for classification')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    results_dir = os.path.join(os.getcwd(), "results", "art")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to {results_dir}")

    device, platform_type = setup_device()

    model_path = os.path.join(results_dir, "best_art_model.pth")

    if args.image:
        path = kagglehub.dataset_download(
            "thedownhill/art-images-drawings-painting-sculpture-engraving")
        path = path + "/dataset/dataset_updated"
        train_set = os.path.join(path, "training_set")

        temp_dataset = torchvision.datasets.ImageFolder(root=train_set)
        classes = temp_dataset.classes

        classify_image(args.image, model_path, device, classes)
        return

    if platform_type == "nvidia":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    path = kagglehub.dataset_download(
        "thedownhill/art-images-drawings-painting-sculpture-engraving")
    path = path + "/dataset/dataset_updated"
    print("Path to dataset files:", path)

    train_set = os.path.join(path, "training_set")
    val_set = os.path.join(path, "validation_set")

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    print("Loading training dataset...")
    train_data = SafeImageFolder(root=train_set, transform=transform_train)
    print("Loading validation dataset...")
    val_data = SafeImageFolder(root=val_set, transform=transform_val)

    if platform_type == "apple_silicon":
        batch_size = 128
        num_workers = 0
    elif platform_type == "nvidia":
        batch_size = 64
        num_workers = min(8, multiprocessing.cpu_count())
    else:
        batch_size = 32
        num_workers = min(4, multiprocessing.cpu_count())

    print(f"Using batch size: {batch_size}, worker processes: {num_workers}")

    print("Creating data loaders...")
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(platform_type != "cpu"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(platform_type != "cpu")
    )

    print(f"Classes: {train_data.classes}")
    print(f"Training samples: {len(train_data)
                               }, Validation samples: {len(val_data)}")

    print("Initializing model...")
    num_classes = len(train_data.classes)
    model = OptimizedArtCNN(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler() if platform_type == "nvidia" else None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    effective_batch_size = 64
    if platform_type == "cpu":
        effective_batch_size = 32
    grad_accum_steps = max(1, effective_batch_size // batch_size)
    print(f"Using gradient accumulation with {grad_accum_steps} steps")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler, grad_accum_steps
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

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
            torch.save(model.state_dict(), os.path.join(
                results_dir, "best_art_model.pth"))
            print(f"Saved new best model with validation accuracy: {
                  val_acc:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")

    model.load_state_dict(torch.load(
        os.path.join(results_dir, "best_art_model.pth")))

    print("Performing final evaluation...")
    _, final_acc, predictions, actual_labels = validate(
        model, val_loader, criterion, device)
    print(f"Final Validation Accuracy: {final_acc:.2f}%")

    print("Generating training plots...")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1),
             val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")

    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, label="Train Accuracy")
    plt.plot(range(1, len(val_accs) + 1),
             val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training vs Validation Accuracy")

    plt.subplot(1, 3, 3)
    cm = confusion_matrix(actual_labels, predictions)

    if len(train_data.classes) > 10:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_norm, annot=False, cmap="Blues",
            xticklabels=train_data.classes, yticklabels=train_data.classes)
        plt.title("Normalized Confusion Matrix")
    else:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=train_data.classes, yticklabels=train_data.classes)
        plt.title("Confusion Matrix")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_results.png"), dpi=300)

    print(f"Training results saved to {
          os.path.join(results_dir, 'training_results.png')}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
