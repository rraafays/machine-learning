#!/usr/bin/env python3

import os
import time
import platform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
import multiprocessing
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import kagglehub


class SafeImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            image = self.loader(path)
            # More robust handling of palette images with transparency
            if image.mode == 'P' and 'transparency' in image.info:
                # Convert to RGBA only if transparency exists in palette image
                image = image.convert('RGBA')
            elif image.mode == 'L' and 'transparency' in image.info:
                # Handle grayscale images with transparency
                image = image.convert('RGBA')
            if self.transform is not None:
                image = self.transform(image)
            return image, target
        except (IOError, OSError, RuntimeError):
            # Skip problematic images with a fallback strategy
            return self.__getitem__((index + 1) % len(self.samples))


class OptimizedArtCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(OptimizedArtCNN, self).__init__()
        # Optimized feature extractor with fewer parameters but better design
        self.features = nn.Sequential(
            # First block - 32 filters, less parameters to start
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Second block - 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Third block - 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Fourth block - 256 filters
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Calculate feature size based on input size of 224x224
        feature_size = 256 * (224 // 16) * (224 // 16)

        # Classifier with proper regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # Initialize weights for better training
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


def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None,
                grad_accum_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Add progress bar for training
    pbar = tqdm(train_loader, desc="Training", leave=False)

    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Mixed precision training if scaler is provided (for CUDA)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            # Gradient accumulation
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / grad_accum_steps

            loss.backward()

            # Gradient accumulation
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        # Track statistics
        running_loss += loss.item() * grad_accum_steps
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar with current loss and accuracy
        current_acc = 100 * correct / total
        pbar.set_postfix({"loss": f"{running_loss/(i+1):.4f}",
                         "acc": f"{current_acc:.2f}%"})

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # Add progress bar for validation
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

            # Update progress bar with current loss and accuracy
            current_acc = 100 * correct / total
            pbar.set_postfix(
                {
                    "loss": f"{running_loss/(pbar.n+1):.4f}",
                    "acc": f"{current_acc:.2f}%",
                })

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy, all_preds, all_labels


def setup_device():
    """Set up the appropriate device based on the platform"""
    system = platform.system()
    print(f"Detected operating system: {system}")

    # Check for Apple Silicon (MPS)
    if (system == "Darwin" and
            hasattr(torch.backends, 'mps') and
            torch.backends.mps.is_available()):
        print("Apple Silicon detected - using MPS backend")
        # Enable MPS fallback for operations not yet supported in MPS
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return torch.device("mps"), "apple_silicon"

    # Check for NVIDIA GPU (CUDA)
    elif torch.cuda.is_available():
        cuda_device_count = torch.cuda.device_count()
        cuda_device_name = torch.cuda.get_device_name(0)
        print(
            f"NVIDIA GPU detected - using CUDA "
            f"({cuda_device_name}, {cuda_device_count} device(s))")
        return torch.device("cuda"), "nvidia"

    # Fallback to CPU
    else:
        print("No GPU acceleration available - using CPU")
        return torch.device("cpu"), "cpu"


def classify_image(image_path, model_path, device, classes):
    """Classify a single image using a trained model"""
    print(f"Classifying image: {image_path}")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {
              model_path} not found. Please train the model first.")
        return

    # Load the model
    print("Loading model...")
    num_classes = len(classes)
    model = OptimizedArtCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare image transforms - same as validation transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    try:
        # Open and transform the image
        image = Image.open(image_path)

        # Handle transparency similar to SafeImageFolder
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        elif image.mode == 'L' and 'transparency' in image.info:
            image = image.convert('RGBA')

        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        # Apply transforms and add batch dimension
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = classes[predicted_idx.item()]
            confidence = probabilities[predicted_idx.item()].item() * 100

        print(f"\nPrediction: {
              predicted_class} (Confidence: {confidence:.2f}%)")

        # Show top 3 predictions if there are enough classes
        if len(classes) > 2:
            top_k = min(3, len(classes))
            top_probs, top_idx = torch.topk(probabilities, top_k)
            print("\nTop predictions:")
            for i in range(top_k):
                print(f"  {classes[top_idx[i].item()]}: {
                      top_probs[i].item()*100:.2f}%")

        # Optionally display the image
        plt.figure(figsize=(6, 6))
        plt.imshow(np.array(image))
        plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing image: {str(e)}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Art Classification')
    parser.add_argument('--image', type=str,
                        help='Path to an image for classification')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.getcwd(), "results", "art")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to {results_dir}")

    # Configure device and platform specifics
    device, platform_type = setup_device()

    # Path to the model
    model_path = os.path.join(results_dir, "best_art_model.pth")

    # If image path is provided, classify the image
    if args.image:
        # Need to get the classes from the dataset
        path = kagglehub.dataset_download(
            "thedownhill/art-images-drawings-painting-sculpture-engraving")
        path = path + "/dataset/dataset_updated"
        train_set = os.path.join(path, "training_set")

        # We only need to get the class names without loading the full dataset
        temp_dataset = torchvision.datasets.ImageFolder(root=train_set)
        classes = temp_dataset.classes

        # Classify the image
        classify_image(args.image, model_path, device, classes)
        return

    # Otherwise, run the training process
    # Performance optimizations
    if platform_type == "nvidia":
        torch.backends.cudnn.deterministic = False  # For performance
        torch.backends.cudnn.benchmark = True       # For performance

    # Get dataset
    path = kagglehub.dataset_download(
        "thedownhill/art-images-drawings-painting-sculpture-engraving")
    path = path + "/dataset/dataset_updated"
    print("Path to dataset files:", path)

    train_set = os.path.join(path, "training_set")
    val_set = os.path.join(path, "validation_set")

    # Optimized data augmentation pipeline
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

    # Load datasets with optimized parameters
    print("Loading training dataset...")
    train_data = SafeImageFolder(root=train_set, transform=transform_train)
    print("Loading validation dataset...")
    val_data = SafeImageFolder(root=val_set, transform=transform_val)

    # Platform-specific optimizations
    if platform_type == "apple_silicon":
        # M1/M2/M3 optimizations: higher batch size due to unified memory
        batch_size = 128
        num_workers = 0
    elif platform_type == "nvidia":
        # NVIDIA optimizations
        batch_size = 64
        num_workers = min(8, multiprocessing.cpu_count())
    else:
        # CPU fallback
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
        batch_size=batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(platform_type != "cpu")
    )

    print(f"Classes: {train_data.classes}")
    print(f"Training samples: {len(train_data)
                               }, Validation samples: {len(val_data)}")

    # Initialize model
    print("Initializing model...")
    num_classes = len(train_data.classes)
    model = OptimizedArtCNN(num_classes).to(device)

    # Loss function with label smoothing for regularization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler with cosine annealing
    epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    # Mixed precision training setup (CUDA only)
    scaler = torch.cuda.amp.GradScaler() if platform_type == "nvidia" else None

    # Training setup
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    # Calculate gradient accumulation steps based on batch size and platform
    effective_batch_size = 64
    if platform_type == "cpu":
        # Aggressive gradient accumulation to compensate for smaller batches
        effective_batch_size = 32
    grad_accum_steps = max(1, effective_batch_size // batch_size)
    print(f"Using gradient accumulation with {grad_accum_steps} steps")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler, grad_accum_steps
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step()

        # Print statistics
        epoch_time = time.time() - epoch_start
        print(
            f"Time: {epoch_time:.1f}s - "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% - "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(
                results_dir, "best_art_model.pth"))
            print(f"Saved new best model with validation accuracy: {
                  val_acc:.2f}%")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")

    # Load best model for final evaluation
    model.load_state_dict(torch.load(
        os.path.join(results_dir, "best_art_model.pth")))

    # Final evaluation
    print("Performing final evaluation...")
    _, final_acc, predictions, actual_labels = validate(
        model, val_loader, criterion, device)
    print(f"Final Validation Accuracy: {final_acc:.2f}%")

    # Plot training curves
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

    # Confusion matrix
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(actual_labels, predictions)

    # Normalize confusion matrix for better visualization if many classes
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

    # Save the final model
    torch.save(model.state_dict(), os.path.join(
        results_dir, "final_art_model.pth"))
    print(f"Final model saved to {os.path.join(
        results_dir, 'final_art_model.pth')}")
    print(f"Training results saved to {
          os.path.join(results_dir, 'training_results.png')}")


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows support
    main()
