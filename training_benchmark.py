import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import GradientCosineScheduler

def load_data(dataset_name, batch_size=128, num_workers=4):
    """
    Load dataset and create data loaders
    
    Args:
        dataset_name (str): Name of the dataset (currently supports 'CIFAR10', 'CIFAR100')
        batch_size (int): Batch size for training and testing
        num_workers (int): Number of workers for data loading
        
    Returns:
        train_loader, test_loader, num_classes
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Select dataset
    if dataset_name.upper() == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform)
        num_classes = 10
    elif dataset_name.upper() == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=test_transform)
        num_classes = 100
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader, num_classes

def get_model(config):
    """
    Create a model based on the configuration
    
    Args:
        config (dict): Model configuration with name and specific parameters
        
    Returns:
        model: PyTorch model
    """
    model_name = list(config.keys())[0]
    model_config = config[model_name]
    
    if model_name == "ResNet":
        if model_config == "resnet50":
            # No weights - random initialization
            model = resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet configuration: {model_config}")
    # Add more model types here as needed
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    current_lr = optimizer.param_groups[0]['lr']
    
    with tqdm(train_loader, desc=f"Training (LR: {current_lr:.6f})") as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Update scheduler if it's step-based (not epoch-based)
            if scheduler and isinstance(scheduler, GradientCosineScheduler):
                scheduler.step()
                current_lr = scheduler.get_lr()[0]
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total,
                'lr': current_lr
            })
    
    return running_loss / len(train_loader), 100. * correct / total, current_lr

def validate(model, test_loader, criterion, device):
    """Evaluate the model on the test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Validating") as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1),
                    'acc': 100. * correct / total
                })
    
    return running_loss / len(test_loader), 100. * correct / total

def benchmark_model(config, dataset_name, num_epochs=100, batch_size=128, 
                   initial_lr=0.01, final_lr=0.00001, warmup_epochs=5,
                   gradient_scale=0.15, oscillation_frequency=4.0):
    """
    Benchmark a model with the custom scheduler
    
    Args:
        config (dict): Model configuration
        dataset_name (str): Name of the dataset to use
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        initial_lr (float): Initial learning rate
        final_lr (float): Final learning rate
        warmup_epochs (int): Number of warmup epochs
        gradient_scale (float): Scale factor for gradient in the scheduler
        oscillation_frequency (float): Frequency of cosine oscillation
        
    Returns:
        dict: Dictionary with benchmark results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader, num_classes = load_data(
        dataset_name, batch_size=batch_size
    )
    
    # Get model
    model = get_model(config)
    
    # Modify final layer if needed to match the number of classes
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
    
    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = GradientCosineScheduler(
        optimizer, num_epochs, steps_per_epoch, model,
        initial_lr=initial_lr, final_lr=final_lr,
        warmup_epochs=warmup_epochs,
        gradient_scale=gradient_scale, 
        oscillation_frequency=oscillation_frequency
    )
    
    # Initialize tracking variables
    results = {
        "model_config": config,
        "dataset": dataset_name,
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
        "best_acc": 0.0,
        "total_time": 0.0
    }
    
    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc, current_lr = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Update scheduler if it's epoch-based
        if scheduler and not isinstance(scheduler, GradientCosineScheduler):
            scheduler.step()
        
        # Track results
        results["epochs"].append(epoch + 1)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["learning_rates"].append(current_lr)
        
        # Save best model
        if val_acc > results["best_acc"]:
            results["best_acc"] = val_acc
            model_name = list(config.keys())[0] + "_" + config[list(config.keys())[0]]
            torch.save(model.state_dict(), f"best_{model_name}_{dataset_name}.pth")
            print(f"New best accuracy: {val_acc:.2f}%! Saved model checkpoint.")
    
    # Calculate total time
    results["total_time"] = time.time() - start_time
    
    # Save results
    model_name = list(config.keys())[0] + "_" + config[list(config.keys())[0]]
    results_file = f"benchmark_{model_name}_{dataset_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_file}")
    
    # Plot results
    plot_results(results)
    
    return results

def plot_results(results):
    """Plot training curves"""
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(results["epochs"], results["train_acc"], label="Training Accuracy")
    plt.plot(results["epochs"], results["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy")
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(results["epochs"], results["train_loss"], label="Training Loss")
    plt.plot(results["epochs"], results["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(results["epochs"], results["learning_rates"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    
    # Save plot
    model_name = list(results["model_config"].keys())[0] + "_" + results["model_config"][list(results["model_config"].keys())[0]]
    plt.tight_layout()
    plt.savefig(f"benchmark_{model_name}_{results['dataset']}.png")
    plt.close()

def run_benchmarks(configs, dataset_name, **kwargs):
    """
    Run benchmarks for multiple model configurations
    
    Args:
        configs (list): List of model configurations
        dataset_name (str): Name of the dataset to use
        **kwargs: Additional arguments for benchmark_model function
    """
    results = {}
    
    for config in configs:
        model_name = list(config.keys())[0] + "_" + config[list(config.keys())[0]]
        print(f"\n{'='*50}")
        print(f"Benchmarking {model_name} on {dataset_name}")
        print(f"{'='*50}")
        
        results[model_name] = benchmark_model(config, dataset_name, **kwargs)
    
    return results

if __name__ == "__main__":
    # Example configurations
    configs = [
        {"ResNet": "resnet50"},
        # Add more models as needed
        # {"VGG": "vgg16"},
        # {"DenseNet": "densenet121"}
    ]
    
    # Run benchmarks
    results = run_benchmarks(
        configs, 
        dataset_name="CIFAR100",
        num_epochs=50,
        batch_size=128,
        initial_lr=0.01,
        final_lr=0.0001,
        warmup_epochs=5,
        gradient_scale=0.15,
        oscillation_frequency=4.0
    )
    
    print("\nBenchmark completed!")
    for model_name, result in results.items():
        print(f"{model_name}: Best accuracy = {result['best_acc']:.2f}%, Training time = {result['total_time']:.2f}s")