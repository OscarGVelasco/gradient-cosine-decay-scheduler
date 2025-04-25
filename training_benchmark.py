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
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_sch
from GradientCosineScheduler import GradientCosineScheduler
from GradientCosineOOPScheduler import GradientCosineOOPScheduler
import torchvision.models as tmodels

def load_data(dataset_name, batch_size=128, num_workers=4):
    """
    Load dataset and create data loaders

    AVAILABLE DATASETS: https://pytorch.org/vision/main/datasets.html
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
            root='./data', train=True, download=False, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=False, transform=test_transform)
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

def get_lr_scheduler(scheduler_name, optimizer, steps_per_epoch, model, num_epochs):
    """
    Load scheduler and initiate
    
    Args:
        scheduler_name (str): Name of the scheduler
        
    Returns:
        scheduler
    """
    # Select dataset
    if scheduler_name == 'LinearLR':
        scheduler = lr_sch.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
    if scheduler_name == 'GradientCosineScheduler':
        initial_lr=0.01
        final_lr=0.001
        warmup_epochs=0
        oscillation_frequency=4.0
        scheduler = GradientCosineScheduler(
            optimizer, num_epochs, steps_per_epoch, model,
            initial_lr=initial_lr, final_lr=final_lr,
            warmup_epochs=warmup_epochs,
            oscillation_frequency=oscillation_frequency
        )
    if scheduler_name == 'GradientCosineSchedulerAuto':
        initial_lr=0.01
        final_lr=0.001
        warmup_epochs=0
        oscillation_frequency=4.0
        scheduler = GradientCosineScheduler(
            optimizer, num_epochs, steps_per_epoch, model,
            mode="auto", initial_lr=initial_lr, final_lr=final_lr,
            warmup_epochs=warmup_epochs,
            oscillation_frequency=oscillation_frequency
        )
    if scheduler_name == 'GradientCosineOOPScheduler':
        initial_lr=0.01
        final_lr=0.001
        warmup_epochs=0
        gradient_scale=0.15
        oscillation_frequency=4.0
        scheduler = GradientCosineOOPScheduler(
            optimizer, num_epochs, steps_per_epoch, model,
            initial_lr=initial_lr, final_lr=final_lr,
            warmup_epochs=warmup_epochs,
            gradient_scale=gradient_scale, 
            oscillation_frequency=oscillation_frequency
        )
    if scheduler_name == 'ExponentialLR':
        scheduler = lr_sch.ExponentialLR(optimizer, gamma=0.9)
    if scheduler_name == 'CosineAnnealingLR':
        scheduler = lr_sch.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    if scheduler_name == 'CyclicLR':
        scheduler = lr_sch.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="exp_range",gamma=0.95)

    print("Loaded LR scheduler: "+scheduler_name)
    return scheduler
        
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
            model = tmodels.resnet50(weights=None)
    elif model_name == "ResNext":
        if model_config == "resnext101":
            model = tmodels.resnext101_32x8d(weights=None)    
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
    # Create DataFrame to store scheduler metrics
    scheduler_data = []
    
    with tqdm(train_loader, desc=f"Training (LR: {current_lr:.6f})") as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # Update Gradient tracking on scheduler
            if scheduler and isinstance(scheduler, GradientCosineScheduler):
                scheduler.update_gradient_norm()
            optimizer.step()
            
            # Update scheduler if it's step-based (not epoch-based)
            if scheduler and isinstance(scheduler, GradientCosineScheduler):
                scheduler.step()
            elif scheduler and isinstance(scheduler, GradientCosineOOPScheduler):
                scheduler.step()
            #current_lr = scheduler.actual_lr
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler and isinstance(scheduler, GradientCosineOOPScheduler):
                current_lr = pd.DataFrame([ [i,group["lr"]] for i,group in enumerate(optimizer.param_groups)])
                current_lr.columns = ["layer", "lr"]
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            current_accuracy = 100. * correct / total

            # Collect scheduler data
            if scheduler and isinstance(scheduler, GradientCosineOOPScheduler):
                step_data = current_lr.assign(epoch=scheduler.last_epoch, step=scheduler._step_count, loss=loss.item(), accuracy=current_accuracy)
            else:
                step_data = {
                    'epoch': scheduler.last_epoch,
                    'step': scheduler._step_count,
                    'lr': current_lr,
                    'loss': loss.item(),
                    'accuracy': current_accuracy
                }
            scheduler_data.append(step_data)

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total,
                'lr': optimizer.param_groups[0]['lr'],
                'avg grad_norm': scheduler.epoch_grad_norms[-1]
            })
    if scheduler and isinstance(scheduler, GradientCosineOOPScheduler):
        scheduler_df = pd.concat(scheduler_data)
    else:
        scheduler_df = pd.DataFrame(scheduler_data)
    return running_loss / len(train_loader), 100. * correct / total, optimizer.param_groups[0]["lr"], scheduler_df

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

def benchmark_model(config, dataset_name, scheduler_name, num_epochs=100, batch_size=128, 
                   initial_lr=0.01, final_lr=0.00001, warmup_epochs=5, oscillation_frequency=4.0):
    """
    Benchmark a model with the custom scheduler
    
    Args:
        config (dict): Model configuration
        dataset_name (str): Name of the dataset to use
        scheduler_name (str): Name of the scheduler to use
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        initial_lr (float): Initial learning rate
        final_lr (float): Final learning rate
        warmup_epochs (int): Number of warmup epochs
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
    model = None
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
    #scheduler_name = "ExponentialLR"
    #scheduler_name = "LinearLR"
    steps_per_epoch = len(train_loader)
    #scheduler = get_lr_scheduler("GradientCosineScheduler")
    # scheduler = get_lr_scheduler("CosineAnnealingLR")
    scheduler = get_lr_scheduler(scheduler_name, optimizer, steps_per_epoch, model, num_epochs)
    # Initialize tracking variables
    results = []
    results_val = []
    start_time = time.time()
    best_acc = 0.0
    for epoch in range(num_epochs):
    #for epoch in range(10):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training loop
        # Train
        train_loss, train_acc, current_lr, epoch_stats = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
                
        # Track results
        results.append(epoch_stats)
        # Collect scheduler data
        validation_data = {
            'epoch': epoch,
            'lr': current_lr,
            'loss': val_loss,
            'accuracy': val_acc
        }
        results_val.append(validation_data)
        # Save best model
        # 41.29 for Linear
        # 48.9 FOR COSINE
        # 46.46 for Cosine Annealing
        if val_acc > best_acc and val_acc > 0.9:
            best_acc = val_acc
            model_name = list(config.keys())[0] + "_" + config[list(config.keys())[0]]
            #torch.save(model.state_dict(), f"best_{model_name}_{dataset_name}.pth")
            print(f"New best accuracy: {val_acc:.2f}%! Saved model checkpoint.")
        else:
            print(f"[no improv.] Actual accuracy: {val_acc:.2f}%")
        # Update scheduler if it's epoch-based
        if scheduler and isinstance(scheduler, GradientCosineScheduler):
            scheduler.epoch_end()
        elif scheduler and isinstance(scheduler, GradientCosineOOPScheduler):
            scheduler.epoch_end()
        else:
            scheduler.step()

    # Calculate total time
    #results["total_time"] = time.time() - start_time
    results = pd.concat(results)
    results_val = pd.DataFrame(results_val)
    # Save results
    model_name = list(config.keys())[0] + "_" + config[list(config.keys())[0]]
    results_file = f"benchmark_{model_name}_{dataset_name}_{scheduler_name}.csv"
    results.to_csv(results_file)
    results_file_val = f"benchmark_{model_name}_{dataset_name}_{scheduler_name}_validation.csv"
    results_val.to_csv(results_file_val)
    print("Training Finished")
    print("Maximum Validation Acc: "+ str(best_acc))
    print(f"Results saved to {results_file}")
    
    # Plot results
    #plot_results(results)
    
    return results

def extract_lr_value(lr):
    if isinstance(lr, list):
        return lr[0]
    return lr
    
def plot_results(results, results_val):
    """Plot training curves"""
    plt.figure(figsize=(60, 15))
    # Plot accuracy
    #plt.subplot(2, 2, 1)
    #plt.plot(results["step"], results["lr"], label="Learning rate per step")
    plt.plot(results["step"], [extract_lr_value(a) for a in results["lr"]], label="Learning rate per step")
    plt.savefig(f"benchmark_lrs_per_epoch.png",  format='png')

    # Plot accuracy
    plt.figure(figsize=(30, 20))
    plt.plot(results["step"], results["accuracy"], label="accuracy rate per step")

    plt.figure(figsize=(30, 20))
    plt.plot(results_val["epoch"], results_val["accuracy"], label="accuracy rate per step")
    plt.plot(results["epoch"], results["accuracy"], label="accuracy rate per step")

    plt.figure(figsize=(8, 6))
    plt.plot(results_val["epoch"], results_val["loss"], label="accuracy rate per step")
    plt.plot(results["epoch"], results["loss"], label="accuracy rate per step")

    plt.plot(results["step"], results["loss"], label="Validation Loss")
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

def run_benchmarks(configs, dataset_name, num_epochs=100, **kwargs):
    """
    Run benchmarks for multiple model configurations
    
    Args:
        configs (list): List of model configurations
        dataset_name (str): Name of the dataset to use
        **kwargs: Additional arguments for benchmark_model function
    """
    results = {}
    #schedulers = ['GradientCosineScheduler','GradientCosineOOPScheduler','LinearLR','ExponentialLR','CosineAnnealingLR','CyclicLR']
    schedulers = ['GradientCosineSchedulerAuto']
    for config in configs:
        model_name = list(config.keys())[0] + "_" + config[list(config.keys())[0]]
        for scheduler_name in schedulers:
            print(f"\n{'='*50}")
            print(f"Benchmarking {model_name} on {dataset_name} with {scheduler_name}")
            print(f"{'='*50}")            
            benchmark_model(config, dataset_name, scheduler_name, num_epochs)
    
    return "Done"

if __name__ == "__main__":
    # Example configurations
    configs = [
        #{"ResNet": "resnet50"},
        {"ResNext": "resnext101"}
        # Add more models as needed
        # {"VGG": "vgg16"},
        # {"DenseNet": "densenet121"}
    ]
    dataset_name="CIFAR100"
    num_epochs=250
    batch_size=128
    initial_lr=0.01
    final_lr=0.001
    
    # Run benchmarks
    results = run_benchmarks(
        configs,
        dataset_name="CIFAR100",
        num_epochs=250,
        batch_size=128,
        initial_lr=0.01,
        final_lr=0.001,
        warmup_epochs=0,
        oscillation_frequency=4.0
    )
    
    print("\nBenchmark completed!")
    """
    for model_name, result in results.items():
        print(f"{model_name}: Best accuracy = {result['best_acc']:.2f}%, Training time = {result['total_time']:.2f}s")
    """