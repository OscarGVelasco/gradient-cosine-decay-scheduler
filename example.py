import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.optim.lr_scheduler import _LRScheduler


class SimpleModel(nn.Module):
    """A simple model to use for demonstration purposes"""
    def __init__(self, input_size=784, hidden_size=100, output_size=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def plot_lr_schedule_comparison(total_epochs=100, initial_lr=0.001, final_lr=0):
    """
    Plot learning rate schedules with different configurations and compare them.
    
    This function creates a dummy model, applies different scheduler configurations, 
    and visualizes the resulting learning rate patterns.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a dummy model
    model = SimpleModel()
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    
    # Set up configurations for comparison
    configs = [
        {"name": "Basic (No Warmup)", "warmup": 0, "amp": 0.3, "grad_scale": 0.15, "freq": 1.0},
        {"name": "With Warmup", "warmup": 10, "amp": 0.3, "grad_scale": 0.15, "freq": 1.0},
        {"name": "Higher Amplitude", "warmup": 0, "amp": 0.5, "grad_scale": 0.15, "freq": 1.0},
        {"name": "Faster Oscillation", "warmup": 0, "amp": 0.3, "grad_scale": 0.15, "freq": 2.0},
        {"name": "Gradient Sensitive", "warmup": 0, "amp": 0.3, "grad_scale": 0.3, "freq": 1.0}
    ]
    
    # Plot setup
    plt.figure(figsize=(15, 10))
    
    # Generate random gradient norms - same for all configs to ensure fair comparison
    np.random.seed(42)  # For reproducibility
    grad_norms = np.random.normal(1.0, 0.3, total_epochs)
    
    # Create a line for the linear decay reference
    linear_decay = [initial_lr * (1 - epoch / total_epochs) for epoch in range(total_epochs)]
    plt.plot(linear_decay, '--', color='gray', alpha=0.5, label='Linear Decay Reference')
    
    # Process each configuration
    for i, config in enumerate(configs):
        # Create a dummy optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Initial LR doesn't matter
        
        # Create scheduler with this configuration
        scheduler = GradientCosineScheduler(
            optimizer=optimizer,
            total_epochs=total_epochs,
            model=model,
            initial_lr=initial_lr,
            final_lr=final_lr,
            warmup_epochs=config["warmup"],
            oscillation_amplitude=config["amp"],
            gradient_scale=config["grad_scale"],
            oscillation_frequency=config["freq"]
        )
        
        # Simulate training and record learning rates
        lrs = []
        
        for epoch in range(total_epochs):
            # Update gradient norm with simulated value
            scheduler.last_grad_norm = grad_norms[epoch]
            
            # Update learning rate
            scheduler.step()
            print(scheduler._step_count)
            # Store current learning rate
            lrs.append(optimizer.param_groups[0]['lr'])
        
        # Plot this configuration
        plt.plot(lrs, label=config["name"], linewidth=2)
    
    # Add guidelines and labels
    plt.axhline(y=initial_lr, color='r', linestyle='--', label=f'Initial LR: {initial_lr}')
    plt.axhline(y=final_lr, color='g', linestyle='--', label=f'Final LR: {final_lr}')
    
    # Add plot details
    plt.title('Learning Rate Schedule Comparison', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('lr_schedule_comparison.png', dpi=300)
    plt.close()
    
    print("Plot saved as 'lr_schedule_comparison.png'")
    
    # Return the basic configuration learning rates for reference
    return lrs

def simulate_training_with_scheduler(model, total_epochs=20, initial_lr=0.001):
    """
    Simulate a training loop to demonstrate how to use the scheduler in practice.
    """
    # Create a simple model and dummy data
    batch_size = 32
    input_size = 784
    
    # Create some random data
    dummy_inputs = torch.randn(batch_size, input_size)
    dummy_targets = torch.randint(0, 10, (batch_size,))
    
    # Set up optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Initial LR will be overridden
    criterion = nn.CrossEntropyLoss()
    
    # Create scheduler
    scheduler = ImprovedGradientCosineScheduler(
        optimizer=optimizer,
        total_epochs=total_epochs,
        model=model,
        initial_lr=initial_lr,
        final_lr=0.0001,
        warmup_epochs=3,
        oscillation_amplitude=0.3,
        gradient_scale=0.15
    )
    
    # Lists to store metrics
    losses = []
    learning_rates = []
    gradient_norms = []
    
    # Training loop
    print("Starting training simulation...")
    for epoch in range(total_epochs):
        # Forward pass
        outputs = model(dummy_inputs)
        loss = criterion(outputs, dummy_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Track gradient norm
        current_grad_norm = scheduler.update_gradient_norm()
        gradient_norms.append(current_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        # Scheduler step
        scheduler.step()
        
        # Record metrics
        losses.append(loss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        print(f"Epoch {epoch+1}/{total_epochs} - Loss: {loss.item():.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
              f"Grad Norm: {current_grad_norm:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(3, 1, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate
    plt.subplot(3, 1, 2)
    plt.plot(learning_rates)
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.grid(True, alpha=0.3)
    
    # Plot gradient norm
    plt.subplot(3, 1, 3)
    plt.plot(gradient_norms)
    plt.title('Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_simulation.png', dpi=300)
    plt.close()
    
    print("Training simulation completed and plot saved as 'training_simulation.png'")
    
    return {
        'losses': losses,
        'learning_rates': learning_rates,
        'gradient_norms': gradient_norms
    }


if __name__ == "__main__":
    # Run the comparison plot
    print("Generating learning rate schedule comparison...")
    
    plot_lr_schedule_comparison(total_epochs=100, initial_lr=0.001, final_lr=0)
    
    # Run the training simulation
    print("\nRunning training simulation...")
    simulate_training_with_scheduler(total_epochs=20, initial_lr=0.001)
