"""
Author: Óscar González-Velasco, PhD
Email: oscargvelasco@gmail.com / oscar.gonzalezvelasco@dkfz-heidelberg.de
Date: 17 March 2025

This code provides a custom PyTorch learning rate scheduler that implements the ncosad learning rate scheduler:
GraCos

Cosine oscillation pattern that follows a descending linear trajectory towards zero
Model size awareness - larger models have slower oscillations (using parameter count as a proxy)
Gradient-based modulation - the amplitude of oscillations is affected by gradient magnitudes

Documentation:
http://labman.phys.utk.edu/phys135core/modules/m9/harmonic_motion.html#:~:text=x(t)%20%3D%20A%20cos,return%20to%20the%20starting%20position.

Features:

Adaptive oscillation frequency: Based on model size (number of parameters)
Gradient-aware amplitude: Oscillation amplitude is modulated by gradient magnitude
Warm-up period: Optional linear warm-up for stable initial training
Base trajectory: Linear descent towards zero, with oscillations around this trajectory

This controls how much lower than the base learning rate we'll start (default 0.5 = 50% of base LR)
For example, if base LR is 0.01 and initial_multiplier is 0.3, we'll start at 0.003


Phase offset for cosine function:

Added phase_offset = math.pi / 2 to start the cosine at its lowest point
This ensures the first value is below the linear decay trajectory, used as a natural
warm-up period.


Early-stage blending:

Special handling for the initial phase of training
Gradually blends from the initial lower rate to the regular oscillating schedule
This gives a smooth transition from the low starting point


Optional explicit warmup:

Kept the explicit warmup functionality (if needed)
This would provide an additional linear warmup before starting oscillations

"""
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
import numpy as np


class GradientAwareCosineScheduler(_LRScheduler):
    """
    Cosine learning rate scheduler with oscillation frequency dependent on model size and gradient information.
    
    The learning rate follows a cosine function that oscillates around a linear descending trajectory.
    The scheduler starts with a lower value than the base learning rate and oscillates upward.
    """
    
    def __init__(self, optimizer, total_epochs, model, warmup_epochs=0, min_lr=0, 
                 last_epoch=-1, verbose=False, base_amplitude=0.5, gradient_scale=0.1,
                 initial_multiplier=0.5):
        """
        Args:
            optimizer: PyTorch optimizer
            total_epochs: Total number of epochs for training
            model: PyTorch model to count parameters
            warmup_epochs: Number of epochs for additional linear warmup (if needed)
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
            verbose: Print learning rate updates
            base_amplitude: Base amplitude of cosine oscillation
            gradient_scale: Scale factor for gradient-based modulation
            initial_multiplier: Starting LR will be base_lr * initial_multiplier
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.model = model
        self.base_amplitude = base_amplitude
        self.gradient_scale = gradient_scale
        self.initial_multiplier = initial_multiplier
        
        # Calculate number of parameters as a proxy for model size
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model size factor affects oscillation frequency (larger models = slower oscillations)
        self.model_size_factor = math.log10(max(1, self.num_params)) / 10
        
        # Initialize gradient norm tracker
        self.last_grad_norm = 1.0
        
        super(GradientAwareCosineScheduler, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        # In explicit warmup phase (if specified)
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            # Linear warmup starts from initial_multiplier * base_lr to base_lr
            warmup_factor = self.initial_multiplier + (1.0 - self.initial_multiplier) * (self.last_epoch / self.warmup_epochs)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Main scheduling phase
        if self.warmup_epochs > 0:
            # Adjust for warmup period
            current_progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        else:
            current_progress = self.last_epoch / self.total_epochs
        
        # Linear decay trajectory from 1.0 to 0.0
        linear_decay = 1.0 - current_progress
        
        # Oscillation frequency slows down as model gets larger
        frequency = 1.0 / (1.0 + self.model_size_factor)
        
        # Oscillation amplitude is modulated by gradient norm
        amplitude = self.base_amplitude * min(1.0, self.last_grad_norm * self.gradient_scale)
        
        # Phase offset to ensure we start below the base learning rate
        phase_offset = math.pi / 2  # Start at the lowest point of cosine
        
        # Cosine oscillation around the linear decay trajectory
        # Using phase offset to ensure we start below base_lr
        oscillation = amplitude * math.cos(current_progress * math.pi * 2 * frequency * 5 + phase_offset)
        
        # Combine linear decay and oscillation
        # Start below base_lr by using initial_multiplier to scale the starting point
        if self.last_epoch == 0:
            # First epoch uses initial_multiplier directly
            factor = self.initial_multiplier
        else:
            # Calculate cosine wave that starts below base_lr and oscillates
            factor = max(0.0, linear_decay + oscillation)
            
            # Ensure the first oscillation starts below the base learning rate
            if self.last_epoch <= self.total_epochs / (10 * frequency):  # Early in training
                # Blend between initial_multiplier and the regular schedule
                blend_factor = min(1.0, self.last_epoch / (self.total_epochs / (10 * frequency)))
                initial_factor = self.initial_multiplier + (linear_decay + oscillation - self.initial_multiplier) * blend_factor
                factor = initial_factor
        
        # Calculate new learning rates
        return [max(self.min_lr, base_lr * factor) for base_lr in self.base_lrs]
    
    def update_gradient_norm(self, optimizer):
        """
        Update the gradient norm tracker based on current gradients in the optimizer.
        Call this method after loss.backward() but before optimizer.step()
        Uses efficient computation to minimize overhead.
        """
        total_norm = 0.0
        param_count = 0
        
        # Option for efficient calculation: sample parameters or process in batches
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    # Calculate L2 norm efficiently
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
        
        if param_count > 0:
            avg_norm = math.sqrt(total_norm / param_count)
            # Use exponential moving average to smooth gradient norm changes
            self.last_grad_norm = 0.9 * self.last_grad_norm + 0.1 * avg_norm
        
        return self.last_grad_norm


# Example usage showing visual plotting of the learning rate schedule
def plot_lr_schedule(model, total_epochs=100, initial_lr=0.01, initial_multiplier=0.3):
    import matplotlib.pyplot as plt
    
    # Create a dummy optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
    
    # Create our scheduler
    scheduler = GradientAwareCosineScheduler(
        optimizer=optimizer,
        total_epochs=total_epochs,
        model=model,
        initial_multiplier=initial_multiplier,
        base_amplitude=0.3,
        gradient_scale=0.15
    )
    
    # Simulate training and record learning rates
    lrs = []
    for epoch in range(total_epochs):
        # Update learning rate
        scheduler.step()
        
        # Store current learning rate
        lrs.append(optimizer.param_groups[0]['lr'])
    
    # Plot the learning rate schedule
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.axhline(y=initial_lr, color='r', linestyle='--', label=f'Base LR: {initial_lr}')
    plt.axhline(y=initial_lr*initial_multiplier, color='g', linestyle='--', 
                label=f'Starting LR: {initial_lr*initial_multiplier}')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_schedule.png')
    plt.close()
    
    return lrs