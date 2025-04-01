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
http://labman.phys.utk.edu/phys135core/modules/m9/harmonic_motion.html

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
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class GradientCosineScheduler(_LRScheduler):
    
    def __init__(self, optimizer, total_epochs, model, 
                 initial_lr=0.001, final_lr=0, 
                 warmup_epochs=0, oscillation_amplitude=0.3,
                 gradient_scale=0.15, oscillation_frequency=1.0,
                 last_epoch=-1, verbose=False):
        """
        Args:
            optimizer: PyTorch optimizer
            total_epochs: Total number of epochs for training
            model: PyTorch model to count parameters
            initial_lr: Starting learning rate (default 0.001)
            final_lr: Final learning rate (default 0)
            warmup_epochs: Number of epochs for linear warmup
            oscillation_amplitude: Base amplitude of cosine oscillation
            gradient_scale: Scale factor for gradient-based amplitude modulation
            oscillation_frequency: Base frequency multiplier
            last_epoch: Last epoch index
            verbose: Print learning rate updates
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = final_lr
        self.initial_lr = initial_lr
        self.model = model
        self.oscillation_amplitude = oscillation_amplitude
        self.gradient_scale = gradient_scale
        self.oscillation_frequency = oscillation_frequency
        
        # Calculate number of parameters as a proxy for model size
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model size factor affects oscillation frequency (larger models = slower oscillations)
        self.model_size_factor = math.log10(max(1, self.num_params)) / 10
        
        # Initialize gradient norm tracker
        self.last_grad_norm = 1.0
        
        # Override base_lrs with our initial learning rate
        self.base_lrs_original = None
        
        super(GradientCosineScheduler, self).__init__(optimizer, last_epoch, verbose)
    
    def _set_initial_lrs(self):
        """Override to set base_lrs to the user-specified value rather than optimizer's lr"""
        self.base_lrs_original = [group['lr'] for group in self.optimizer.param_groups]
        self.base_lrs = [self.initial_lr for _ in self.optimizer.param_groups]
        
    def get_lr(self):
        if self.base_lrs_original is None:
            # First call, save original LRs and set our initial LR
            self._set_initial_lrs()
        
        # In warm-up phase
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            # Linear warmup from min_lr to initial_lr
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [self.min_lr + (self.initial_lr - self.min_lr) * warmup_factor 
                    for _ in self.base_lrs]
        
        # After warmup, calculate main schedule
        if self.warmup_epochs > 0:
            # Adjust progress to account for warmup
            current_progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        else:
            current_progress = self.last_epoch / self.total_epochs
        
        # Ensure current_progress is clipped to [0, 1]
        current_progress = min(max(0, current_progress), 1)
        
        # Linear decay trajectory from initial_lr to min_lr
        linear_decay_rate = (self.initial_lr - self.min_lr) * (1.0 - current_progress)
        linear_decay = self.min_lr + linear_decay_rate
        
        # Oscillation frequency adjusted by model size
        frequency = self.oscillation_frequency / (1.0 + self.model_size_factor)
        
        # Oscillation amplitude modulated by gradient norm
        gradient_factor = min(1.0, self.last_grad_norm * self.gradient_scale)
        amplitude = self.oscillation_amplitude * gradient_factor * linear_decay_rate
        
        # Phase offset to ensure we start at the lowest point of cosine
        phase_offset = math.pi / 2
        
        # Cosine oscillation around the linear decay trajectory
        oscillation = amplitude * math.cos(current_progress * math.pi * 2 * frequency * 5 + phase_offset)
        
        # Final learning rate with oscillation
        return [max(self.min_lr, linear_decay + oscillation) for _ in self.base_lrs]
    
    def update_gradient_norm(self, optimizer=None):
        """
        Update the gradient norm tracker based on current gradients.
        Call this method after loss.backward() but before optimizer.step()
        
        Args:
            optimizer: Optional. If None, uses the optimizer this scheduler is attached to.
        """
        if optimizer is None:
            optimizer = self.optimizer
            
        total_norm = 0.0
        param_count = 0
        
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
        
        if param_count > 0:
            avg_norm = math.sqrt(total_norm / param_count)
            # Use exponential moving average to smooth gradient norm changes
            self.last_grad_norm = 0.9 * self.last_grad_norm + 0.1 * avg_norm
        
        return self.last_grad_norm
    
    def step(self, epoch=None):
        """
        Override step method to update gradient norm if not done manually.
        It's still better to call update_gradient_norm() explicitly after backward().
        """
        # Update learning rate
        super(GradientCosineScheduler, self).step(epoch)

    def harmonic_cos(x_O, A, W, t, phi):
        # x_O = equilibrium position
        # W = angular frequency
        # phi = phase
        # A = amplitude
        x_t = x_O + (A * math.cos((W * t) + phi))
        return(x_t)

"""
I might be missing something, but calculating the norm twice isn’t necessary. It’s about 3x faster to concat all the grads into a single tensor then calculate the norm once:
"""
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    norm = torch.cat(grads).norm()