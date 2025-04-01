import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class GradientCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, model, batch_size,
                 initial_lr=0.001, final_lr=0, warmup_epochs=0, 
                 oscillation_amplitude=0.3, gradient_scale=0.15, 
                 oscillations_per_epoch=1.0, last_epoch=-1, verbose=False):
        """
        Learning rate scheduler with gradient-sensitive cosine oscillations and per-epoch frequency adjustment.
        
        Args:
            optimizer: PyTorch optimizer
            total_epochs: Total number of epochs for training
            model: PyTorch model (for parameter count)
            batch_size: Number of samples per batch
            initial_lr: Starting learning rate
            final_lr: Final learning rate
            warmup_epochs: Number of epochs for linear warmup
            oscillation_amplitude: Base amplitude for cosine oscillations
            gradient_scale: Scale factor for gradient-based modulation
            oscillations_per_epoch: Initial number of oscillations per epoch
            last_epoch: Last epoch index
            verbose: Print learning rate updates
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = final_lr
        self.initial_lr = initial_lr
        self.oscillation_amplitude = oscillation_amplitude
        self.gradient_scale = gradient_scale
        
        # Model size affects oscillation frequency (larger models = slower oscillations)
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.model_size_factor = math.log10(max(1, self.num_params)) / 10
        
        # Frequency and gradient tracking
        self.current_frequency = oscillations_per_epoch / (1.0 + self.model_size_factor)
        self.last_grad_norm = 1.0
        self.epoch_grad_norms = []
        self.steps_in_epoch = 0
        self.current_epoch = 0
        
        super(GradientCosineScheduler, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        # Handle warmup phase
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [self.min_lr + (self.initial_lr - self.min_lr) * warmup_factor 
                   for _ in self.base_lrs]
        
        # Calculate progress (0 to 1)
        if self.warmup_epochs > 0:
            epoch_progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        else:
            epoch_progress = self.last_epoch / self.total_epochs
        
        epoch_progress = min(max(0, epoch_progress), 1)
        step_progress = min(1.0, self.steps_in_epoch / max(1, self.steps_in_epoch))
        
        # Linear decay trajectory 
        linear_decay_rate = (self.initial_lr - self.min_lr) * (1.0 - epoch_progress)
        linear_decay = self.min_lr + linear_decay_rate
        
        # Calculate oscillation
        gradient_factor = min(1.0, self.last_grad_norm * self.gradient_scale)
        amplitude = self.oscillation_amplitude * gradient_factor * linear_decay_rate
        phase_offset = math.pi / 2
        
        # Oscillation formula with per-epoch frequency
        oscillation = amplitude * math.cos(
            (epoch_progress + step_progress/self.total_epochs) * 
            math.pi * 2 * self.current_frequency * self.total_epochs + phase_offset
        )
        
        return [max(self.min_lr, linear_decay + oscillation) for _ in self.base_lrs]
    
    def update_gradient_norm(self):
        """Update gradient statistics after backward() but before optimizer.step()"""
        total_norm = 0.0
        param_count = 0
        
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
        
        if param_count > 0:
            avg_norm = math.sqrt(total_norm / param_count)
            self.last_grad_norm = 0.9 * self.last_grad_norm + 0.1 * avg_norm
            self.epoch_grad_norms.append(self.last_grad_norm)
        
        return self.last_grad_norm
    
    def step_batch(self):
        """Call after each batch to track within-epoch progress"""
        self.steps_in_epoch += 1
    
    def update_frequency(self):
        """Adjust oscillation frequency based on epoch's gradient behavior"""
        if not self.epoch_grad_norms:
            return
            
        # Calculate gradient statistics
        avg_norm = sum(self.epoch_grad_norms) / len(self.epoch_grad_norms)
        variance = sum((g - avg_norm) ** 2 for g in self.epoch_grad_norms) / max(1, len(self.epoch_grad_norms))
        
        # Higher variance → more oscillations for exploration
        # Lower variance → fewer oscillations for exploitation
        norm_variance = min(2.0, max(0.5, 1.0 + variance / (avg_norm + 1e-8)))
        base_freq = self.base_lrs[0] / (1.0 + self.model_size_factor)
        self.current_frequency = base_freq * norm_variance
        
        # Reset for next epoch
        self.epoch_grad_norms = []
        self.steps_in_epoch = 0
        self.current_epoch += 1
        
        if self.verbose:
            print(f"Epoch {self.current_epoch}: Frequency set to {self.current_frequency:.4f}")
    
    def step(self, epoch=None):
        """End of epoch: update frequency and step LR scheduler"""
        self.update_frequency()
        super().step(epoch)