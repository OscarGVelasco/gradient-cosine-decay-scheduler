import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class GradientCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, steps_per_epoch, model, 
                 initial_lr=0.01, final_lr=0.00001, 
                 warmup_epochs=0,
                 gradient_scale=0.15, oscillation_frequency=1.0,
                 last_epoch=-1, verbose=False):
        """
        Args:
            optimizer: PyTorch optimizer
            total_epochs: Total number of training epochs
            steps_per_epoch: Total steps per epoch
            model: PyTorch model to count parameters
            initial_lr: Starting learning rate
            final_lr: Final learning rate
            warmup_epochs: Number of epochs for linear warmup
            oscillation_amplitude: Base amplitude of cosine oscillation
            gradient_scale: Scale factor for gradient-based amplitude modulation
            oscillation_frequency: Base frequency multiplier
            last_epoch: Last epoch index
            verbose: Print learning rate updates
        """
        self.total_epochs = total_epochs
        self.total_steps = steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.min_lr = final_lr
        self.alpha = final_lr/10
        self.initial_lr = initial_lr
        self.model = model
        
        # Oscillation parameters
        self.base_oscillation_amplitude = initial_lr
        self.oscillation_amplitude = initial_lr
        self.gradient_scale = gradient_scale
        self.base_oscillation_frequency = oscillation_frequency
        self.oscillation_frequency = oscillation_frequency
        
        # Model size and gradient tracking
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.model_size_factor = math.log10(max(1, self.num_params)) / 10
        
        # Gradient norm tracking
        self.last_grad_norm = 1.0
        self.epoch_grad_norms = []
        
        # Tracking variables
        self.current_step = 0
        self.current_epoch = 0
        
        # Initialize base learning rates
        self.base_lrs_original = None
        
        super(GradientCosineScheduler, self).__init__(optimizer, last_epoch, verbose)
        # Initiate lrs:
        self._set_initial_lrs()

    @staticmethod
    def harmonic_cos(self, x_O, A, W, t, phi):
        """
        Harmonic cosine oscillation function
        Args:
            x_O: Equilibrium position
            A: Amplitude
            W: Angular frequency
            t: Time
            phi: Phase offset
        Returns:
            Oscillation value
        """
        x_t = x_O + (A * math.cos((W * t) + phi))
        x_t = max(x_t, self.alpha)
        return x_t
    
    def _set_initial_lrs(self):
        """
        Initialize the scheduler at the start of training
        """
        # Set initial learning rates
        self.base_lrs_original = [group['lr'] for group in self.optimizer.param_groups]
        self.base_lrs = [self.initial_lr for _ in self.optimizer.param_groups]
    
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
            self.epoch_grad_norms.append(avg_norm)
        
        return self.last_grad_norm
    
    def get_lr(self):
                
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
    
    def step(self):
        """
        Update learning rate step
        """        
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



        # Calculate progress within the current epoch
        progress_in_epoch = self.current_step / self.total_steps
        
        # Linear decay base
        progress_overall = (self.current_epoch + progress_in_epoch) / self.total_epochs
        linear_decay_rate = (self.initial_lr - self.min_lr) * (1.0 - progress_overall)
        linear_decay = self.min_lr + linear_decay_rate
        
        # Gradient-aware amplitude modulation
        gradient_factor = min(1.0, self.last_grad_norm * self.gradient_scale)
        current_amplitude = self.oscillation_amplitude * gradient_factor
        
        # Oscillation frequency adjusted by model size
        current_frequency = self.oscillation_frequency / (1.0 + self.model_size_factor)
        
        # Compute learning rate with oscillation
        lr_values = [
            self.harmonic_cos(
                x_O=linear_decay, 
                A=current_amplitude * linear_decay_rate, 
                W=current_frequency * 2 * math.pi, 
                t=progress_in_epoch, 
                phi=math.pi
            ) for _ in self.base_lrs
        ]
        
        # Update learning rates
        for param_group, lr in zip(self.optimizer.param_groups, lr_values):
            param_group['lr'] = lr
        
        # Increment step
        self.current_step += 1
    
    def epoch_end(self):
        """
        Adjust oscillation frequency based on gradient behavior in the previous epoch.
        Called at the end of each epoch to set frequency for the next epoch.
        """
        if not self.epoch_grad_norms:
            return  # No gradient information available yet
        
        # Calculate gradient statistics for the epoch
        avg_grad_norm = sum(self.epoch_grad_norms) / len(self.epoch_grad_norms)
        grad_variance = sum((g - avg_grad_norm) ** 2 for g in self.epoch_grad_norms) / len(self.epoch_grad_norms)
        
        # Normalize variance to a reasonable range (0.5 to 2.0)
        normalized_variance = min(2.0, max(0.5, 1.0 + grad_variance / (avg_grad_norm + 1e-8)))
        
        # Adjust frequency based on gradient behavior
        # Higher variance → more oscillations needed for exploration
        # Lower variance → fewer oscillations for exploitation
        self.current_frequency = self.base_oscillations_per_epoch * normalized_variance / (1.0 + self.model_size_factor)
        
        # Reset for next epoch
        self.epoch_grad_norms = []
        self.steps_in_current_epoch = 0
        self.current_step = 0
        self.current_epoch += 1
        
        if self.verbose:
            print(f"Epoch {self.current_epoch}: Setting oscillation frequency to {self.current_frequency:.4f}")