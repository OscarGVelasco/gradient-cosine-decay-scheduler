import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import _LRScheduler
import math

class GradientCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, steps_per_epoch, model, 
                 initial_lr=0.01, final_lr=0.00001, 
                 warmup_epochs=0,
                 gradient_scale=0.15, oscillation_frequency=4.0,
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
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = steps_per_epoch*total_epochs
        self.initial_lr = initial_lr
        self.min_lr = final_lr
        self.alpha = final_lr*0.1
        self.model = model
        
        # Warm-up and init lr
        self.warmup_epochs = warmup_epochs
        if self.warmup_epochs > 0:
            # Correct total step number if warmup steps
            self.total_steps = self.total_steps - (self.warmup_epochs*self.steps_per_epoch)
            # We start with a 0.1 of the minimum lr for warmp-up
            self.actual_lr = self.alpha
            # Smooth transition from linear warmup to cos intial wave at initial lr
            self.phi=-(math.pi/2)
        # No warmup
        else:
            self.actual_lr = self.initial_lr
            # If no warmpup we start the wave at maximum x negative position
            self.phi=math.pi

        # Compute the linear step-decay (How much the lr decrease each step)
        self.linear_step_decay = (self.initial_lr - self.min_lr)*((self.total_steps-1) / self.total_steps)

        # Oscillation parameters
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
        self._set_initial_lrs(optimizer)

    def _set_initial_lrs(self, optimizer):
        """
        Initialize the scheduler at the start of training
        """
        # Set initial learning rates
        self.base_lrs_original = [group['lr'] for group in optimizer.param_groups]
        self.base_lrs = [self.actual_lr for _ in optimizer.param_groups]
        # Update learning rates
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.actual_lr

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
        x_t = max(x_O + (A * math.cos((W * t) + phi)), self.alpha)
        return x_t
            
    def get_lr(self):
        
        # Oscillation amplitude modulated by gradient norm
        gradient_factor = min(1.0, self.last_grad_norm * self.gradient_scale)
                
        W = (2 * math.pi) / (self.steps_per_epoch/self.oscillation_frequency)

        # Compute learning rate with oscillation
        lr_values = [
            self.harmonic_cos(
                x_O=self.actual_lr, 
                A=self.oscillation_amplitude, 
                W=W, 
                t=self.current_step, 
                phi=self.phi
            ) for _ in self.base_lrs
        ]

        # Final learning rate with oscillation
        return lr_values
    
    def step(self, optimizer):
        """
        Update learning rate step
        """        
        # In warm-up phase
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            # Linear warmup from min_lr to initial_lr
            warmup_factor = self.last_epoch / self.warmup_epochs
            self.actual_lr = self.alpha + (self.initial_lr - self.alpha) * warmup_factor
            return [self.actual_lr for _ in self.base_lrs]

        # After warmup, calculate main schedule
        self.actual_lr = self.actual_lr - self.linear_step_decay
        """
        if self.warmup_epochs > 0:
            # Adjust progress to account for warmup
            current_progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        else:
            current_progress = self.last_epoch / self.total_epochs
        """

        # Gradient-aware amplitude modulation
        #gradient_factor = min(1.0, self.last_grad_norm * self.gradient_scale)
        #current_amplitude = self.oscillation_amplitude * gradient_factor
        
        # Oscillation frequency adjusted by model size
        #current_frequency = self.oscillation_frequency / (1.0 + self.model_size_factor)   

        # Compute learning rate with oscillation
        lr_values = self.get_lr()
        
        # Update learning rates
        for param_group, lr in zip(optimizer.param_groups, lr_values):
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
        print(f"Epoch {self.current_epoch}: Normalized variance: {self.normalized_variance:.4f}")

        # Adjust frequency based on gradient behavior
        # Higher variance → more oscillations needed for exploration
        # Lower variance → fewer oscillations for exploitation
        #self.current_frequency = self.base_oscillations_per_epoch * normalized_variance / (1.0 + self.model_size_factor)
        
        # Reset for next epoch
        self.epoch_grad_norms = []
        self.steps_in_current_epoch = 0
        #self.current_step = 0
        self.current_epoch += 1
        
        if self.verbose:
            print(f"Epoch {self.current_epoch}: Setting oscillation frequency to {self.current_frequency:.4f}")
    
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
