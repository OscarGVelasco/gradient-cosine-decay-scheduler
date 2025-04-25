import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class GradientCosineOOPScheduler(_LRScheduler):
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
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = steps_per_epoch*total_epochs
        self.initial_lr = initial_lr
        self.min_lr = final_lr
        self.alpha = final_lr*0.1
        self.model = model
        # Warm-up and init lr
        self.warmup_epochs = warmup_epochs
        # No warmup
        self.base_lr = self.initial_lr
        # If no warmpup we start the wave at maximum x negative position
        self.phi=math.pi
        self.phase_shift = None
        #  calling the constructor (initialization method) of the parent class that GradientCosineScheduler inherits from, which is PyTorch's _LRScheduler.
        super(GradientCosineOOPScheduler, self).__init__(optimizer, last_epoch, verbose)
        # Tracking variables
        self.last_epoch = 0
        self._step_count = 0
        self.lr = initial_lr
        if self.warmup_epochs > 0:
            # Correct total step number if warmup steps
            self.total_steps = self.total_steps - (self.warmup_epochs*self.steps_per_epoch)
            # We start with a 0.1 of the minimum lr for warmp-up
            self.base_lr = self.min_lr
            # Smooth transition from linear warmup to cos intial wave at initial lr
            self.phi=-(math.pi/2)

        # Compute the linear step-decay (How much the lr decrease each step)
        self.linear_step_decay = (self.initial_lr - self.min_lr)*(1- ((self.total_steps-1) / self.total_steps))

        # Oscillation parameters
        self.oscillation_amplitude = initial_lr*0.8
        self.gradient_scale = gradient_scale
        self.base_oscillation_frequency = oscillation_frequency
        self.oscillation_frequency = oscillation_frequency
        
        # Model size and gradient tracking
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.model_size_factor = math.log10(max(1, self.num_params)) / 10
        
        # Gradient norm tracking
        self.last_grad_norm = 1.0
        self.epoch_grad_norms = []
                
        # Initialize base learning rates
        self.base_lrs_original = None
        
        # Initiate lrs:
        self._set_initial_lrs()

    def _initial_step(self):
        return None

    def get_last_lr(self) -> list[float]:
        """Return last computed learning rate by current scheduler."""
        return self.lr

    def _set_initial_lrs(self):
        """
        Initialize the scheduler at the start of training
        """
        # Set initial learning rates
        self.base_lrs_original = [group['lr'] for group in self.optimizer.param_groups]
        # Reconstruct model layers on optimizer:
        named_params = list(self.model.named_parameters())
        num_layers = len(named_params)    
        # Calculate phase shift increment across layers
        # Complete phase cycle (2π) distributed across layers
        phase_increment = 2 * math.pi / num_layers
        self.phase_shift = [i * phase_increment for i in list(range(1,num_layers+1))]
        # Clear existing parameter groups and create new ones
        self.optimizer.param_groups.clear()    
        param_groups = []
        param_group_names = []
        # Update learning rates
        for name, parameters in self.model.named_parameters():
            param_groups.append({'params': [parameters], 'lr': self.base_lr})
            param_group_names.append(name)
            self.optimizer.add_param_group({
                'params': parameters,
                'lr': self.base_lr
            })
        self.base_lrs = [self.base_lr for _ in self.optimizer.param_groups]

    def harmonic_cos(self, x_O, A, W, t, phi, alpha):
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
        x_t = max(x_O + (min(A, x_O) * math.cos((W * t) + phi)), alpha)
        return float(x_t)
            
    def get_lr(self):
        # Oscillation amplitude modulated by gradient norm
        gradient_factor = min(1.0, self.last_grad_norm * self.gradient_scale)
        self.oscillation_amplitude = self.base_lr*0.8
        W = (2 * math.pi) / (self.steps_per_epoch/self.oscillation_frequency)

        # Compute learning rate with oscillation
        lr_values = [
            self.harmonic_cos(
                x_O=self.base_lr, 
                A=self.oscillation_amplitude, 
                W=W,
                t=self._step_count, 
                phi=phi_shifted,
                alpha=self.alpha
            ) for phi_shifted,blr in zip(self.phase_shift, self.base_lrs)
        ]

        # Final learning rate with oscillation
        return lr_values
    
    def step(self):
        """
        Update learning rate step
        """        
        # In warm-up phase
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            # Linear warmup from min_lr to initial_lr
            warmup_factor = self.last_epoch / self.warmup_epochs
            self.base_lr = self.min_lr + (self.initial_lr - self.min_lr) * warmup_factor
            self.lr = self.base_lr
            # Increment step
            self._step_count += 1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
            return [self.lr for _ in self.base_lrs]
        else:
            # After warmup, calculate main schedule
            self.base_lr = self.base_lr - self.linear_step_decay
            self.base_lr = max(self.base_lr, self.min_lr)
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
        self.lr = self.get_lr()
        
        # Update learning rates
        #for param_group, lr in zip(self.optimizer.param_groups, self.lr):
        for param_group, lr in zip(self.optimizer.param_groups, self.lr):
            param_group['lr'] = lr

        # Rotate phases
        self.phase_shift = [self.phase_shift[-1]] + self.phase_shift[:-1]
        # Increment step
        self._step_count += 1
    
    def epoch_end(self):
        """
        Adjust oscillation frequency based on gradient behavior in the previous epoch.
        Called at the end of each epoch to set frequency for the next epoch.
        """
        self.steps_in_current_epoch = 0
        self.last_epoch += 1
        self.oscillation_amplitude = self.base_lr
        
        if not self.epoch_grad_norms:
            return  # No gradient information available yet
        
        # Calculate gradient statistics for the epoch
        avg_grad_norm = sum(self.epoch_grad_norms) / len(self.epoch_grad_norms)
        grad_variance = sum((g - avg_grad_norm) ** 2 for g in self.epoch_grad_norms) / len(self.epoch_grad_norms)
        
        # Normalize variance to a reasonable range (0.5 to 2.0)
        normalized_variance = min(2.0, max(0.5, 1.0 + grad_variance / (avg_grad_norm + 1e-8)))
        print(f"Epoch {self.last_epoch}: Normalized variance: {self.normalized_variance:.4f}")

        # Adjust frequency based on gradient behavior
        # Higher variance → more oscillations needed for exploration
        # Lower variance → fewer oscillations for exploitation
        #self.current_frequency = self.base_oscillations_per_epoch * normalized_variance / (1.0 + self.model_size_factor)
        
        # Reset for next epoch
        self.epoch_grad_norms = []
        
        if self.verbose:
            print(f"Epoch {self.last_epoch}: Setting oscillation frequency to {self.current_frequency:.4f}")
    
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
