"""
File: GradientCosineScheduler
Author: Óscar González-Velasco, PhD
Email: oscargvelasco@gmail.com / oscar.gonzalezvelasco@dkfz-heidelberg.de
Date: 17 March 2025

This code provides a custom PyTorch learning rate scheduler that implements a Gradient-aware Cosine with linear decay learning rate scheduler:
- GraCos -

Learning Rate follows a Cosine oscillation within each epoch (LR is adjusted per step).
The Cosine oscillation pattern follows a descending linear trajectory towards a defined minimum learning rate: the equilibrium LR from which
the wave oscillates decreases steadily each epoch.
The <auto> mode incorporates a Gradient-based modulation - the amplitude and frequency of oscillations are affected by MAV gradient 
magnitudes (computed at the end of each epoch).

Cosine wave Documentation:
http://labman.phys.utk.edu/phys135core/modules/m9/harmonic_motion.html#:~:text=x(t)%20%3D%20A%20cos,return%20to%20the%20starting%20position.

Features:

Warm-up period: Optional linear warm-up for stable initial training
Base trajectory: Linear descent towards zero, with oscillations around this trajectory
Initial oscillation amplitude: default 1 = 100% of initial LR value.
For example, if base LR is 0.01, initial amplitud=lr -> maximum lr value per oscillation = 0.02

Phase offset for cosine function:
*Added phase_offset = math.pi / 2 to start the cosine at its lowest point
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
    def __init__(self, optimizer, total_epochs, steps_per_epoch, model, 
                 mode="constant", initial_lr=0.01, final_lr=0.00001, 
                 warmup_epochs=0, oscillation_frequency=4.0,
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
        # Minimum lr allowed:
        self.alpha = final_lr*0.1
        self.model = model
        self.mode = mode
        # Warm-up and init lr
        self.warmup_epochs = warmup_epochs
        # Base value of the oscillating wave
        self.base_lr = self.initial_lr
        # If no warmpup we start the wave at maximum x negative position
        self.phi=math.pi
        #  calling the constructor (initialization method) of the parent class that GradientCosineScheduler inherits from, which is PyTorch's _LRScheduler.
        #super(GradientCosineScheduler, self).__init__(optimizer, last_epoch, verbose)
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
        # Initial Oscillation parameters
        self.oscillation_amplitude = initial_lr*1
        #self.oscillation_frequency = oscillation_frequency
        self.oscillation_frequency = round(math.sqrt(self.steps_per_epoch))

        # Gradient norm tracking
        self.epoch_grad_norms = []
                
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
        self.base_lrs = [self.base_lr for _ in self.optimizer.param_groups]
        # Update learning rates
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

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
        W = (2 * math.pi) / (self.steps_per_epoch/self.oscillation_frequency)

        # Compute learning rate with oscillation
        lr_values = [
            self.harmonic_cos(
                x_O=self.base_lr, 
                A=self.oscillation_amplitude, 
                W=W, 
                t=self._step_count, 
                phi=self.phi,
                alpha=self.alpha
            ) for _ in self.base_lrs
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
        # Oscillation frequency adjusted by model size
        #current_frequency = self.oscillation_frequency / (1.0 + self.model_size_factor)   

        # Compute learning rate with oscillation
        self.lr = self.get_lr()
        
        # Update learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.lr):
            param_group['lr'] = lr
        
        # Increment step
        self._step_count += 1
    
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
        # Gradient-aware amplitude and freq. modulation
        if self.mode=="auto" and self.last_epoch>0:
            grad_factor = max(0.05, min(1.5, (1/math.exp(avg_grad_norm))))
            self.oscillation_amplitude = self.base_lr * grad_factor
            #self.oscillation_frequency = round(1/grad_factor)
            #self.oscillation_frequency = round(1/grad_factor) + int(math.log10(self.steps_per_epoch))
            self.oscillation_frequency = round(math.sqrt(self.steps_per_epoch))
            # IDEA
            #self.oscillation_frequency = round(math.sqrt(self.steps_per_epoch) / math.exp(avg_grad_norm))
        elif self.mode=="constant":
            self.oscillation_amplitude = self.base_lr * 1

        self.steps_in_current_epoch = 0
        self.last_epoch += 1

        print("Epoch Grad statistics:")
        print(f"Epoch {self.last_epoch}: Average grad norm: {avg_grad_norm:.4f}")
        print(f"Epoch {self.last_epoch}: Grad Normalized variance: {grad_variance:.4f}")
        print(f"Epoch {self.last_epoch}: Osc. Amplitude: {self.oscillation_amplitude:.4f}")
        print(f"Epoch {self.last_epoch}: Osc. Freq.: {self.oscillation_frequency:.4f}")
        # Reset grads for next epoch
        self.epoch_grad_norms = []
            
    def update_gradient_norm(self):
        """
        Update the gradient norm tracker based on current gradients.
        Call this method after loss.backward() but before optimizer.step()
        """            
        # More efficient version
        with torch.no_grad():  # Prevents tracking unnecessary computations
            total_norm = 0.0
            param_count = 0
            # Vectorized approach for all parameters at once
            all_grads = [p.grad.view(-1) for param_group in self.optimizer.param_groups
                         for p in param_group['params'] if p.grad is not None]           
            if all_grads:  # Check if there are any gradients
                all_grads_cat = torch.cat(all_grads)
                total_norm = all_grads_cat.norm(2).item() ** 2
                param_count = len(all_grads)
                avg_norm = math.sqrt(total_norm / param_count)
        self.epoch_grad_norms.append(avg_norm)
        return avg_norm
