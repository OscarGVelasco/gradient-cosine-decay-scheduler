# gradient-cosine-decay-scheduler

This code provides a custom PyTorch learning rate scheduler that implements a Gradient-aware Cosine with linear decay 
learning rate scheduler:
< GraCos >

### Versions: 

- GradientCosineScheduler: Learning Rate follows a Cosine that oscillates within each epoch (LR is adjusted per step). Two modes *normal* and *auto*.

- GradientCosineOOPScheduler: Modified Out Of Phase (OOP) version of the GradientCosineScheduler. OOP version will produce an Out Of Phase Cosine wave across layers (each layer will be slightly out of phase from the previos layer). This means that each layer will have an independent LR value, that also will oscillate across steps using a cosine wave function, out of phase in contrast with the rest of the layers.

# The Scheduler:

The Cosine oscillation pattern follows a descending linear trajectory towards a defined minimum learning rate: the equilibrium LR (base_lr) from which the wave oscillates decreases steadily each epoch. *Normal* mode will use an amplitud = base_lr*1 and a frequency = square root of n. steps per epoch.
 
The *auto* mode incorporates a Gradient-based modulation - the amplitude and frequency of oscillations are affected by MAV gradient magnitudes (computed at the end of each epoch).

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

# Use during training

Pseudocode:

```
scheduler.INITIATE

for i in N_EPOCH:
    for j in N_STEPS:

        OUTPUTS = model.TRAINING(inputs)
        LOSS = model.COMPUTE_LOSS(outputs)
        
        # Backward pass
        optimizer.COMPUTE_GRADIENTS()
        model.BACKWARD_PASS()
        
        # Track gradient norm
        scheduler.UPDATE_GRADIENT_NORM()
        
        # Optimizer step
        optimizer.STEP()
        
        # Scheduler step
        # update LR amplitude
        scheduler.STEP()
    
    # Scheduler EPOCH END
    # Update LR amplitude and frequency
    # Apply base linear decay
    scheduler.EPOCH()
```