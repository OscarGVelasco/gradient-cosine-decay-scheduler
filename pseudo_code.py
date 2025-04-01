

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
    # Update LR frequency
    # Apply base linear decay
    scheduler.EPOCH()