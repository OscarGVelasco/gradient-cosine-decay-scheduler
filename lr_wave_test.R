x_O=0.01
x_F=0.0001
A=x_O/2
steps=100
alpha=((x_O-x_F)*(1-((steps-1)/steps)))
alpha=((x_O-x_F)*(1-((200-1)/200)))

freq=4
omega=0.0001
W=(2*pi)/(steps/freq)
t=1
phi=pi
lrs <- c()
for (i in 1:(steps*2)){
  #if(i>10)W=(2*pi)/(steps/sample(c(2,3,4,5,6)))
  t=i
  x_t = x_O + (A * cos((W * t) + phi))
  x_t = max(x_t, omega)
  print(paste("lr:",x_t," t=",i))
  lrs <- c(lrs, x_t)
  x_O = x_O - alpha
  A = x_O/2
}

plot(data.frame(t=1:steps, lrs))

plot(data.frame(t=1:200, lrs))


lr=0.01
final_lr=0.0001
alpha=((lr-final_lr)*(1-((steps-1)/steps)))
for (i in 1:steps){
  lr = lr - alpha
  print(paste(i,": lr:",lr))
}

lr==final_lr
lr

last_epoch=0
warmup_epochs=5
for(i in 0:warmup_epochs){
  last_epoch=i
  warmup_factor = last_epoch / warmup_epochs
  actual_lr = alpha + (lr - alpha) * warmup_factor
  print(actual_lr)
}

