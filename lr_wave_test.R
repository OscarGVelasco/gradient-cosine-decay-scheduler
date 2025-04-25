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


lr=0.1
alpha=((0.1-0.01)*(1-(99/100)))
for (i in 1:100){
  lr = lr - alpha
  print(paste(i,": lr:",lr))
}


##########################
# Multilayer approach
library(plotly)
n_layers=10
phase_increment = 2*pi / n_layers
for (i in 1:n_layers){
  phase_shift[[i]] = i * phase_increment
  print(phase_shift)
}
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
lrs <- data.frame()
for (i in 1:(steps*2)){
  #if(i>10)W=(2*pi)/(steps/sample(c(2,3,4,5,6)))
  t=i
  for(j in 1:n_layers){
    x_t = x_O + (A * cos((W * t) + phase_shift[j]))
    x_t = max(x_t, omega)
    print(paste("lr:",x_t," t=",i))
    lrs <- rbind.data.frame(lrs, c(x_t,i,j))
  }
  phase_shift <- c(phase_shift[length(phase_shift)], phase_shift[-length(phase_shift)])
  x_O = x_O - alpha
  A = x_O/2
}
colnames(lrs) <- c("lr", "step", "layer")
head(lrs)

plot_ly(data=lrs, x=~layer, y=~step, z=~lr,
        type="scatter3d", mode="markers", color=lrs$lr,size = 1)

plot(lrs[lrs$step==1,c(3,1)])
plot(lrs[lrs$step==2,c(3,1)])
plot(lrs[lrs$layer==1,c(2,1)])
plot(lrs[lrs$layer==5,c(2,1)])
