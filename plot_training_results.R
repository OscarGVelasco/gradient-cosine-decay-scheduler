library(ggplot2)
library(plotly)
setwd("/omics/groups/OE0436/data/oscargvelasco/nCosAd_learning_rate_scheduler/")


training <- read.csv(file = "benchmark_ResNet_resnet50_CIFAR100_GradientCosineOOPScheduler.csv", header = T)
training <- training[!duplicated(training$step),]

training$step_cont <- (1:nrow(training)/nrow(training))*100
training$name = "GradientCosineOOP"
#
training2 <- read.csv(file = "benchmark_ResNet_resnet50_CIFAR100_GradientCosineOOPScheduler_validation.csv", header = T)
training2$step_cont <- training2$epoch
training2$name = "GradientCosineOOP_2"

plot_ly(data=training, x=~layer, y=~step, z=~lr,
        type="scatter3d", mode="markers", color=training$lr,size = 1)
tmp <- training %>% filter(layer==1)
tmp2 <- training %>% filter(layer==40)
tmp3 <- training %>% filter(layer==70)
tmp4 <- training %>% filter(layer==110)
ggplot(tmp[,c("step","lr")], aes(x=step, y=lr)) +
  geom_point(alpha=0.6, color="gray") + theme_minimal() +
  ggplot(tmp2[,c("step","lr")], aes(x=step, y=lr)) +
  geom_point(alpha=0.6, color="gray") + theme_minimal() +
  ggplot(tmp3[,c("step","lr")], aes(x=step, y=lr)) +
  geom_point(alpha=0.6, color="gray") + theme_minimal() +
  ggplot(tmp4[,c("step","lr")], aes(x=step, y=lr)) +
  geom_point(alpha=0.6, color="gray") + theme_minimal()

training_val <- read.csv(file = "benchmark_ResNet_resnet50_CIFAR100_GradientCosineOOPScheduler_validation.csv", header = T)
training_val$step_cont <- training_val$epoch
training_val$name = "validation_GradientCosineOOP"
tmp <- rbind(training[,c("lr","loss","accuracy","step_cont","name")], training_val[,c("lr","loss","accuracy","step_cont","name")])
ggplot(tmp, aes(x=step_cont, y = lr,  color=name)) +
  #geom_point(size=0.2, alpha=0.4) +
  geom_line(linewidth=0.6, alpha=.4) +
  theme_minimal() +
  theme(legend.key.width = unit(3,"cm")) +
  guides(linetype = guide_legend(override.aes = list(linewidth = 6)))

#

models <- c("benchmark_ResNet_resnet50_CIFAR100_LinearLR.csv"="benchmark_ResNet_resnet50_CIFAR100_LinearLR_validation.csv",
            "benchmark_ResNet_resnet50_CIFAR100_GradientCosineScheduler.csv"="benchmark_ResNet_resnet50_CIFAR100_GradientCosineScheduler_validation.csv",
  "benchmark_ResNet_resnet50_CIFAR100_CyclicLR.csv"="benchmark_ResNet_resnet50_CIFAR100_CyclicLR_validation.csv",
  "benchmark_ResNet_resnet50_CIFAR100_CosineAnnealingLR.csv"="benchmark_ResNet_resnet50_CIFAR100_CosineAnnealingLR_validation.csv",
  "benchmark_ResNet_resnet50_CIFAR100_ExponentialLR.csv"="benchmark_ResNet_resnet50_CIFAR100_ExponentialLR_validation.csv",
  "benchmark_ResNet_resnet50_CIFAR100_GradientCosineSchedulerAuto.csv"="benchmark_ResNet_resnet50_CIFAR100_GradientCosineSchedulerAuto_validation.csv")
#

#######
final <- list()
#for (i in 1:length(models)){
final <- lapply(1:length(models), function(i){
      res_val = models[i]
      res = names(models)[i]
      training <- read.csv(file = res, header = T)
      training_val <- read.csv(file = res_val, header = T)
      training$step_cont <- (1:nrow(training)/nrow(training))*200
      res <- strsplit(res, split = "benchmark_ResNet_resnet50_CIFAR100_")[[1]][2]
      res <- substr(res,1,nchar(res)-4)
      training$name = res
      training_val$step_cont <- training_val$epoch
      res_val <- strsplit(res_val, split = "benchmark_ResNet_resnet50_CIFAR100_")[[1]][2]
      res_val <- substr(res_val,1,nchar(res_val)-4)
      training_val$name <- res_val
      tmp <- rbind(training[,c("lr","loss","accuracy","step_cont","name")], training_val[,c("lr","loss","accuracy","step_cont","name")])
      final[[res]] = tmp
})
final <- do.call(rbind.data.frame, final)

final <- rbind(final, training[,c("lr","loss", "accuracy", "step_cont","name")])
final <- rbind(final_val, training[,c("lr","loss", "accuracy", "step_cont","name")])

colors_paired <- RColorBrewer::brewer.pal(name = "Paired", n = 10)
colors_paired <- c(colors_paired, "#000000","#BBBBBB")
names(colors_paired) <- unique(final$name)

ggplot(final, aes(x=step_cont, y = lr,  color=name)) +
  #geom_point(size=0.2, alpha=0.4) +
  geom_line(linewidth=0.6, alpha=.4) +
  theme_minimal() +
  scale_color_manual(values = colors_paired) +
  theme(legend.key.width = unit(3,"cm")) +
  guides(linetype = guide_legend(override.aes = list(linewidth = 6)))


ggplot(final, aes(x=step_cont, y = accuracy, fill=name, color=name)) +
  #geom_point(size=0.2, alpha=0.4) +
  geom_line(linewidth=0.8, alpha=.8) +
  theme_minimal() +
  scale_color_manual(values = colors_paired) +
  theme(legend.text = element_text(size=8), legend.key.size = unit(2,"line")) +
  geom_hline(yintercept = 49.44) +
  geom_vline(xintercept = 115)

final_val <- final[grep(pattern = "validation", final$name),]
final_val <- rbind(final_val, training_val[,c("lr","loss", "accuracy", "step_cont","name")])
#final_val <- rbind(final_val, training2[,c("lr","loss", "accuracy", "step_cont","name")])
a = final_val %>% group_by(name) %>% summarise(across(accuracy, max)) %>% arrange(accuracy)
a$name <- factor(a$name, levels = a$name)
ggplot(a, aes(x=name,y=accuracy, fill=name)) +
  geom_bar(stat = "identity")

ggplot(final_val, aes(x=step_cont, y = accuracy, fill=name, color=name)) +
  #geom_point(size=0.2, alpha=0.4) +
  geom_line(linewidth=0.8, alpha=.6) +
  theme_minimal() +
  scale_color_brewer(palette = "Set2") +
  theme(legend.text = element_text(size=8), legend.key.size = unit(2,"line")) +
  geom_hline(yintercept = 55.39) +
  geom_vline(xintercept = 239)

ggplot(rbind(training, training_gracos), aes(x=step_cont, y = lr, fill=name, color=name)) +
  geom_point(size=0.2, alpha=0.4) +
  geom_line(linewidth=0.2, alpha=.4) +
  theme_minimal()

ggplot(rbind(tmp, tmp2), aes(x=step_cont, y = accuracy, fill=name, color=name)) +
  geom_point(size=0.2, alpha=0.4) +
  geom_line(linewidth=0.2, alpha=.4) +
  theme_minimal()

ggplot(tmp, aes(x=step_cont, y = loss, fill=name, color=name)) +
  geom_point(size=0.2, alpha=0.4) +
  geom_line(linewidth=0.1, alpha=.3) +
  theme_minimal()

######## ConvNext
models <- c("benchmark_ResNext_resnext101_CIFAR100_GradientCosineScheduler_validation.csv",
              "benchmark_ResNext_resnext101_CIFAR100_GradientCosineOOPScheduler_validation.csv",
              "benchmark_ResNext_resnext101_CIFAR100_LinearLR_validation.csv",
            "benchmark_ResNext_resnext101_CIFAR100_ExponentialLR_validation.csv")
#######
final <- list()
#for (i in 1:length(models)){
final <- lapply(1:length(models), function(i){
  res_val = models[i]
  training_val <- read.csv(file = res_val, header = T)
  res_val <- strsplit(res_val, split = "benchmark_ResNext_resnext101_CIFAR100_")[[1]][2]
  res_val <- substr(res_val,1,nchar(res_val)-4)
  training_val$step_cont <- training_val$epoch
  training_val$name <- res_val
  tmp <- training_val[,c("lr","loss","accuracy","step_cont","name")]
  final[[res_val]] = tmp
})
final <- do.call(rbind.data.frame, final)

final %>% group_by(name) %>% summarise(across(accuracy, max))
# 34 acc
ggplot(final, aes(x=step_cont, y = accuracy, fill=name, color=name)) +
  #geom_point(size=0.2, alpha=0.4) +
  geom_line(linewidth=0.8, alpha=.6) +
  theme_minimal() +
  scale_color_brewer(palette = "Set2") +
  theme(legend.text = element_text(size=8), legend.key.size = unit(2,"line")) +
  geom_hline(yintercept = 49.5) +
  geom_vline(xintercept = 115)

