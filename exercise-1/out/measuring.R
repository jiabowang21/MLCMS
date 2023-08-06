d61 <- read.csv('6(120, 147, 6)_records.csv', header=FALSE)
d62 <- read.csv('6(270, 147, 6)_records.csv', header=FALSE)
d63 <- read.csv('6(270, 153, 6)_records.csv', header=FALSE)

d41 <- read.csv('4(120, 147, 6)_records.csv', header=FALSE)
d42 <- read.csv('4(270, 147, 6)_records.csv', header=FALSE)
d43 <- read.csv('4(270, 153, 6)_records.csv', header=FALSE)

d21 <- read.csv('2(120, 147, 6)_records.csv', header=FALSE)
d22 <- read.csv('2(270, 147, 6)_records.csv', header=FALSE)
d23 <- read.csv('2(270, 153, 6)_records.csv', header=FALSE)

d61 <- as.data.frame(table(as.data.frame(as.integer(d61 / 3))))
d62 <- as.data.frame(table(as.data.frame(as.integer(d62 / 3))))
d63 <- as.data.frame(table(as.data.frame(as.integer(d63 / 3))))

d41 <- as.data.frame(table(as.data.frame(as.integer(d41 / 3))))
d42 <- as.data.frame(table(as.data.frame(as.integer(d42 / 3))))
d43 <- as.data.frame(table(as.data.frame(as.integer(d43 / 3))))

d21 <- as.data.frame(table(as.data.frame(as.integer(d21 / 3))))
d22 <- as.data.frame(table(as.data.frame(as.integer(d22 / 3))))
d23 <- as.data.frame(table(as.data.frame(as.integer(d23 / 3))))

data <- merge(d61, d62, by = 'Var1', all = TRUE)
colnames(data) <- c('Var1', 'density = 6, measuring point 1', 'density = 6, measuring point 2')
data <- merge(data, d63, by = 'Var1', all = TRUE)
names(data)[names(data) == 'Freq'] <- 'density = 6, measuring point 3'
data <- merge(data, d41, by = 'Var1', all = TRUE)
names(data)[names(data) == 'Freq'] <- 'density = 4, measuring point 1'
data <- merge(data, d42, by = 'Var1', all = TRUE)
names(data)[names(data) == 'Freq'] <- 'density = 4, measuring point 2'
data <- merge(data, d43, by = 'Var1', all = TRUE)
names(data)[names(data) == 'Freq'] <- 'density = 4, measuring point 3'
data <- merge(data, d21, by = 'Var1', all = TRUE)
names(data)[names(data) == 'Freq'] <- 'density = 2, measuring point 1'
data <- merge(data, d22, by = 'Var1', all = TRUE)
names(data)[names(data) == 'Freq'] <- 'density = 2, measuring point 2'
data <- merge(data, d23, by = 'Var1', all = TRUE)
names(data)[names(data) == 'Freq'] <- 'density = 2, measuring point 3'


names(data)[names(data) == 'Var1'] <- 'time'


library(reshape2)
library(ggplot2)
library(tidyverse)

data_long <- melt(data, id="time") 
data_long <- data_long[complete.cases(data_long), ]
data_long$time <- as.numeric(data_long$time)

## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
ggplot(data=data_long,
       aes(x=time, y=value, colour=variable)) +
  geom_line(aes(linetype=variable), lwd = 1.5) + geom_point(size = 3) +
  scale_linetype_manual(values=c(1,1,1,5,5,5,4,4,4))+
  labs(x = "Time (s)", y = "Flow (P/(m·s))")

