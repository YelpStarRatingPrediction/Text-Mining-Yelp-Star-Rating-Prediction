remove(list=ls())
library(dplyr);library(data.table);library(readr);library(stringr);library(tidytext)
library(pryr);library(tidyverse);library(widyr);library(gutenbergr);library(igraph)
library(ggraph);library(janeaustenr);library(rPython);library(data.table);library(Matrix)
library(glmnet);library(randomForest);library(tm)
library(e1071);library(nnet);
library(rpart);library(rpart.plot);library(MASS);library(RColorBrewer)
###Read data####
test_clean <- read_csv("./data/test_clean.csv") #the test_val dataset
test_val <- read_csv("./data/train_val.csv") #the labeled validition dataset extracted from train set
train_sample <- read_csv("./data/train_sample.csv") #the balanced. training data.
#### power transformation ####
power_trans <- function(x){
  x$ave_score2 <- x$ave_score^2
  x$ave_score3 <- x$ave_score^3
  x$min_score2 <- x$min_score^2
  x$min_score3 <- x$min_score^3
  x$max_score2 <- x$max_score^2
  x$max_score3 <- x$max_score^3
  x$words_log <- log(x$words)
  x$slope_power <- sign(x$slope)*abs(x$slope)^0.5
  return(x)
}

train_sample <- power_trans(train_sample)
test_clean <- power_trans(test_clean)
test_val <- power_trans(test_val)

####Visualization####
theme1 <- theme_light() + 
  theme(plot.title = element_text(size=30,colour = "black",face = "bold",hjust = 0.5),
        axis.title = element_text(size=25,colour = "black"),
        axis.text.y= element_text(size=20),
        axis.text.x= element_text(size=20),
        legend.position="bottom")
#Histogram
hist(train_sample$ave_score,breaks = 3e3)
hist(train_sample$slope,breaks = 3e2)
hist(train_sample$ave_score,breaks = 1e3)
#mean score vs y
ggplot(train_sample, aes(stars, ave_score)) +
  geom_boxplot(aes(group = stars,fill=factor(stars)),notch=TRUE,outlier.shape= NA,show_guide = FALSE)+
  scale_y_continuous(limits = quantile(train_sample$ave_score, c(0.25, 0.75))) + theme1 +
  labs(title="Average Score by Star",y="Average Score",x="Star")
ggsave("./image/ave_score.png",dpi = 600,width=7,height=9)
#min score vs y
ggplot(train_sample, aes(stars, min_score)) +
  geom_boxplot(aes(group = stars,fill=factor(stars)),notch=TRUE,outlier.shape= NA,show_guide = FALSE)+
  scale_y_continuous(limits = quantile(train_sample$min_score, c(0.25, 0.75))) + theme1 +
  labs(title="Minimum Score by Star",y="Minimum Score",x="Star")
ggsave("./image/min_score.png",dpi = 600,width=7,height=9)
#max score vs y
ggplot(train_sample, aes(stars, max_score)) +
  geom_boxplot(aes(group = stars,fill=factor(stars)),notch=TRUE,outlier.shape= NA,show_guide = FALSE)+
  scale_y_continuous(limits = quantile(train_sample$max_score, c(0.25, 0.75))) + theme1 +
  labs(title="Maximum Score by Star",y="Maximum Score",x="Star")
ggsave("./image/max_score.png",dpi = 600,width=7,height=9)
#slope vs y
train_sample$slope_power <- sign(train_sample$slope)*abs(train_sample$slope)^0.5
ggplot(train_sample, aes(stars, slope_power)) +
  geom_boxplot(aes(group = stars,fill=factor(stars)),notch=TRUE,outlier.shape= NA,show_guide = FALSE)+
  scale_y_continuous(limits = quantile(train_sample$slope_power, c(0.25, 0.75))) + 
  theme1 +
  labs(title="Root Slope by Star",y="Root Slope",x="Star")
ggsave("./image/root_slope.png",dpi = 600,width=15,height=9)


train_sample <- train_sample %>% rename(Ave=ave_score,Ave2=ave_score2,Ave3=ave_score3,
                                        Min=min_score,Min2=min_score2,Min3=min_score3,
                                        Max=max_score,Max2=max_score2,Max3=max_score3,
                                        Slope=slope_power)
test_clean <- test_clean %>% rename(Ave=ave_score,Ave2=ave_score2,Ave3=ave_score3,
                                        Min=min_score,Min2=min_score2,Min3=min_score3,
                                        Max=max_score,Max2=max_score2,Max3=max_score3,
                                        Slope=slope_power)
test_val <- test_val %>% rename(Ave=ave_score,Ave2=ave_score2,Ave3=ave_score3,
                                        Min=min_score,Min2=min_score2,Min3=min_score3,
                                        Max=max_score,Max2=max_score2,Max3=max_score3,
                                        Slope=slope_power)
####Regression Tree####
Formula <- stars~Ave+Ave2+Ave3+Min+Min2+Min3+Max+Max2+Max3+Slope
tc <- rpart.control(minsplit=35000,minbucket=15000,maxdepth=4,xval=10,cp=0.0001)
rpart.mod=rpart(Formula,data=train_sample,method="anova",control = tc)
rpart.mod.pru<-prune(rpart.mod, cp= rpart.mod$cptable[which.min(rpart.mod$cptable[,"xerror"]),"CP"]) 
yhat <- as.numeric(predict(rpart.mod.pru,test_val))
sqrt(mean((yhat-test_val$stars)^2))
rpart.plot(rpart.mod.pru, main="Decision Tree of Star Rating (RMSE:0.877)",type = 3,cex=.7)

