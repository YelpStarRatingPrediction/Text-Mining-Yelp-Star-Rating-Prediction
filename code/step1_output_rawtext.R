remove(list=ls())
library(dplyr);library(data.table);library(readr);library(stringr);library(tidytext)
library(pryr);library(tidyverse);library(widyr);library(gutenbergr);library(igraph)
library(ggraph);library(janeaustenr);library(rPython)

#prepare the data to be processed by 628module2_step1.py
train <- read_csv('./data/train_data.csv')
train$rowID <- 1:nrow(train)
train <- train %>% select(rowID,stars,text)
train <- train %>% mutate(text=gsub(pattern = "\\s+",x=text,replacement =" "))
train <- train %>% mutate(text=gsub(pattern = ",",x=text,replacement =" "))
train <- train %>% mutate(text=gsub(pattern = "\"",x=text,replacement =" "))
#The y is not balanced
theme1 <- theme_light() + 
  theme(plot.title = element_text(size=30,colour = "black",face = "bold",hjust = 0.5),
        axis.title = element_text(size=25,colour = "black"),
        axis.text.y= element_text(size=20),
        axis.text.x= element_text(size=20),
        legend.position="bottom")
star_num <- train %>% select(stars) %>% group_by(stars) %>% summarise(n=n()) %>% ungroup()
ggplot(star_num, aes(x = stars, y = n, group = factor(1))) + 
  geom_bar(stat = "identity") + theme1 +
  labs(title="Bar Plot of #Observations of Each Star")
ggsave("./image/uneven.png",dpi = 600,width=15,height=9)


train1 <- train %>% filter(stars==1)
train2 <- train %>% filter(stars==2)
train3 <- train %>% filter(stars==3)
train4 <- train %>% filter(stars==4)
train5 <- train %>% filter(stars==5)
set.seed(20180221)
train1_sampled <- train1[sample(1:nrow(train1),100000),]
train2_sampled <- train2[sample(1:nrow(train2),100000),]
train3_sampled <- train3[sample(1:nrow(train3),100000),]
train4_sampled <- train4[sample(1:nrow(train4),100000),]
train5_sampled <- train5[sample(1:nrow(train5),100000),]
sample_data <- rbind(train1_sampled,train2_sampled,
                     train3_sampled,train4_sampled,train5_sampled)
write_csv(sample_data,"./data/raw_text.csv")
