setwd('/Users/Sushi/Desktop/申请MSDS/Course/628/Module2/even_sampled')
remove(list=ls())
library(dplyr);library(data.table);library(readr);library(stringr);library(tidytext)
library(pryr);library(tidyverse);library(widyr);library(gutenbergr);library(igraph)
library(ggraph);library(janeaustenr);library(rPython)

#prepare the data to be processed by 628module2_step1.py
test <- read_csv('./data/testval_data.csv')
test <- test %>% select(Id,text)
test <- test %>% mutate(text=gsub(pattern = "\\s+",x=text,replacement =" "))
test <- test %>% mutate(text=gsub(pattern = ",",x=text,replacement =" "))
test <- test %>% mutate(text=gsub(pattern = "\"",x=text,replacement =" "))
write_csv(test,"./data/raw_text_test.csv")
