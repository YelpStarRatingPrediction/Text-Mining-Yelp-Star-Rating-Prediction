remove(list=ls())
library(dplyr);library(data.table);library(readr);library(stringr);library(tidytext)
library(pryr);library(tidyverse);library(widyr);library(gutenbergr);library(igraph)
library(ggraph);library(janeaustenr);library(rPython);library(data.table);library(Matrix)
library(glmnet);library(randomForest)

##### read the data processed by 628module2_step1.py#######
time1 <- Sys.time()
test <- read_csv('./data/testval_data.csv')
test_text <- read_csv('./data/test_text.csv')
phrase_united <- read_csv("./data/phrase_united.csv")
phrase <- read_csv("phrase.csv")
test <- test %>% select(-text) %>% left_join(test_text)
test$text[is.na(test$text)] <- ""
for(i in 1:nrow(phrase)){
  test <- test %>% 
    mutate(text=gsub(pattern = phrase_united$phr[i],x=text,
                     replacement = paste(phrase$word1[i],"_",phrase$word2[i],sep = ""))
    )
}
time2 <- Sys.time() #1.058916 hours
time2-time1
write.csv(test,"test_clean.csv",row.names = FALSE)

remove(list = ls())
test_clean <- read_csv("./data/test_clean.csv")
words_score <- read_csv("./data/words_score.csv")
getscores <- function(x,num){
  #x = train_sample$text[1]
  t <- tibble(id=1,text=x)
  tt <- t %>% unnest_tokens(word, text) %>% anti_join(stop_words,by="word")
  tt <- tt %>% left_join(words_score,by="word")
  scores <- tt$stars[!is.na(tt$stars)]
  score_len <- length(scores)
  if(score_len==0){
    return(rep(NA,8)[num])
  }
  slope <- 0
  if(score_len>2){
    x <- 1:score_len
    y <- scores
    slope <- lm(y~x)$coefficients[2]
  }
  scores.sign <- scores>0
  scores.nodes <- c(0,diff(scores.sign))!=0
  first_turn <- 0;last_turn <- 0
  if(score_len==1){
    scores.nodes[1]==TRUE
  }
  for(k in 1:score_len){
    if(scores.sign[min(k+1,score_len)]){
      first_turn <- k
      break
    }
  }
  for(k in score_len:1){
    if(scores.sign[k]){
      last_turn <- k
      break
    }
  }
  total_nodes <- sum(scores.nodes)
  ave_score <- mean(scores)
  sd_score <- sd(scores)
  first_mean <- mean(scores[1:first_turn])
  last_mean <- mean(scores[last_turn:score_len])
  first_ratio <- first_turn/score_len
  last_ratio <- (score_len-last_turn+1)/score_len
  r <- c(ave_score,sd_score,first_mean,last_mean,first_ratio,last_ratio,total_nodes,slope)
  return(r[num])
}
test_clean_features <- matrix(numeric(nrow(test_clean)*8),nrow=nrow(test_clean))
for(i in 1:nrow(test_clean_features)){
  test_clean_features[i,] <- getscores(test_clean$text[i],1:8)
  print(i)
}
colnames(test_clean_features) <- c("ave_score","sd_score","first_mean","last_mean",
                                   "first_ratio","last_ratio","total_nodes","slope")
test_clean <- cbind(test_clean,test_clean_features)
test_clean[is.na(test_clean)] <- 0
test_clean$words <- sapply(str_split(test_clean$text,pattern = " "),length)
write.csv(test_clean,"./data/test_clean.csv",row.names = FALSE)
