setwd('/Users/Sushi/Desktop/申请MSDS/Course/628/Module2/even_sampled')
remove(list=ls())
library(dplyr);library(data.table);library(readr);library(stringr);library(tidytext)
library(pryr);library(tidyverse);library(widyr);library(gutenbergr);library(igraph)
library(ggraph);library(janeaustenr);library(rPython);library(data.table);library(Matrix)
library(glmnet);library(randomForest)

##### evenly sample the data#####
train <- read_csv('./data/train_data.csv')
train$rowID <- 1:nrow(train)
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
train <- rbind(train1_sampled,train2_sampled,
                     train3_sampled,train4_sampled,train5_sampled)
remove(train1_sampled,train2_sampled,train3_sampled,train4_sampled,train5_sampled)
remove(train1,train2,train3,train4,train5)
##### read the data processed by python#######
processed_text <- read_csv('./data/processed_text.csv')
train <- train %>% select(-text) %>% right_join(processed_text,by=c("rowID"="row_number"))
remove(processed_text)
##### Data cleaning#####
#Count the number of words
train$words <- sapply(str_split(train$text,pattern = " "),length)
#Check the category value
train <- train %>% mutate(isRestaurants=!is.na(str_match(categories,"Restaurants")))
sum(!train$isRestaurants) #sum equals to 0, all are restaurants.
#Summarize the table into businesses
byBusiness <- train %>% group_by(longitude,latitude,city,name) %>% 
  summarise(comments=n(),mean_stars=mean(stars),minDate=min(as.Date(date)),maxDate=max(as.Date(date))) %>% 
  mutate(duration=maxDate-minDate) #add the minimum and maximum date, and their difference
#get the quantile of #comments for each businesses.
quantile(byBusiness$comments,c(0,1/4,1/2,3/4,1))
#histogram for #comments for each businesses.
hist(byBusiness$comments,breaks=1e3)
#histogram for average star for each businesses.
hist(byBusiness$mean_stars,breaks=1e3)
#histogram for the difference of minimum and maximum comment date for each businesses.
hist(as.numeric(byBusiness$duration),breaks=1e2)
# number of businesses with difference of minimum and maximum comment date less than 14. (>0)
byBusiness %>% filter(as.numeric(duration)<=14) %>% nrow() #8042
#clean the train data by exclude those businesses with #comments less than 3
#and difference of minimum and maximum comment date less than 14.
byBusiness <- byBusiness %>% mutate(id=paste('lon:',longitude,'lat:',latitude,sep=""))
byBusiness_clean <- byBusiness %>% filter(comments>3,as.numeric(duration)>14) %>% 
  ungroup() %>% select(id,comments,duration) #24582/42735-1
# add id for train dataset for convenience of merging
train <- train %>% mutate(id=paste('lon:',longitude,'lat:',latitude,sep=""))
#comments with less than 1 words are considered to be not serious
train <- train %>% filter(words>0,id %in% byBusiness_clean$id)
#get the quantile of #words for each stars
words <- matrix(numeric(6*5),nrow = 6)
words[1,] <- quantile(unlist(train %>% select(words)),c(0,1/4,1/2,3/4,1))
words[2,] <- quantile(unlist(train %>% filter(stars==1) %>% select(words)),c(0,1/4,1/2,3/4,1))
words[3,] <- quantile(unlist(train %>% filter(stars==2) %>% select(words)),c(0,1/4,1/2,3/4,1))
words[4,] <- quantile(unlist(train %>% filter(stars==3) %>% select(words)),c(0,1/4,1/2,3/4,1))
words[5,] <- quantile(unlist(train %>% filter(stars==4) %>% select(words)),c(0,1/4,1/2,3/4,1))
words[6,] <- quantile(unlist(train %>% filter(stars==5) %>% select(words)),c(0,1/4,1/2,3/4,1))
row.names(words) <- c("all","star1","star2","star3","star4","star5")
colnames(words) <- c("0%","25%","50%","75%","100%")
words
#histogram for #words overall
hist(train$words,breaks = 1e2)
#histogram for #comments for each stars.
hist(train$stars)

####explore the phrases#####
train <- train %>% mutate(commendid=row.names(train))
phrase <- train %>% select(commendid,text) %>% unnest_tokens(phr, text, token = "ngrams", n = 2)
phrase <- phrase %>% count(phr,sort=TRUE)
data(stop_words) #load stop words and exclude them later
#choose the words within the sentiments words dictionary and not in the stopwords
phrase <- phrase %>% separate(phr, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word) %>% filter(!word2 %in% stop_words$word) %>% 
  filter(word1 %in% sentiments$word | word2 %in% sentiments$word)
phrase <- phrase %>% mutate(id=row.names(phrase))
hist(unlist(phrase %>% filter(n>70) %>% select(n)),breaks = 1e4)
# 99.7% of the phrase occurs less than 900 times.
quantile(phrase$n,c(0,1/4,2/4,3/4,0.9995,1))
phrase %>% filter(n>500) %>% nrow()
phrase_graph <- phrase %>% filter(n>1200) %>% graph_from_data_frame()
a <- grid::arrow(type = "closed", length = unit(.12, "inches"))
set.seed(12345)
ggraph(phrase_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                 arrow = a, end_cap = circle(.1, 'inches')) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 0.1,size=6) +
  theme_void()
ggsave("./image/phrase.jpg",width = 33 ,height = 20,units = "cm",dpi=300)
phrase <- phrase %>% filter(n > 500)
# in order to treat these phrases as a words, we need to add a "_" to link them together.
phrase_united <- phrase %>% unite(phr, word1, word2, sep = " ")
for(i in 1:nrow(phrase)){
  train <- train %>% 
    mutate(text=gsub(pattern = phrase_united$phr[i],x=text,
                     replacement = paste(phrase$word1[i],"_",phrase$word2[i],sep = ""))
    )
}
#load("phrase.RData")
#### Diversity an IDF####
words.data <- train %>% select(commendid,text,stars) %>% 
  unnest_tokens(word,text) %>% anti_join(stop_words)
words.data <- words.data %>% count(commendid,stars,word, sort = TRUE) #columes: coomendid,stars,word,n
words.data %>% summarise(countall=sum(n)) 
words_stars <- words.data %>% group_by(stars,word) %>% summarise(n=sum(n)) #columes: stars,word,n
words_stars <- words_stars %>% bind_tf_idf(word, stars, n)
words_count <- words_stars %>% group_by(word) %>% summarise(n=sum(n)) %>% arrange(desc(n)) #columes: word,n
#exlcude those uncommon words which are meaningless for future prediction.
words.data <- words.data %>% inner_join(words_count %>% filter(n>=40) %>% select(word))
words_stars <- words.data %>% group_by(stars,word) %>% summarise(n=sum(n)) %>% ungroup() %>% bind_tf_idf(word, stars, n)
words_count <- words_stars %>% group_by(word) %>% summarise(n=sum(n)) %>% arrange(desc(n)) %>% ungroup()
### TF-IDF
words.data <- words.data %>% bind_tf_idf(word, commendid, n)
#convert data into format by word
byword <- words_stars %>% select(-tf,-idf,-tf_idf) %>% spread(key = stars, value = n) %>% 
  left_join(words.data %>% select(-tf,-tf_idf) %>% group_by(word) %>% summarise(idf=mean(idf))) #columes: word,star1,star2,star3,star4,star5,idf
byword <- byword %>% left_join(
  words_stars %>% select(-tf_idf,-tf) %>% 
    group_by(word) %>% summarise(idf_star=mean(idf)) %>% select(word,idf_star)
)
#below shows a normal distribution
#hist(log(unlist(byword %>% filter(tf_idf>0) %>% select(tf_idf))),
#     breaks = 1e3,main="Histogram of log TF-IDF",xlab="",freq=TRUE)
colnames(byword)[2:6] <- c("star1","star2","star3","star4","star5")
byword[is.na(byword)] <- 0
# convert the numbers into column (a sprcified star) frequency
for(i in 2:6){
  byword[,i] <- byword[,i]/sum(byword[,i])
}
# define diversity as standard deviation over mean
diversity <- function(x){
  return(sd(x)/mean(x))
}
byword$diversity <- apply(byword[,2:6],1,diversity)
byword <- byword %>% arrange(desc(diversity))
# histogram of diversity for all words
hist(byword$diversity,breaks = 1e2)
hist(log(byword$diversity),breaks = 1e2)
byword %>% arrange(desc(idf)) %>% head(20)
byword %>% arrange(idf) %>% head(20)
byword %>% arrange(desc(diversity)) %>% head(20)
byword %>% arrange(diversity) %>% head(20)
t.intercept <- 6
t.slope <- 4.5
t.vertical <- 0.4
#scatter plot of diversity vs idf, the words fall into the bottom-right area would be included
theme1 <- theme_light() + 
  theme(plot.title = element_text(size=30,colour = "black",face = "bold",hjust = 0.5),
        axis.title = element_text(size=25,colour = "black"),
        axis.text.y= element_text(size=20),
        axis.text.x= element_text(size=20),
        legend.position="bottom")
ggplot(byword,aes(x=diversity,y=idf))+geom_point(alpha=0.2,size=2) + theme1 +
  labs(title = "Diversity vs IDF",y = "IDF",x="Diversity") + 
  geom_abline(slope = t.slope,intercept = t.intercept,color="red",size=1.5) +
  geom_vline(xintercept=t.vertical,color="red",size=1.5)
ggsave("./image/diversity_idf.png",dpi = 600,width=15,height=9)
#   bottom_right words, the best words for classification
bottom_right_star1 <- unlist(byword %>% select(word,star1,idf,diversity) %>% 
                               filter((idf < diversity*t.slope+t.intercept) & (diversity>t.vertical)) %>% 
                               arrange(desc(star1)) %>% select(word))
bottom_right_star2 <- unlist(byword %>% select(word,star2,idf,diversity) %>% 
                               filter((idf < diversity*t.slope+t.intercept) & (diversity>t.vertical)) %>% 
                               arrange(desc(star2)) %>% select(word))
bottom_right_star3 <- unlist(byword %>% select(word,star3,idf,diversity) %>% 
                               filter((idf < diversity*t.slope+t.intercept) & (diversity>t.vertical)) %>% 
                               arrange(desc(star3)) %>% select(word))
bottom_right_star4 <- unlist(byword %>% select(word,star4,idf,diversity) %>% 
                               filter((idf < diversity*t.slope+t.intercept) & (diversity>t.vertical)) %>% 
                               arrange(desc(star4)) %>% select(word))
bottom_right_star5 <- unlist(byword %>% select(word,star5,idf,diversity) %>% 
                               filter((idf < diversity*t.slope+t.intercept) & (diversity>t.vertical)) %>% 
                               arrange(desc(star5)) %>% select(word))
bottom_right <- data.frame(star1=bottom_right_star1,star2=bottom_right_star2,
                           star3=bottom_right_star3,star4=bottom_right_star4,star5=bottom_right_star5)
remove(bottom_right_star1,bottom_right_star2,bottom_right_star3,bottom_right_star4,bottom_right_star5)
words.data %>% filter(word %in% bottom_right$star1) %>% group_by(commendid) %>% summarise(n=n()) %>% nrow
write.csv(bottom_right,"./data/bottom_right.csv",row.names = FALSE)

## save data
write.csv(phrase,"./data/phrase.csv",row.names = FALSE)
write.csv(phrase_united,"./data/phrase_united.csv",row.names = FALSE)
write.csv(bottom_right,"./data/bottom_right.csv",row.names = FALSE)
write.csv(byword,"./data/byword.csv",row.names = FALSE)

## word score measure
remove(byBusiness,byBusiness_clean)
set.seed(20180224)
val_row <- sample(1:nrow(train),nrow(train)-400000)
train_sample <- train[-val_row,]
train_val <- train[val_row,]
remove(train)
words.data_train <- words.data %>% filter(commendid %in% train_sample$commendid)
words_score <- words.data_train %>% group_by(word) %>% summarise(stars=mean(stars)) %>% ungroup()
words_score$stars <- as.vector(scale(words_score$stars))
words_score <- words_score %>% arrange(desc(stars))

theme1 <- theme_light() + 
  theme(plot.title = element_text(size=40,colour = "black",face = "bold",hjust = 0.5),
        axis.title = element_text(size=35,colour = "black"),
        axis.text.y= element_text(size=20),
        axis.text.x= element_text(size=20),
        legend.position="bottom")
ggplot(words_score,aes(x=stars))+geom_histogram(bins = 100) + theme1 +
  labs(title = "Histogram of Word Score",x="Word Score",y="")
ggsave("./image/Histogram1.png",dpi = 600,width=15,height=9)
words_score <- words_score %>% filter(word %in% bottom_right$star1)
ggplot(words_score,aes(x=stars))+geom_histogram(bins = 100) + theme1 +
  labs(title = "Histogram of Word Score After Cleaning",x="Word Score",y="")
ggsave("./image/Histogram2.png",dpi = 600,width=15,height=9)
remove(words.data_train)
write.csv(words_score,"./data/words_score.csv",row.names = FALSE)

#####Get the final features####
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
train_sample_features <- matrix(numeric(nrow(train_sample)*8),nrow=nrow(train_sample))
train_val_features <- matrix(numeric(nrow(train_val)*8),nrow=nrow(train_val))
for(i in 1:nrow(train_sample_features)){
  train_sample_features[i,] <- getscores(train_sample$text[i],1:8)
  print(i)
}
colnames(train_sample_features) <- c("ave_score","sd_score","first_mean","last_mean",
                                     "first_ratio","last_ratio","total_nodes","slope")
train_sample <- cbind(train_sample,train_sample_features)
train_sample[is.na(train_sample)] <- 0
write.csv(train_sample,"./data/train_sample.csv",row.names = FALSE)
for(i in 1:nrow(train_val_features)){
  train_val_features[i,] <- getscores(train_val$text[i],1:8)
  print(i)
}
colnames(train_val_features) <- c("ave_score","sd_score","first_mean","last_mean",
                                     "first_ratio","last_ratio","total_nodes","slope")
train_val <- cbind(train_val,train_val_features)
train_val[is.na(train_val)] <- 0
write.csv(train_val,"./data/train_val.csv",row.names = FALSE)
