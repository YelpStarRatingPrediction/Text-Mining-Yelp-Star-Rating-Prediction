# Text-Mining-Yelp-Star-Rating-Prediction
[Guanxu Su](https://github.com/GuanxuSu), [Shurong Gu](https://github.com/JacquelineGu), [Yuwei Sun](https://github.com/YuweiS)

This is a course project aims at finding out what makes a review positive or negative based on the review and a small set of attributes and proposing a prediction model to predict the ratings of reviews based on the text.
predicting review stars based on yelp review text features

There are 4 files in this repository: ***data***, ***image*** and ***code*** and **summary.ipynb**.
## **data** 

There is a file size limit of 100MB on github. Hence we put the large files together with all other relavent files at the google drive link: https://drive.google.com/open?id=1GTTO_KAm55bn2m3ZsT2xsLjwOyxv8Kvg

## **code**

**step1_output_rawtext.R** : Precleaning the text data of training set (remove line feed, "," and "\"). Sample training data evenly. 

*Input : train_data.csv ; Output: raw_text.csv*

**step2_process_text_(1-5).py** : clean the review texts of sampled training set, by removing punctuation, digits, extra whitespaces and transforming all cases to lower cases. Then we performed spelling correction and lemmatization, parallely. 

*Input : raw_text.csv ; Output : processed_text(1-5).csv*

**step2_process_text_merge.py** : merge the output of step2_process_text_(1-5).py together. 

*Input : processed_text(1-5).csv ; Output : processed_text.csv (rows: 500,000)*

**step3_fiture.R** : Remove the businesses with less than 3 comments and 14 running days. Extract phrases, select words for interpretable model on training data of size 400,000. Extract features for interpretable model for both training data of size 400,000 and testset of size about 80,000.

*Input : train_data.csv, processed_text.csv ; Output : bottom_right.csv, phrase.csv, phrase_united.csv, byword.csv, words_score.csv, train_sample.csv, train_val.csv*

**step4_output_rawtext_test.R** : Precleaning the text data of test and validation set (remove line feed, "," and "\"). Sample test and validation data evenly.

*Input : testval_data.csv ; Output: raw_text_test.csv*

**step5_test_text_(1-5).py** : clean the review texts of test and validation set, by removing punctuation, digits, extra whitespaces and transforming all cases to lower cases. Then we performed spelling correction and lemmatization, parallely. 

*Input : raw_text_test.csv ; Output : test_text(1-5).csv*

**step5_test_text_merge.py** : merge the output of step5_test_text_(1-5).py together. 

*Input : test_text(1-5).csv ; Output : test_text.csv*

**step5_test_features.R** : deal with the phrases in test and validation set. Extract features for test and validation set.

*Input : testval_data.csv, test_text.csv, phrase_united.csv, phrase.csv, words_score.csv ; Output : test_clean.csv*

**step6_interpretable.R** : perform CART model on the extracted features, and test the model on testset of size about 80,000 (extract from train_data.csv and do not envolve in words score calculation and phrase extraction procedure).

*Input : train_val.csv, train_sample.csv; Output : None*

**step7_kaggle prediction.py** : perform high accuracy model on the sparse matrix, and predict on test and validation set.
*Input : train_sample.csv, test_clean.csv; Output : result1.csv, result2.csv, result3.csv, result4.csv*



## **image**

