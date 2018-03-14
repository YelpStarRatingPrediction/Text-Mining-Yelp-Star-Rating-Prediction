#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:42:32 2018
@author: jacqueline
"""

import pandas as pd
import numpy as np

# read the data from disk and split into lines

train = pd.read_csv("/Users/jacqueline/Google 云端硬盘/628 Module2/github/data/train_sample.csv")
test = pd.read_csv("/Users/jacqueline/Google 云端硬盘/628 Module2/github/data/test_clean.csv")


# we're interested in the text of each review and the stars rating
# so we load these into separate lists
"""Below are 2-step models"""
texts_train = train["text"]
stars_train = train["stars"]
train_id = train["commendid"]
texts_test = test["text"]
test_id = test["Id"]

whole_text = pd.concat([texts_train,texts_test])

from sklearn.feature_extraction.text import TfidfVectorizer
# This vectorizer breaks text into single words and bi-grams
# and then calculates the TF-IDF representation
vectorizer = TfidfVectorizer(ngram_range=(1,2))


# the 'fit' builds up the vocabulary from all the reviews
# while the 'transform' step turns each indivdual text into
# a matrix of numbers.
vectors = vectorizer.fit_transform(whole_text.values.astype('U'))
vectors_train = vectors[0:400000]
vectors_test = vectors[400000:]

X_train = vectors_train
X_test = vectors_test

y_train = stars_train

# convert the train set

y_train2 = ["n" if (y == 1 or y == 2) else "p" for y in y_train]
 
# convert the test set


from sklearn.svm import LinearSVC

# initialise the SVM classifier
classifier = LinearSVC(C=0.2)       

# train the classifier

classifier.fit(X_train, y_train2)
preds = classifier.predict(X_test)


train2 = pd.concat([train_id,texts_train,stars_train],axis=1)
test2 = pd.concat([test_id,texts_test,pd.Series(preds)],axis=1)
test2.columns = ['commendid','text','stars'] 

train_positive = train2[(True-train2['stars'].isin([1,2]))]  
train_negative = train2[(True-train2['stars'].isin([3,4,5]))]

        
texts_train_positive = train_positive['text']
y_train_positive = train_positive['stars']
texts_train_negative = train_negative['text']
y_train_negative = train_negative['stars']

test_positive = test2[(True-test2['stars'].isin(['n']))]
test_negative = test2[(True-test2['stars'].isin(['p']))]

texts_test_positive = test_positive['text']
test_id_positive = test_positive['commendid']
texts_test_negative = test_negative['text']
test_id_negative = test_negative['commendid']

# Combine the train and test texts
whole_text_positive = pd.concat([texts_train_positive,texts_test_positive])
whole_text_negative = pd.concat([texts_train_negative,texts_test_negative])

# the 'fit' builds up the vocabulary from all the reviews
# while the 'transform' step turns each indivdual text into
# a matrix of numbers.
vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=20000)
vectors_positive = vectorizer.fit_transform(whole_text_positive.values.astype('U'))
vectors_train_positive = vectors_positive[0:241038]
vectors_test_positive = vectors_positive[241038:]

vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=10000)
vectors_negative = vectorizer.fit_transform(whole_text_negative.values.astype('U'))
vectors_train_negative = vectors_negative[0:158962]
vectors_test_negative = vectors_negative[158962:]

X_train_positive = vectors_train_positive
X_test_positive = vectors_test_positive
y_train_positive = y_train_positive.astype(int)

X_train_negative = vectors_train_negative
X_test_negative = vectors_test_negative
y_train_negative = y_train_negative.astype(int)

# initialise the SVM classifier
classifier = LinearSVC(C=0.2)       
from sklearn.linear_model import LinearRegression
# train the classifier

classifier.fit(X_train_positive, y_train_positive)
regr = LinearRegression().fit(X_train_positive, y_train_positive)
positive_preds1 = classifier.predict(X_test_positive)
positive_preds2 = regr.predict(X_test_positive)
test_id_positive = np.array(test_id_positive, dtype=int)
positive_predictions1 = pd.concat([pd.DataFrame(test_id_positive,columns=['commendid']),pd.DataFrame(positive_preds1,columns=['predict_stars'])],axis=1)
positive_predictions2 = pd.concat([pd.DataFrame(test_id_positive,columns=['commendid']),pd.DataFrame(positive_preds2,columns=['predict_stars'])],axis=1)

classifier.fit(X_train_negative, y_train_negative)
regr = LinearRegression().fit(X_train_negative, y_train_negative)
negative_preds1 = classifier.predict(X_test_negative)
negative_preds2 = regr.predict(X_test_negative)
test_id_negative = np.array(test_id_negative, dtype=int)
negative_predictions1 = pd.concat([pd.DataFrame(test_id_negative,columns=['commendid']),pd.DataFrame(negative_preds1,columns=['predict_stars'])],axis=1)
negative_predictions2 = pd.concat([pd.DataFrame(test_id_negative,columns=['commendid']),pd.DataFrame(negative_preds2,columns=['predict_stars'])],axis=1)

predictions1 = pd.concat([positive_predictions1,negative_predictions1],axis=0)
predictions1['predict_stars'] = [5 if (y>=4.8) else y for y in predictions1['predict_stars']]
predictions1['predict_stars'] = [1 if (y<=1.1) else y for y in predictions1['predict_stars']]
predictions2 = pd.concat([positive_predictions2,negative_predictions2],axis=0)
predictions2['predict_stars'] = [5 if (y>=4.8) else y for y in predictions2['predict_stars']]
predictions2['predict_stars'] = [1 if (y<=1.1) else y for y in predictions2['predict_stars']]


whole_predict1 = test2.set_index('commendid').join(predictions1.set_index('commendid'))
whole_predict2 = test2.set_index('commendid').join(predictions2.set_index('commendid'))


#write as csv file
whole_predict1.to_csv("result1.csv",sep=',')   #SVC+SVC result
whole_predict2.to_csv("result2.csv",sep=',')   #SVC+Linear result

########################################################################################

"""Below are 1-step models"""
texts_train = train["text"]
stars_train = train["stars"]
train_id = train["commendid"]
texts_test = test["text"]
test_id = test["Id"]

whole_text = pd.concat([texts_train,texts_test])

from sklearn.feature_extraction.text import TfidfVectorizer
# This vectorizer breaks text into single words and bi-grams
# and then calculates the TF-IDF representation
vectorizer = TfidfVectorizer(ngram_range=(1,2))


# the 'fit' builds up the vocabulary from all the reviews
# while the 'transform' step turns each indivdual text into
# a matrix of numbers.
#vectors_train = vectorizer.fit_transform(texts_train.values.astype('U'))
#vectors_test = vectorizer.fit_transform(texts_test.values.astype('U'))
vectors = vectorizer.fit_transform(whole_text.values.astype('U'))
vectors_train = vectors[0:400000]
vectors_test = vectors[400000:]
vectors_kaggle = vectors[400000:]

X_train = vectors_train
X_test = vectors_test
y_train = stars_train

"""Support Vector Classifier"""

# initialise the SVM classifier
classifier = LinearSVC(C=0.2)  
# train the classifier

classifier.fit(X_train, y_train)

preds = classifier.predict(X_test)  #prediction
whole_predict3 = pd.concat([pd.DataFrame(test_id,columns=['Id']),pd.DataFrame(preds,columns=['Prediction1'])],axis=1)
whole_predict3.to_csv("result3.csv",sep=',')   #SVC result




"""Naive Bayes Method"""
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)

preds_nb = nb.predict(X_test)
whole_predict4 = pd.concat([pd.DataFrame(test_id,columns=['Id']),pd.DataFrame(preds_nb,columns=['Prediction1'])],axis=1)
whole_predict4.to_csv("result4.csv",sep=',')   #Naive Bayes result
