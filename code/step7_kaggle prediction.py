#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:42:32 2018

@author: jacqueline
"""

import pandas as pd
import numpy as np

# read the data from disk and split into lines
# we use .strip() to remove the final (empty) line

train = pd.read_csv("data/train_sample.csv")
test = pd.read_csv("data/test_clean.csv")


# we're interested in the text of each review and the stars rating
# so we load these into separate lists
texts_train = train["text"]
stars_train = train["stars"]
train_id = train["commendid"]
texts_test = test["text"]
#stars_test = test["stars"]
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
# classifier = LinearSVC(C=0.2)       
from sklearn.linear_model import LinearRegression
# train the classifier

# classifier.fit(X_train_positive, y_train_positive)
regr = LinearRegression().fit(X_train_positive, y_train_positive)
positive_preds = regr.predict(X_test_positive)
test_id_positive = np.array(test_id_positive, dtype=int)
positive_predictions = pd.concat([pd.DataFrame(test_id_positive,columns=['commendid']),pd.DataFrame(positive_preds,columns=['predict_stars'])],axis=1)

# classifier.fit(X_train_negative, y_train_negative)
regr = LinearRegression().fit(X_train_negative, y_train_negative)
negative_preds = regr.predict(X_test_negative)
test_id_negative = np.array(test_id_negative, dtype=int)
negative_predictions = pd.concat([pd.DataFrame(test_id_negative,columns=['commendid']),pd.DataFrame(negative_preds,columns=['predict_stars'])],axis=1)

predictions = pd.concat([positive_predictions,negative_predictions],axis=0)
predictions['predict_stars'] = [5 if (y>=4.8) else y for y in predictions['predict_stars']]
predictions['predict_stars'] = [1 if (y<=1.1) else y for y in predictions['predict_stars']]


whole_predict = test2.set_index('commendid').join(predictions.set_index('commendid'))

#write as csv file
whole_predict.to_csv("result.csv",sep=',')

