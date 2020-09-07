# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:33:06 2020

@author: sai
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import itertools
import pickle

def train_model():
    df=pd.read_csv("data/train.csv")
    X=df.drop('label',axis=1)
    y=df['label']
    df=df.dropna()
    messages=df.copy()
    messages.reset_index(inplace=True)
    
    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
        review = review.lower()
        review = review.split()

        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    
    print(corpus[1])
    cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
    X = cv.fit_transform(corpus).toarray()
    
    y=messages['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    count_df = pd.DataFrame(X_train, columns=cv.get_feature_names())
    count_df.to_csv("data/data.csv")
    classifier=MultinomialNB()
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    
    train_score=classifier.score(X_train,y_train)
    test_score=classifier.score(X_test,y_test)
    total_score = metrics.accuracy_score(y_test, pred)
    model_detail={
        "train_score":train_score,
        "test_score":train_score,
        "total_score":total_score
        }
    
    filename = 'data/model details.pkl'
    outfile = open(filename,'wb')
    pickle.dump(model_detail,outfile)
    outfile.close()
    
    cm = metrics.confusion_matrix(y_test, pred)
    filename = 'data/train model.pkl'
    outfile = open(filename,'wb')
    pickle.dump(classifier,outfile)
    outfile.close()
    return count_df
    