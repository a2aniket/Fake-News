# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:39:45 2020

@author: sai
"""
import pandas as pd
from nltk.corpus import stopwords 
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle 



         
def check(text):
    data=pd.read_csv("data/data.csv")
        
    ps = PorterStemmer()
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
        
    cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
    X = cv.fit_transform(corpus).toarray()
        
    count_df = pd.DataFrame(X, columns=cv.get_feature_names())
    data=data.append(count_df,ignore_index=True)
        
    data.fillna(0,inplace=True)
        
    data=data.astype(int)
    data=data.drop("Unnamed: 0",axis=1)
        
    start=data.shape[1]%5000
    stop=data.shape[1]
        
    model = pickle.load(open('data/train model.pkl', 'rb'))
    return model.predict(data.iloc[len(data)-1:,start:stop])

