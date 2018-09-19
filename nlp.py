# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:06:40 2018

@author: Abhishek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
#quoting=3->to ignore double quotes
dataset = pd.read_csv('SMS.tsv', delimiter='\t', quoting=3)

print(dataset['ham'][0])

#Cleaning Texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 5574):
    review = re.sub('[^a-zA-Z]', ' ', dataset['message'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder();
y = le.fit_transform(y) 

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0, max_depth=None)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Cheking the performance of algorithm using kFoldCrossValidation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10,n_jobs=-1)
scores.mean()
classifier.score(X_train, y_train)
classifier.score(X_test, y_test)


