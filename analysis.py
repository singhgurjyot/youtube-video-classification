# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:30:39 2019

@author: singh
"""

import pandas as pd
import re 
import nltk 
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

df = pd.read_csv('final_data.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.dropna(axis=0, inplace=True)

corpus = []
for title in df['title']:
  review = re.sub('[^a-zA-Z]', ' ', title)
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus.append(review)

i=0
corpus1 = []
for description in df['description']:
  review = re.sub('[^a-zA-Z]', ' ', description)
  i+=1
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus1.append(review)
  
df_title = pd.DataFrame({'title':corpus})
df_description = pd.DataFrame({'description':corpus1})

df_category = df[['category']]

from sklearn.preprocessing import LabelEncoder

dfcategory = df_category.apply(LabelEncoder().fit_transform)

df_new = pd.concat([df[['vid']], df_title, df_description, dfcategory], axis=1, join_axes= [df[['vid']].index])

from sklearn.feature_extraction.text import CountVectorizer   
cv = CountVectorizer(max_features = 1500) 
X = cv.fit_transform(corpus, corpus1).toarray() 
y = df_new.iloc[:, 3].values


#model

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
classifier.score(X_test, y_test)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm