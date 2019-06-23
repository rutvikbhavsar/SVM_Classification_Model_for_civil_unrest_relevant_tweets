# -*- coding: utf-8 -*-

import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


y = []
tweets = []

for line in open('training_tweets_2017.txt').readlines():
    items = line.split(',')
    y.append(int(items[0]))
    tweets.append(items[1].lower().strip())
y = np.array(y)

vv = CountVectorizer(min_df=5, max_df=0.7, stop_words="english")
X = vv.fit_transform(tweets)

tf_transformer = TfidfTransformer().fit(X)
X = tf_transformer.transform(X)

objser = "saved_model.pkl"

with open(objser, 'rb') as file:
    model = pickle.load(file)

score = model.score(X, y)
print("Training accuracy score: {0:.2f} %".format(100 * score))
Ypredict = model.predict(X)