# -*- coding: utf-8 -*-

import pickle
from sklearn import svm
import numpy as np
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import json, io, os


y = []
tweets = []

if os.path.isfile("training_tweets_2017.txt"):
    os.remove("training_tweets_2017.txt")

file = open('training_tweets_2017.txt', 'a+')
for line in io.open('train_nega_tweets_2017.txt').readlines():
    tweet = json.loads(line)
    text = tweet['text'].encode('utf-8').replace("'", '').replace('\r', '').replace('\n', '').lower().strip()
    file.write(str(0) + ", " + text + '\n')
# file.close()

# file = open('training_tweets_2017.txt', 'a+')
for line in io.open('train_posi_tweets_2017.txt').readlines():
    tweet = json.loads(line)
    text = tweet['text'].encode('utf-8').replace("'", '').replace('\r', '').replace('\n', '').lower().strip()
    file.write(str(1) + ", " + text + '\n')
file.close()

for line in open('training_tweets_2017.txt').readlines():
    items = line.split(',')
    y.append(int(items[0]))
    tweets.append(items[1].lower().strip())
y = np.array(y)


vv = CountVectorizer(min_df=5, max_df=0.7, stop_words="english")
X = vv.fit_transform(tweets)

tf_transformer = TfidfTransformer().fit(X)
X = tf_transformer.transform(X)
vocab = vv.vocabulary_
print "The total number of training tweets: {} ({} positives, {}: negatives)".format(len(y), sum(y), len(y) - sum(y))
print "The size of vocabulary: {}".format(X.shape[1])
print "The vocabulary includes the following keywords: {}".format(vocab)

# 10 folder cross validation to estimate the best w and b
svc = svm.LinearSVC()
Cs = range(1, 20)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv = 10)
clf.fit(X, y)

print "The estimated w: "
print clf.best_estimator_.coef_

print "The estimated b: "
print clf.best_estimator_.intercept_

print "The estimated C after the grid search for 10 fold cross validation: "
print clf.best_params_

print "ten-fold cross-validation training accuracy"
print clf.best_score_ 

print "training accuracy"
pred_y = clf.predict(X)
print sum([1 for y1, y2 in zip(pred_y, y) if y1 == y2])/(len(y) * 1.0)

t_ids = []
test_tweets = []
loader = []
for line in open('test_tweets.txt').readlines():
    loader = json.loads(line)
    t_ids.append(loader['embersId'].encode('utf-8'))
    test_tweets.append(loader['text'].encode('utf-8').replace('\r', '').replace('\n', ''))

test_X = CountVectorizer(vocabulary = vocab).fit_transform(test_tweets)
tf_transformer = TfidfTransformer().fit(test_X)
test_X = tf_transformer.transform(test_X)
test_y = clf.predict(test_X)

print "Class label details: false (0): negatives, true (1): positives."
print "The total number of testing tweets: {} ({} are predicted as positives, {} are predicted as negatives)".format(len(test_y), sum(test_y), len(test_y) - sum(test_y))

if os.path.isfile("pridictions.txt"):
    os.remove("pridictions.txt")

file = open('pridictions.txt','w')
file.write("{")
for l in range(len(test_y)):
    if (test_y[l] == 1):
        file.write('"' + t_ids[l] + '"' + ":" + " true, ")
    else:
        file.write('"' + t_ids[l] + '"' + ":" + " false, ")
file.write("}")
file.close()


objser = "saved_model.pkl"
with open(objser, 'wb') as file:
    pickle.dump(clf, file)

with open(objser, 'rb') as file:
    model = pickle.load(file)

score = model.score(X, y)
print("Training accuracy score: {0:.2f} %".format(100 * score))
Ypredict = model.predict(X)