"""Classifiers"""

#import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Lasso, ElasticNet, Ridge, SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier

import itertools

TRAIN_GOLD = "../data/rel-trainset.gold"
DEV_GOLD = "../data/rel-devset.gold"
TEST_GOLD = "../data/rel-testset.gold"

#Column names for gold data
COLUMNS = "Rel,File,Word0Sent,Word0Start,Word0End,Word0NE,Word0Num,Word0Token,Word1Sent,Word1Start,Word1End,Word1NE,Word1Num,Word1Token".split(",")

FEATURES = ["Word0NE","Word0Token","Word1NE","Word1Token"]

LABEL = "Rel"

def sk_input():
    train = pd.read_csv(TRAIN_GOLD,sep='\t',names=COLUMNS)
    dev = pd.read_csv(DEV_GOLD,sep='\t',names=COLUMNS)
    test = pd.read_csv(TEST_GOLD,sep='\t',names=COLUMNS)

    X_train = train.filter(FEATURES)
    y_train = train.Rel

    X_dev = dev.filter(FEATURES)
    y_dev = dev.Rel

    X_test = test.filter(FEATURES)
    y_test = test.Rel

    all_labels = pd.concat([y_train,y_dev,y_test])

    all_dfs = pd.concat([train,dev,test])

    #Convert string-based features to integer values
    #TODO: convert integer values to binary
    for feature in FEATURES:
        feature_values = all_dfs[feature].unique()
        feature_dict = {val:num for num,val in enumerate(feature_values)}
        X_train[feature].replace(feature_dict,inplace=True)
        X_dev[feature].replace(feature_dict,inplace=True)
        X_test[feature].replace(feature_dict,inplace=True)

    print(X_train)

    lenc = LabelEncoder()
    lenc.fit(all_labels)

    return X_train, y_train, X_dev, y_dev, X_test, y_test

def sk_classify():
    X_train, y_train, X_dev, y_dev, X_test, y_test = sk_input()

    strat_names = ['Baseline (stratified)','Baseline (most_frequent)','Baseline (prior)','Baseline (uniform)']
    strats = ['stratified','most_frequent','prior','uniform']
    dummies = list()
    for strat in strats:
        dummies.append(DummyClassifier(strategy=strat))
    #models = [model1, model2, model3]
    model_names = ["SGD",
    "LogisticRegression",
    "KNeighbors",
    "SVC",
    "DecisionTree",
    "RandomForest",
    "MLP",
    "AdaBoost",
    "QuadraticDiscriminantAnalysis"
    ]
    models = [
        SGDClassifier(random_state=42),
        LogisticRegression(random_state=42),
        KNeighborsClassifier(3),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        QuadraticDiscriminantAnalysis(),
        ]

    summaries = list()

    print("Dummy classifier (baseline) results:")
    for i in range(len(dummies)):
        model_name = strat_names[i]
        model = dummies[i]
        print("Dummy model: {}\n".format(model_name))
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        print(classification_report(predictions, y_test))
        accuracy = accuracy_score(predictions, y_test)
        print("Accuracy: {}\n".format(accuracy))
        summaries.append((model_name,accuracy))

    for i in range(len(models)):
        model_name = model_names[i]
        model = models[i]
        print("Model: {}\n".format(model_name))
        model.fit(X_train,y_train)

        predictions = model.predict(X_test)
        print(classification_report(predictions, y_test))
        accuracy = accuracy_score(predictions, y_test)
        print("Accuracy: {}\n".format(accuracy))
        summaries.append((model_name,accuracy))

    print("Summary:")
    for model, acc in sorted(summaries,key=lambda x: x[1], reverse=True):
        print("{}:\t{}".format(model, acc))

if __name__ == "__main__":
    sk_classify()
