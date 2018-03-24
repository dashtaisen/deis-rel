"""Classifiers

Tries various ML classifiers and prints results to stoud

How to run:

1. Ensure that TRAIN_GOLD, DEV_GOLD, TEST_GOLD refer to correct file paths
2. Run from command line and save to text file if you want:
    python classifiers.py > classifier_results.txt
"""

#import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
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

def featurize(df,features=FEATURES, label=LABEL):
    """Convert data to features
    Inputs:
        df: pandas dataframe of data
    Returns:
        X: list of features (as dicts)
        y: list of labels (as strings)
    """
    data_list = list()
    y = df.Rel #TODO: don't hard-code the label like this

    for index, row in df.iterrows():
        datum = dict()
        for feature in FEATURES:
            datum[feature] = row[feature]
        data_list.append(datum)

    return data_list, y

def sk_input(train_data=TRAIN_GOLD, dev_data=DEV_GOLD, test_data=TEST_GOLD):
    """Convert data to sklearn classifier input
    Inputs (optional):
        train_data,dev_data,test_data: paths to data set CSVs
    Returns:
        X_train, y_train, X_dev, y_dev, X_test, y_test: numpy arrays
    """
    train = pd.read_csv(TRAIN_GOLD,sep='\t',names=COLUMNS)
    dev = pd.read_csv(DEV_GOLD,sep='\t',names=COLUMNS)
    test = pd.read_csv(TEST_GOLD,sep='\t',names=COLUMNS)

    #Convert data to features
    data_list_train, y_train = featurize(train)
    data_list_dev, y_dev = featurize(dev)
    data_list_test, y_test = featurize(test)
    data_list_all = data_list_train + data_list_dev + data_list_test

    #Convert feature dicts to arrays
    v = DictVectorizer()
    v.fit(data_list_all) #Don't need to do anything with vec
    X_train = v.transform(data_list_train).toarray()
    X_dev = v.transform(data_list_dev).toarray()
    X_test = v.transform(data_list_test).toarray()

    return X_train, y_train, X_dev, y_dev, X_test, y_test

def mallet_input():
    """Convert data to mallet classifier input"""
    pass

def sk_classify():
    """Use scikit-learn to classify the data
    """
    X_train, y_train, X_dev, y_dev, X_test, y_test = sk_input()

    strat_names = ['Baseline (stratified)','Baseline (most_frequent)','Baseline (prior)','Baseline (uniform)']
    strats = ['stratified','most_frequent','prior','uniform']
    dummies = list()
    for strat in strats:
        dummies.append(DummyClassifier(strategy=strat))
    model_names = ["SGD",
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    ]
    models = [
        SGDClassifier(random_state=42),
        LogisticRegression(random_state=42),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        ]

    #Summaries is a list of (model_name,f1_score) pairs
    summaries = list()


    #Get dummy classifer results as baseline
    #Definitely want to do better than these
    print("Dummy classifier (baseline) results:")
    for i in range(len(dummies)):
        model_name = strat_names[i]
        model = dummies[i]
        print("Training dummy model: {}".format(model_name))
        model.fit(X_train,y_train)

        predictions = model.predict(X_test)
        labels_to_report = [label for label in predictions if label != "no_rel"]
        try: #May except ValueError if only predicts no_rel
            #Print classification report for detailed analysis
            #print(classification_report(y_test,predictions,labels=labels_to_report))
            f1 = f1_score(y_test,predictions,labels=labels_to_report,average='weighted')
            summaries.append((model_name,f1))
        except ValueError:
            print("{} only predicted no_rel for everything. Skipping results...".format(model_name))

    #Get actual ML results
    for i in range(len(models)):
        model_name = model_names[i]
        model = models[i]
        print("Training model: {}".format(model_name))
        model.fit(X_train,y_train)

        predictions = model.predict(X_test)
        labels_to_report = [label for label in predictions if label != "no_rel"]
        try: #May except ValueError if only predicts no_rel
            #Print classification report for detailed analysis
            #print(classification_report(y_test,predictions,labels=labels_to_report))
            f1 = f1_score(y_test,predictions,labels=labels_to_report,average='weighted')
            summaries.append((model_name,f1))
        except ValueError:
            print("{} only predicted no_rel for everything. Skipping results...".format(model_name))

    print("Summary:")
    for model, acc in sorted(summaries,key=lambda x: x[1], reverse=True):
        print("{}:\t{}".format(model, acc))

if __name__ == "__main__":
    sk_classify()
