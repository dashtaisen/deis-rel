"""Classifiers
Tries various ML classifiers and prints results to stoud
How to run:
1. Ensure that TRAIN_GOLD, DEV_GOLD, TEST_GOLD, POS_TAGGED refer to correct file paths
2. Ensure that countries.txt, json files are in the same directory as classifiers.py
2. Run from command line and save to text file if you want:
    python classifiers.py > classifier_results.txt
"""

import dbquery
import pandas as pd
import numpy as np
import json
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
import itertools

TRAIN_GOLD = ""
DEV_GOLD = ""
TEST_GOLD = ""
POS_TAGGED = ""

#Column names for gold data
COLUMNS = "Rel,File,Word0Sent,Word0Start,Word0End,Word0NE,Word0Num,Word0Token,Word1Sent,Word1Start,Word1End,Word1NE,Word1Num,Word1Token".split(",")

FEATURES = ["Word0NE","Word0Token","Word1NE","Word1Token", "Word0Num", "Word1Num"]

LABEL = "Rel"

def make_dict(df):
    """
    :param df: pandas dataframe
    :return: dict with key=filename and value=(text, pos) tuples in a nested list of sentences
    """
    filedict = dict()
    for index, row in df.iterrows():
        if row['File'] not in filedict:
            sents = list()
            filename = os.path.join(*[POS_TAGGED, row['File'] + '.head.rel.tokenized.raw.tag'])
            with open(filename) as f:
                lines = filter(None, (line.rstrip().split() for line in f))
                for line in lines:
                    sent = [tuple(word.split("_")) for word in line]
                    sents.append(sent)
            filedict[row['File']] = sents
    return filedict

#TODO:fix maybe?
def get_prev_word(sent_idx, word_idx, sents):
    """
    :param sent_idx:
    :param word_idx:
    :param sents: rawtext associated with the file
    :return: word before mention1
    """
    if word_idx > 0 :
        return sents[sent_idx][word_idx - 1][0]

#TODO:fix maybe?
def get_next_word(sent_idx, word_idx, sents):
    """
    :param sent_idx:
    :param word_idx:
    :param sents: rawtext associated with the file
    :return: word after mention2
    """
    if word_idx + 1 < len(sents[sent_idx]):
        return sents[sent_idx][word_idx + 1][0]

def word_in_both(word0, word1):
    """
    :param word0: mention1
    :param word1: mention2
    :return: bool: is the same word in both mentions?
    """
    mention1 = word0.split("_")
    mention2 = word1.split("_")
    return not set(mention1).isdisjoint(mention2)

#TODO: fix maybe?
# def number_mentions_between(word0end, word1start, sent_idx, sents):

def mention_level_combo(word0sentidx, word0startidx, word1sentidx, word1startidx, sents):
    """
    :param word0sentidx:
    :param word0startidx:
    :param word1sentidx:
    :param word1startidx:
    :param sents: rawtext associated with the file
    :return: Mention1 pos + Mention 2 pos
    """
    word0pos = sents[word0sentidx][word0startidx][1]
    word1pos = sents[word1sentidx][word1startidx][1]
    return word0pos + word1pos

def featurize(df, dbp, filedict, features=FEATURES, label=LABEL):
    """Convert data to features
    Inputs:
        df: pandas dataframe of data
        dbp: json cache of relations from dbpedia
        filedict: dict linking filename to rawtext = {filename: [[(word,pos)],[(word,pos)] }
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

        #combination of mention entity types, ex: PER-ORG, ORG-ORG
        datum['ET12'] = row['Word0NE'] + row['Word1NE']

        #number of words between mention1 and mention2
        datum['#WB'] = row['Word1Start'] - row['Word0End']

        #word in both mentions
        datum['WIBM'] = word_in_both(row['Word0Token'], row['Word1Token'])

        #pos-pos for combination of mention level
        #ex: Name-nominal (NNP-NN), name-pronominal(NNP-$PRP), nominal-pronominal (NN-$PRP)
        datum['ML12'] = mention_level_combo(row['Word0Sent'], row['Word0Start'],
                                            row['Word1Sent'], row['Word1Start'],
                                            filedict[row['File']])

        #bool: does relation exist in data extracted from dbpedia?
        datum['DBPRelExists'] = False
        if row['Word0Token'] in dbp and row['Word1Token'] in dbp[row['Word0Token']]:
            datum['DBPRelExists'] = True

        # entity type of M1 when M2 is country, entity type of M2 when M1 is country
        with open('countries.txt') as f:
            country_list = [line for line in f]
        datum['ET1Country'] = row['Word0NE'] if row['Word1Token'] in country_list else ""
        datum['CountryET2'] = row['Word1NE'] if row['Word0Token'] in country_list else ""

        # UNUSED FEATURES
        # datum['WBM1'] = get_prev_word(row['Word0Sent'], row['Word0Start'], filedict[row['File']])
        # datum['WAM2'] = get_next_word(row['Word1Sent'], row['Word1End'], filedict[row['File']])
        # datum['#MB'] = number_mentions_between(row['Word0End'], row['Word1Start'], row['Word0Sent'], filedict[row['File']])
        
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

    #load dbp json files
    with open('dbp_train.json') as f:
        dbp_train = json.load(f)
    with open('dbp_dev.json') as f:
        dbp_dev = json.load(f)
    with open('dbp_test.json') as f:
        dbp_test = json.load(f)

    #make rawtext dicts
    train_dict = make_dict(train)
    dev_dict = make_dict(dev)
    test_dict = make_dict(test)

    #Convert data to features
    data_list_train, y_train = featurize(train, dbp_train, train_dict)
    data_list_dev, y_dev = featurize(dev, dbp_dev, dev_dict)
    data_list_test, y_test = featurize(test, dbp_test, test_dict)
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
    Classifiers tested: SGD, Logistic Regression, Decision Tree
    """
    X_train, y_train, X_dev, y_dev, X_test, y_test = sk_input()

    # strat_names = ['Baseline (stratified)','Baseline (most_frequent)','Baseline (prior)','Baseline (uniform)']
    # strats = ['stratified','most_frequent','prior','uniform']
    # dummies = list()
    # for strat in strats:
    #     dummies.append(DummyClassifier(strategy=strat))
    model_names = ["SGD",
    "LogisticRegression",
    "DecisionTree",
#     "RandomForest",
    ]
    models = [
        SGDClassifier(random_state=42),
        LogisticRegression(random_state=42),
        DecisionTreeClassifier(max_depth=5),
#         RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        ]

    #Summaries is a list of (model_name,f1_score) pairs
    summaries = list()


    #Get dummy classifer results as baseline
    #Definitely want to do better than these
    # print("Dummy classifier (baseline) results:")
    # for i in range(len(dummies)):
    #     model_name = strat_names[i]
    #     model = dummies[i]
    #     print("Training dummy model: {}".format(model_name))
    #     model.fit(X_train,y_train)
    #
    #     predictions = model.predict(X_test)
    #     labels_to_report = [label for label in predictions if label != "no_rel"]
    #     try: #May except ValueError if only predicts no_rel
    #         #Print classification report for detailed analysis
    #         #print(classification_report(y_test,predictions,labels=labels_to_report))
    #         f1 = f1_score(y_test,predictions,labels=labels_to_report,average='weighted')
    #         summaries.append((model_name,f1))
    #     except ValueError:
    #         print("{} only predicted no_rel for everything. Skipping results...".format(model_name))

    #Get actual ML results
    for i in range(len(models)):
        model_name = model_names[i]
        model = models[i]
        print("Training model: {}".format(model_name))
        model.fit(X_train,y_train)

        predictions = model.predict(X_dev)
        labels_to_report = [label for label in predictions if label != "no_rel"]
        try: #May except ValueError if only predicts no_rel
            #Print classification report for detailed analysis
            #print(classification_report(y_test,predictions,labels=labels_to_report))
            f1 = f1_score(y_dev,predictions,labels=labels_to_report,average='weighted')
            summaries.append((model_name,f1))
        except ValueError:
            print("{} only predicted no_rel for everything. Skipping results...".format(model_name))

    print("Summary:")
    for model, acc in sorted(summaries,key=lambda x: x[1], reverse=True):
        print("{}:\t{}".format(model, acc))

if __name__ == "__main__":
    sk_classify()
