"""Classifiers
Tries various ML classifiers and prints results to stoud
How to run:
1. Ensure that TRAIN_GOLD, DEV_GOLD, TEST_GOLD refer to correct file paths
2. Ensure that countries.txt, json files, postagged files refer to correct filepaths
2. Run from command line (and save to text file if you want):
    $ python classifiers.py > classifier_results.txt
"""

import pandas as pd
import json
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier

from preprocess import read_parsed_file, PARSED_DIR

import codecs
import itertools
import os

TRAIN_GOLD = ""
DEV_GOLD = ""
TEST_GOLD = ""
POS_TAGGED = ""
from sklearn.svm import *
from sklearn.metrics import *

TRAIN_GOLD = "...\\data\\rel-trainset.gold"
DEV_GOLD = "...data\\rel-devset.gold"
TEST_GOLD = "...\\rel-testset.gold"
POS_TAGGED = "...\\postagged-files"

#Column names for gold data
COLUMNS = "Rel,File,Word0Sent,Word0Start,Word0End,Word0NE,Word0Num,Word0Token,Word1Sent,Word1Start,Word1End,Word1NE,Word1Num,Word1Token".split(",")

#Baseline features from column data
FEATURES = ["Word0NE","Word0Token","Word1NE","Word1Token", "Word0Num", "Word1Num"]

LABEL = "Rel"

trees = {filename.rstrip('.head.rel.tokenized.raw.parse'):read_parsed_file(filename) for filename in os.listdir(PARSED_DIR)}

def chunking_features(row):
    filename = row["File"]
    sent_index = row["Word0Sent"]
    word0token = row["Word0Token"]
    word1token = row["Word1Token"]

    # From the parse tree for the given sentence, obtain the smallest subtree containing both words
    subtrees = [t for t in trees[filename][sent_index].subtrees() if word0token in t.leaves() and word1token in t.leaves()]
    subtrees.sort(key = lambda i:len(i.leaves()))
    try:
        chunk = subtrees[0]
    except:
        return {} #This is a kludge to get around situations in which one of the words is represented differently in the data set than in the parse file (e.g. "(AP" vs "AP" in APW20001001.2021.0521). 

    label = chunk.label()
    height = chunk.height()
    children = len(chunk) #The number of children of the root of the phrase chunk

    return {"chunk_label":label, "chunk_height":height, "chunk_children":children}

def make_dict(df):
    """Make dictionaries that map filenames to rawtext for finding pos tag of mentions
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

def word_in_both(word0, word1):
    """Find if mentions share a word
    :param word0: mention1
    :param word1: mention2
    :return: bool: is the same word in both mentions?
    """
    mention1 = word0.split("_")
    mention2 = word1.split("_")
    return not set(mention1).isdisjoint(mention2)

def mention_level_combo(word0sentidx, word0startidx, word1sentidx, word1startidx, sents):
    """Combine mention level POS tags, eg nominal-pronominal, name-nominal, etc.
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
            
        chunk_features = chunking_features(row)
        for chunk_feature in chunk_features:
            datum[chunk_feature] = chunk_features[chunk_feature]

        #combination of mention entity types, ex: PER-ORG, ORG-ORG
        datum['ET12'] = row['Word0NE'] + row['Word1NE']

        #number of words between mention1 and mention2
        datum['#WB'] = row['Word1Start'] - row['Word0End']

        #word in both mentions
        datum['WIBM'] = word_in_both(row['Word0Token'], row['Word1Token'])

        #pos-pos for combination of mention level
        #ex: Name-nominal (NNP-NN), name-pronominal(NNP-$PRP), nominal-pronominal (NN-$PRP)
        # datum['ML12'] = mention_level_combo(row['Word0Sent'], row['Word0Start'],
        #                                     row['Word1Sent'], row['Word1Start'],
        #                                     filedict[row['File']])

        #bool: does relation exist in data extracted from dbpedia?
        # datum['DBPRelExists'] = False
        # if row['Word0Token'] in dbp and row['Word1Token'] in dbp[row['Word0Token']]:
        #     datum['DBPRelExists'] = True
        # if row['Word1Token'] in dbp and row['Word0Token'] in dbp[row['Word1Token']]:
        #     datum['DBPRelExists'] = True

        # entity type of M1 when M2 is country, entity type of M2 when M1 is country
        # with open('countries.txt') as f:
        #     country_list = [line for line in f]
        # datum['ET1Country'] = row['Word0NE'] if row['Word1Token'] in country_list else ""
        # datum['CountryET2'] = row['Word1NE'] if row['Word0Token'] in country_list else ""

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

    #load json files of dbpedia information
    with open('dbp_train.json') as f:
        dbp_train = json.load(f)
    with open('dbp_dev.json') as f:
        dbp_dev = json.load(f)
    with open('dbp_test.json') as f:
        dbp_test = json.load(f)

    #get rawtext dictionaries
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
    Classifiers used:
    SGD
    Logistic Regression (MaxEnt)
    LinearSVC (SVM)
    """
    X_train, y_train, X_dev, y_dev, X_test, y_test = sk_input()

    model_names = ["SGD",
    "LogisticRegression",
    "LinearSVC"
    ]
    models = [
        SGDClassifier(random_state=42),
        LogisticRegression(random_state=42),
        LinearSVC(random_state=42)
        ]

    #Summaries is a list of (model_name,f1_score) pairs
    summaries = list()

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
            # f1 = f1_score(y_dev,predictions,labels=labels_to_report,average='weighted')
            # precision = precision_score(y_dev, predictions,labels=labels_to_report, average='weighted')
            # recall = recall_score(y_dev,predictions,labels=labels_to_report, average='weighted')
            report = classification_report(y_dev,predictions,labels=labels_to_report)
            summaries.append((model_name, report))
        except ValueError:
            print("{} only predicted no_rel for everything. Skipping results...".format(model_name))

    print("Summary:")
    for model, acc in sorted(summaries,key=lambda x: x[1], reverse=True):
        print("{}:\t{}".format(model, acc))

if __name__ == "__main__":
    sk_classify()
