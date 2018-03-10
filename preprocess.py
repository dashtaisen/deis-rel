#Preprocessor

import pandas as pd

SAMPLE_PARSED = "../data/parsed-files/APW20001001.2021.0521.head.rel.tokenized.raw.parse"
SAMPLE_TAGGED = "../data/postagged-files/APW20001001.2021.0521.head.rel.tokenized.raw.tag"
TRAIN_GOLD = "../data/rel-trainset.gold"
DEV_GOLD = "../data/rel-devset.gold"
TEST_GOLD = "../data/rel-testset.gold"
GOLD_COLS = "Rel,File,Word0Sent,Word0Start,Word0End,Word0NE,Word0Num,Word0Token,Word1Sent,Word1Start,Word1End,Word1NE,Word1Num,Word1Token".split(",")

def load_train():
    df = pd.read_csv(TRAIN_GOLD,sep='\t',names=GOLD_COLS)
    return df

def list_train_relations():
    df = load_train()
    print(df[df["Rel"] != "no_rel"][["Rel","Word0Token","Word1Token"]])


def list_relations():
    train_df = pd.read_csv(TRAIN_GOLD,sep='\t',names=GOLD_COLS)
    dev_df = pd.read_csv(DEV_GOLD,sep='\t',names=GOLD_COLS)
    test_df = pd.read_csv(TEST_GOLD,sep='\t',names=GOLD_COLS)

    train_rels = train_df["Rel"].unique()
    dev_rels = dev_df["Rel"].unique()
    test_rels = test_df["Rel"].unique()

    dev_new = [rel for rel in dev_rels if rel not in train_rels]
    test_new = [rel for rel in test_rels if rel not in train_rels]
    return train_rels.tolist() + dev_new + test_new

if __name__ == "__main__":
    all_rels = list_relations()
    with open('relation_list.txt','w') as dest:
        for rel in sorted(all_rels):
            dest.write("{}\n".format(rel))
