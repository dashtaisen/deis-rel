#Various preprocessing tools

import pandas as pd

SAMPLE_PARSED = "../data/parsed-files/APW20001001.2021.0521.head.rel.tokenized.raw.parse"
SAMPLE_TAGGED = "../data/postagged-files/APW20001001.2021.0521.head.rel.tokenized.raw.tag"
TRAIN_GOLD = "../data/rel-trainset.gold"
DEV_GOLD = "../data/rel-devset.gold"
TEST_GOLD = "../data/rel-testset.gold"

#Column names for gold data
GOLD_COLS = "Rel,File,Word0Sent,Word0Start,Word0End,Word0NE,Word0Num,Word0Token,Word1Sent,Word1Start,Word1End,Word1NE,Word1Num,Word1Token".split(",")

def list_relations():
    """List all possible relations"""
    #Load the training, dev, and test gold sets as Pandas dataframes
    #Files are separated by tabs
    #Column names as in GOLD_COLS
    train_df = pd.read_csv(TRAIN_GOLD,sep='\t',names=GOLD_COLS)
    dev_df = pd.read_csv(DEV_GOLD,sep='\t',names=GOLD_COLS)
    test_df = pd.read_csv(TEST_GOLD,sep='\t',names=GOLD_COLS)

    #Find the unique relation values using df.unique()
    #Then convert them to a list
    train_rels = train_df["Rel"].unique().tolist()
    dev_rels = dev_df["Rel"].unique().tolist()
    test_rels = test_df["Rel"].unique().tolist()

    #See if dev and test have any relations that aren't in train
    dev_new = [rel for rel in dev_rels if rel not in train_rels]
    test_new = [rel for rel in test_rels if rel not in train_rels]
    return train_rels + dev_new + test_new

if __name__ == "__main__":
    #Find all possible relation types
    all_rels = list_relations()
    #Write them to a file, sorted alphabetically
    with open('relation_list.txt','w') as dest:
        for rel in sorted(all_rels):
            dest.write("{}\n".format(rel))
