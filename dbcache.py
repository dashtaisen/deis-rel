import json
import dbquery
import classifiers

TRAIN_GOLD = classifiers.TRAIN_GOLD
DEV_GOLD = classifiers.DEV_GOLD
TEST_GOLD = classifiers.TEST_GOLD

def make_caches(source_dir):
    dbp_relations = dict()
    f = open(source_dir)
    with open(source_dir) as f:
        for row in f:
            row = row.split()
            word0 = row[7]
            word1 = row[-1]
            relExistBool = dbquery.relation_exists(word0, word1)
            if relExistBool:
                dbp_relations[word0] = {word1: relExistBool}

    if source_dir==TRAIN_GOLD:
        with open('dbp_train.json', 'w') as f:
            json.dump(dbp_relations, f)
    if source_dir==DEV_GOLD:
        with open('dbp_dev.json', 'w') as f:
            json.dump(dbp_relations, f)
    if source_dir==TEST_GOLD:
        with open('dbp_test.json', 'w') as f:
            json.dump(dbp_relations, f)

if __name__ == "__main__":
    print("making train caches...")
    make_caches(TRAIN_GOLD)
    print("cache complete!")
    print("making caches...")
    make_caches(TRAIN_GOLD)
    print("cache complete!")
    print("making caches...")
    make_caches(TEST_GOLD)
    print("cache complete!")

