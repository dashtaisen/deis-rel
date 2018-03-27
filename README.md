# deis-rel

IE Relations project (Annie Thorburn, Jennifer Storozum, Nicholas Miller)

## Files

- preprocess.py: tools for preprocessing the data sets
- dbquery.py: tools for finding relations using DBPedia SPARQL queries
- dbcache.py: tools for caching dbquery.py results (to save time)
- classifiers.py: tools for training and testing the classifier


- json files: results of dbpquery
- countries.txt: list of country names
- relation_list.txt: list of possible relations (can be ignored)

- The other files in the repository are copies of the dataset and can be ignored

## How to run

1. Ensure that the 'data' folder from the assignment instructions is in the parent folder of these scripts
2. Run classifiers.py, e.g. `python classifiers.py > results.txt`
3. If you want to experiment with different combinations of features, add or remove features from the `BASELINE_FEATURES` and `ADDITIONAL_FEATURES` variables at the top of classifiers.py
