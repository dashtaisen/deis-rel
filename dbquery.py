"""Use SPARQL to query DBPedia for relations

How to run:

To do a SPARQL query of DBPedia:
Use the do_query(first_entity,second_entity) method

To check whether a relation exists:
Use the relation_exists(first_entity,second_entity) method
"""

WIKILINK_RELATION = 'http://dbpedia.org/ontology/wikiPageWikiLink'

from SPARQLWrapper import SPARQLWrapper, JSON

#TODO: ensure entity names are in correct format, e.g. underscore
#TODO: check relation in both directions
#TODO: do something more complex than just doing query
#TODO: process results more deeply than just checking if relation exists
#TODO: Use as features for relation classification pipeline

def do_query(first_entity,second_entity):
    """Query DBPedia to find any relations
    Inputs:
        first_entity: name of entity
        second_entity: name of entity
    Returns:
        result: dict() converted from JSON
    """
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    """The following SPARQL query looks for all relationships
    between the two entities. Note we have to use the long form
    (i.e. no prefix),because entities may have periods etc. in their names.
    """

    query_text = """PREFIX db:<http://dbpedia.org/resource/>
        SELECT ?relationship
        WHERE {{
        <http://dbpedia.org/resource/{}>
        ?relationship
        <http://dbpedia.org/resource/{}>
        }}
    """.format(first_entity,second_entity)

    #print(query_text) #for debugging

    sparql.setQuery(query_text)  # the previous query as a literal string
    return sparql.query().convert()

def relation_exists(first_entity,second_entity):
    """Query DBPedia to find if any relations exist
    Inputs:
        first_entity: name of entity
        second_entity: name of entity
    Returns:
        boolean
    """
    result = do_query(first_entity,second_entity)

    #Find the relations
    #But, we don't care about WIKILINK_RELATION
    relations = [item['relationship']['value'] for item in \
        result['results']['bindings'] if \
        item['relationship']['value'] != WIKILINK_RELATION]
    #if relation exists, the number of bindings will be > 0
    if len(relations) > 0:
        print("\tRelations between {} and {}: {}".format(first_entity,second_entity,relations)) #for debugging
    return len(relations) > 0

if __name__ == "__main__":
    entity0 = "China"
    entity1 = "Beijing"
    entity2 = "New_York"
    entity3 = "Adam_Rippon"
    entity4 = "Scranton,_Pennsylvania"

    result0 = relation_exists(entity0,entity1)
    print("Relation exists between {} and {}: {}".format(entity0,entity1,result0))

    result1 = relation_exists(entity0,entity2)
    print("Relation exists between {} and {}: {}".format(entity0,entity2,result1))

    result2 = relation_exists(entity3,entity4)
    print("Relation exists between {} and {}: {}".format(entity3,entity4,result2))
