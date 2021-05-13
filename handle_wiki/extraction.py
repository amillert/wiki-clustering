"""
    Functions to perform SPARQL query.
"""

from utils import wiki_endpoint, baseQuery, DataRow
from utils.argparser import args

import nltk
import traceback
from SPARQLWrapper import SPARQLWrapper, JSON
import wptools


def getItemsRefs(qid: str) -> list:
    """Function to return reference links.

    Args:
        qid (str): Representing unique identifiers for occupation in Wikidata.

    Returns:
        list: References of qid containing Q-number (unique identifier for data in Wikidata).
    """
    sparql = SPARQLWrapper(wiki_endpoint, returnFormat=JSON, agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11")
    sparql.setQuery(baseQuery.format(qid))

    queryResults = sparql.query().convert()["results"]["bindings"]

    return [x["item"]["value"] for x in queryResults]


def extractDataPerOccupation(refs: list, category: str, group: str) -> list:
    """Function to query Wikidata to collect necessary information for the corpus.

    Args:
        refs (list): Contains reference links of all listed person for a particular category in Wikidata.
        category (str): Occupation_name. e.g. "painters", "musicians" etc...
        group (str): Types of category. 
                    For this project there are 'A' denoting 'Artists' and 'X' denoting 'Non artists'

    Returns:
        list: Formatted result of SPARQL query. 
    """
    results = []
    personCounter = 0

    for ref in refs:
        if personCounter >= args.num_entires: break

        try:
            page = wptools.page(wikibase=ref.split("/")[-1])
            page.get_wikidata()
            description = page.data["description"]
            title = page.data["title"]

            content = page.get_query().data["extext"].replace("\n", " ")
        except:
            traceback.print_exc()
            print(f"Can't obtain content for entry: {ref}")
            continue
        else:
            sentences = nltk.sent_tokenize(content)
            print(f"person name: {title}, sentences: {len(sentences)}")

            if len(sentences) > args.sentences_per_article:
                results.append(DataRow(
                    title = title,
                    description = nltk.sent_tokenize(description),
                    content = sentences[:args.sentences_per_article],
                    category = category,
                    group = group
                    ))
                personCounter += 1

                print(end="\n\n\n")
                print(category, personCounter, end="\n\n\n")

    return results


def executeJob(occupationCategory: tuple) -> list:
    """Function to run the SPARQL query given a occupation category.

    Args:
        occupationCategory (tuple): Consisting (qid , occupation_name , category_types)

    Returns:
        list: Results of SPARQL query.
    """
    occupationObject, (category, group) = occupationCategory

    occupationRefs = getItemsRefs(occupationObject)
    rawDataPoints = extractDataPerOccupation(occupationRefs, category, group)

    return rawDataPoints
