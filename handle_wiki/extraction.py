from utils import wiki_endpoint, baseQuery, DataRow
from utils.argparser import args

import nltk
import threading
import wptools
from SPARQLWrapper import SPARQLWrapper, JSON


def getItemsRefs(o: str) -> list:
    sparql = SPARQLWrapper(wiki_endpoint, returnFormat=JSON)
    sparql.setQuery(baseQuery.format(o))

    queryResults = sparql.query().convert()["results"]["bindings"]

    return [x["item"]["value"] for x in queryResults]


def extractDataPerOccupation(refs: list, category: str, group: str) -> list:
    results = []
    personCounter = 0

    for ref in refs:
        if personCounter >= args.num_entires: break
        
        page = wptools.page(wikibase=ref.split("/")[-1])
        page.get_wikidata()
        
        description = page.data["description"]
        title = page.data["title"]
        try:
            content = page.get_query().data["extext"].replace("\n", " ")
        except:
            # print(f"Can't obtain content for entry: {title}")
            pass
        else:
            sentences = nltk.sent_tokenize(content)
            print(title, len(sentences))

            if len(sentences) > args.sentences_per_article:
                results.append(DataRow(
                    title,
                    nltk.sent_tokenize(description),
                    sentences[:args.sentences_per_article],
                    category,
                    group
                    ))
                personCounter += 1

                print(end="\n\n\n")
                print(threading.get_native_id(), personCounter, end="\n\n\n")

    return results


def executeJob(occupationCategory: tuple) -> list:
    # TODO(amillert): optional splitIntoBatches
    # for now not so important cause I have 12 cores

    occupationObject, (category, group) = occupationCategory

    occupationRefs = getItemsRefs(occupationObject)
    rawDataPoints = extractDataPerOccupation(occupationRefs, category, group)

    return rawDataPoints

def splitIntoBatches(queriesObjects):
    # import os
    # cpus = os.cpu_count()
    pass
