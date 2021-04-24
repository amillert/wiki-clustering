wiki_endpoint = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"

baseQuery = "SELECT ?item WHERE {{ ?item wdt:P31 wd:Q5; wdt:P106 wd:{}. }}"

queryOccupationObjectPainters       = "Q1028181"
queryOccupationObjectSingers        = "Q177220"
queryOccupationObjectWriters        = "Q36180"
queryOccupationObjectArchitects     = "Q42973"
queryOccupationObjectPoliticians    = "Q82955"
queryOccupationObjectMathematicians = "Q170790"

queriesObject2Category = {
    queryOccupationObjectPainters:       ("painters", "A"),
    queryOccupationObjectSingers:        ("singers", "A"),
    queryOccupationObjectWriters:        ("writers", "A"),
    queryOccupationObjectArchitects:     ("architects", "Z"),
    queryOccupationObjectPoliticians:    ("politicians", "Z"),
    queryOccupationObjectMathematicians: ("mathematicians", "Z")
}
