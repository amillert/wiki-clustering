from corpus.preprocessing import DataBuilder
from handle_wiki.extraction import executeJob
from utils import queriesObject2Category
from utils.argparser import args

from multiprocessing import Pool


if __name__ == "__main__":
    if args.parallel:
        # Unfortunatelly we want a limited amount of persons per occupation;
        # hence, we can only parallelize the whole group

        pool = Pool()
        extractedRows = pool.map(executeJob, queriesObject2Category.items())
        pool.close()
    else:
        extractedRows = list(map(executeJob, queriesObject2Category.items()))
    
    df = DataBuilder(extractedRows).getNormalizedDataFrame()
