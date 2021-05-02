from corpus.preprocessing import DataBuilder
from corpus.preprocessing import FeaturesGenerator
from handle_wiki.extraction import executeJob
from utils import queriesObject2Category
from utils.argparser import args

from multiprocessing import Pool
import os


if __name__ == "__main__":
    try:
        if not args.load_data:
            if args.parallel:
                pool = Pool()
                extractedRows = pool.map(executeJob, queriesObject2Category.items())
                pool.close()
            else:
                extractedRows = list(map(executeJob, queriesObject2Category.items()))

            db = DataBuilder(extractedRows, args)
            # schema = db.get_schema()
            df, schema = db.get_df()

            if args.path_corpus_out:
                db.save()
        elif args.load_data and args.path_corpus_out:
            db = DataBuilder(None, args)
            df, schema = db.load()
        # else:
        #     raise AttributeError("Probably missing --path_corpus_out flag")
    except AttributeError:
        exit(1)
    # except:
    #     print("Wiki error")
    #     exit(2)

    fg = FeaturesGenerator(df, schema)
    