from corpus.preprocessing import DataBuilder
from corpus.preprocessing import FeaturesGenerator
from handle_wiki.extraction import executeJob
# from prediction.clustering import cluster, classify
from prediction.clustering import Predictor
from utils import queriesObject2Category
from utils.argparser import args

from multiprocessing import Pool
import os


if __name__ == "__main__":
    if args.subparser == "corpus":
        try:
            if not args.load_data:
                if args.parallel:
                    pool = Pool()
                    extractedRows = pool.map(executeJob, queriesObject2Category.items())
                    pool.close()
                else:
                    extractedRows = list(map(executeJob, queriesObject2Category.items()))

                db = DataBuilder(extractedRows, args)
                df, schema = db.get_df()

                if args.path_corpus_out:
                    db.save()
            # Maybe for testing, but used below in `prediction` subparser
            elif args.load_data and args.path_corpus_out:
                db = DataBuilder(None, args)
                df, schema = db.load()
        except AttributeError:
            exit(1)
        except:
            # probably move it up, so that it doesn't crash
            # Even though, it's already there too...
            print("Wiki error")
            pass
            # exit(2)
    elif args.subparser == "prediction":
        args.load_data = True
        db = DataBuilder(None, args)
        df, schema = db.load()

        fg = FeaturesGenerator(df, schema)
        fg.mutate()  # gets features in-place

        predictor = Predictor(df, args)
        predictor.cluster()
        predictor.classify()
