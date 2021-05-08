from corpus.preprocessing import DataBuilder
from corpus.preprocessing import FeaturesGenerator
from handle_wiki.extraction import executeJob
from prediction.run import Predictor
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
        except:
            exit(1)
    elif args.subparser == "prediction":
        args.load_data = True
        db = DataBuilder(None, args)
        df, schema = db.load()

        fg = FeaturesGenerator(df, schema)
        fg.mutate()  # gets features in-place
        targets = fg.toggle_df_representation().category.values

        predictor = Predictor(df, targets, args)
        predictor.cluster()
        predictor.classify()
