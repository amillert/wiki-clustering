"""
    Main script to run the program.
"""

from multiprocessing import Pool
import traceback

from corpus.preprocessing import DataBuilder, FeaturesGenerator
from handle_wiki.extraction import executeJob
from prediction.run import Predictor
from utils import queriesObject2Category
from utils.argparser import args


if __name__ == "__main__":
    if args.subparser == "corpus":
        try:
            if args.parallel:
                pool = Pool()
                extractedRows = pool.map(executeJob, queriesObject2Category.items())
                pool.close()
            else:
                extractedRows = list(map(executeJob, queriesObject2Category.items()))

            db = DataBuilder(extractedRows, args.save_path, load=False)
            db.save()
        except Exception:
            traceback.print_exc()
            exit(1)
    elif args.subparser == "prediction":
        db = DataBuilder(None, args.saved_path, load=True)
        df, schema = db.load()

        fg = FeaturesGenerator(df, schema)
        fg.mutate()  # gets features in-place
        targets = fg.toggle_df_representation().category.values

        predictor = Predictor(df, targets, args)
        predictor.cluster()
        predictor.classify()

        # TODO(nami) Do visualization
        if args.visualize:
            pass
