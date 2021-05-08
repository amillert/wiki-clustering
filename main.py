"""
    Main script to run the program.
"""

from multiprocessing import Pool

from corpus.preprocessing import DataBuilder, FeaturesGenerator
from handle_wiki.extraction import executeJob
from prediction.clustering import classify, cluster
from utils import queriesObject2Category
from utils.argparser import args
import traceback

def main(args):
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

        except AttributeError:
            exit(1)
        except Exception: 
            traceback.print_exc()
            # probably move it up, so that it doesn't crash
            # Even though, it's already there too...
            print("Wiki error")
            exit(2)
    
    elif args.subparser == "prediction":
        db = DataBuilder(None, args.saved_path, load=True)
        df, schema = db.load()

        fg = FeaturesGenerator(df, schema)
        fg.mutate()
        df_num = fg.toggle_df_representation()

        stacked_vectors = cluster(df, args.num_clusters, "content", args.keep_top_tokens)
        classify(stacked_vectors.toarray(), df_num[["title", "category", "group"]], args)

        # TODO(nami) Do visualization
        if args.visualize:
            pass

if __name__ == "__main__":
    main(args)