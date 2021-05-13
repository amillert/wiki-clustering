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
from utils.visualization import visualize_confussion_matrix, cluster_visualize, classify_visualize


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
        conversion_dics = fg.get_conversion_targets_dics()
        num_sub_df = fg.toggle_df_representation()[["content", "group", "category"]]

        predictor = Predictor(df, num_sub_df, conversion_dics, args)
        predictor.cluster_all()
        clustering_res = predictor.get_clustering_results()
        cluster_visualize(clustering_res)

        predictor.classify_all()
        classification_res = predictor.get_classification_results()
        classify_visualize(classification_res)
