from corpus.preprocessing import DataBuilder
from handle_wiki.extraction import executeJob
from utils import queriesObject2Category
from utils.argparser import args

from multiprocessing import Pool
import os


if __name__ == "__main__":
    if args.parallel:
        # Unfortunatelly we want a limited amount of persons per occupation;
        # hence, we can only parallelize the whole group

        pool = Pool()
        extractedRows = pool.map(executeJob, queriesObject2Category.items())
        pool.close()
    else:
        extractedRows = list(map(executeJob, queriesObject2Category.items()))

    db = DataBuilder(extractedRows, args)
    df = db.get_df()

    # columns = "\t".join(df.columns.tolist())
    # data = df.values

    db.save()
    db.load()

    # _ = DataBuilder.pickle(df, corpus_path)
    # new_df = DataBuilder.unpickle(df, os.path.join(pardir, f"corpus_{len(os.listdir(pardir)) - 1}.tsv"))
    # print()

    # fin.write(f"{columns}\n")
    # for row in data:
    #     line = '\t'.join(map(str, row))
    #     fout.write(f"{line}\n")

    print()
