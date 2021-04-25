from handle_wiki.extraction import *
from utils import queriesObject2Category
from utils.argparser import args

from multiprocessing import Pool


if __name__ == "__main__":
    if args.parallel:
        # Unfortunatelly we want a limited amount of persons per occupation;
        # hence, we can only parallelize the whole group

        pool = Pool()

        outs = pool.map(executeJob, queriesObject2Category.items())
        # 48 s

        pool.close()
    else:
        outs = list(map(executeJob, queriesObject2Category.items()))
        # 140 s
