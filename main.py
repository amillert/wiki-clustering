from handle_wiki.extraction import *
from utils import queriesObject2Category
from utils.argparser import args

from multiprocessing import Pool
import sys
import time


if __name__ == "__main__":
    pool = Pool()

    if args.parallel:
        tok = time.time()
        # Unfortunatelly we want a limited amount of persons per occupation;
        # hence, we can only parallelize the whole group
        outs = pool.map(executeJob, queriesObject2Category.items())

        tik = time.time()

        pool.close()
        pool.join()
        println(f"It took: {tik - tok} in parallel")
        exit(12)

    # TODO(albert): Add argparser with parallel as argument
    # else:
    #     Sequential
    #     tock = time.time()
    #     for queryObject in queriesObjectsAll:
    #         res = executeJob(queryObject)
    #     tick = time.time()
    #     println(f"It took: {tick - tock} in sequentially")
