""" 
    Collection of the arguments needed in the project.
"""

import argparse
import sys


argument_parser = argparse.ArgumentParser(
    prog="",
    description="Argument parser of the Data Science project.",
    epilog="Scrape (SPARQL, wptools) and then cluster the data obtained from wikipedia",
    allow_abbrev=True
)

argument_parser.version = "0.1"

subparsers = argument_parser.add_subparsers()

corpus_subparser = subparsers.add_parser(
    "corpus",
    help="Subparser for extracting and preprocessing corpus data."
)

corpus_subparser.add_argument(
    "-n",
    "--num_entires",
    action="store",
    type=int,
    help="Provide how many persons per occupation",
)

corpus_subparser.add_argument(
    "-k",
    "--sentences_per_article",
    action="store",
    type=int,
    help="Provide how many sentences per article (less will not be saved)",
)

corpus_subparser.add_argument(
    "-p",
    "--parallel",
    action="store_true",
    help="If flag set - run parsing (maybe other part as well) in parallel",
)

corpus_subparser.add_argument(
    "-s",
    "--save_path",
    action="store",
    help="Path to save raw extracted corpus to",
    required=True,
)

prediction_subparser = subparsers.add_parser(
    "prediction",
    help="Subparser for clustering/classification."
)

prediction_subparser.add_argument(
    "-n",
    "--num_clusters",
    action="store",
    type=int,
    help="Provide how many clusters to group dataset into",
    required=True,
)

prediction_subparser.add_argument(
    "-b",
    "--batch_size",
    action="store",
    type=int,
    help="Provide the size of mini-batch for classification",
    required=True,
)

prediction_subparser.add_argument(
    "-x",
    "--epochs",
    action="store",
    type=int,
    help="Provide the amount of epochs to learn",
    required=True,
)

prediction_subparser.add_argument(
    "-t",
    "--eta",
    action="store",
    type=float,
    help="Provide learning rate for classification model",
    required=True,
)

prediction_subparser.add_argument(
    "-k",
    "--keep_top_tokens",
    action="store",
    type=float,
    help="Provide the percentage of top tokens to be kept after tfidf process",
    required=True,
)

prediction_subparser.add_argument(
    "-s",
    "--saved_path",
    action="store",
    help="Path to saved corpus for loading",
    required=True,
)

prediction_subparser.add_argument(
    "-d",
    "--n_hidden",
    action="store",
    type=int,
    help="Provide the amount of hidden units",
    required=True,
)

args = argument_parser.parse_args()
args.subparser = sys.argv[1]
