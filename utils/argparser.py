import argparse


argument_parser = argparse.ArgumentParser(
    prog="",
    description="Argument parser of the Data Science project",
    epilog="Scrape (SPARQL, wptools) and then cluster data obtained from wikipedia",
    allow_abbrev=True
)
argument_parser.version = "0.1"

argument_parser.add_argument(
    "-n",
    "--num_entires",
    action="store",
    type=int,
    help="Provide how many persons per occupation",
    required=True,
)
argument_parser.add_argument(
    "-k",
    "--sentences_per_article",
    action="store",
    type=int,
    help="Provide how many sentences per article (less will not be saved)",
    required=True,
)
argument_parser.add_argument(
    "-p",
    "--parallel",
    action="store_true",
    help="If flag set - run parsing (maybe other part as well) in parallel",
)

args = argument_parser.parse_args()
