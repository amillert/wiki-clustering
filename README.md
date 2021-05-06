# Clustering and Classifying People based on Text and Knowledge Base information

## General overview
The project constitutes of two main subprojects (reflected in terms of subparsers in the argument parser), namely:
1. Corpus extraction and preprocessing,
2. Clustering, Classification; evaluation of the models and visualizations.

### Corpus
Corpus has been extracted from wikidata using sparql. The respective queries provide links to 6 groups of interest - painters, writers, singers, politicians, architects, mathematicians wchich constitute 2 bigger clusters - 3 former - artists, 3 latter - non-artists.

Since wptools, along with the validation of description length to match desired, takes a while to query and process wikipedia data; our scipt allows saving the data after the extraction. This allows one create, for instance, several different corpora to later investigate which parameters yield better results, or simply to extract data once and not have to worry about it ever again.

Initially, even before saving, data is normalized by tokenizing the data, lowercasing all tokens, and filtering meaningless wordforms (function words) not to polute the data. Since our goal doesn't regard understanding the structure of the data, leaving those tokens out in the corpus would only complicate the learning process. Additionally, we remove all the punctuation.

Initial representation is further processed to extract simple NLP-based features such as extraction of tokens matching certain part of speech in their base form (lemmas), and detecting named entities. Based on the outputs, results are being stored in separate columns respectively. Prefix of the column is left from the original column, whilst the suffix contains the information about what has been extracted / processed.

After the whole process, script creates vocabulary and dictionaries mapping tokens to their corresponding ids and to their NL representation. The script allows toggling between the two representation back and forth. The numerical representation is desired format by any machine learning model; whereas NL-form is better understood by humans.

### Clustering / classification

## Installation
Move into the desired location in your local file system and execute the following command:
```bash
$ git clone https://github.com/amillert/wiki-clustering.git
```
Once it's done, all the required files are at your disposal.

The environment.yml file contains all the information to successfully recreate the `anaconda` virtual environment along with the correct python version and all the compatible version of libraries used in the project. In order to build the virtual environment, make sure `anaconda` is availible in your system by typing:
```bash
$ conda --version
```

In case, it's not availible refer to the official manual at https://docs.anaconda.com/anaconda/install/.

If `anaconda` tool is at your disposal, one can proceed and recreate the virtual environment. Simply run the command:
```bash
$ conda env create -f environment.yml
```

The virtual environment's name is `amill_nakaz_2021` and it can be activated by typing:
```bash
$ conda activate amill_nakaz_2021
```

In order to deactivate the `anaconda` virtual environment, one can run the command:
```bash
$ conda deactivate
```

### Using the project
As mentioned before, the general project constitutes two subprojects - `corpus`, and `prediction`. The distinction between the two, and their respective arguments, are reflected in the `utils/argparser.py` file. In a nutshell, `corpus` subparser parses the following arguments: `--num_entires` to specify how many persons to extract per category; `--sentences_per_article` to specify how many sentences to parse per each entry, if there are not enough sentences entry is being discarted; `--parallel` to run extraction in parallel (unfortunatelly script is limited in terms of ability to be parallelized simply by the fact that we need to extract `--num_entires` per category, hence parallelization can only occur per category group, namely it's degree is at most 6 in the most busy time of processing when all categories are still being extracted); `--path_corpus_out` allows one provide the path in the local file system in which extracted corpora are being stored; the corpus `pd.DataFrame` is being stored along with the schema for loading data in the correct format; when `--path_corpus_out` arguement is composed with the `--load_data` flag, it serves as a path to the directory storing all corpora to load it back into the script. In case of loading the dataset, if `--path_corpus_out` is the exact path to specific `*.tsv` file corresponding the  `pd.DataFrame` of one's interest - this exact file with respective schema are being loaded; however, if it denotes just a directory with different corpora, then it loads the corpus with the highest suffix number. In order to preserve compatibility in terms of functionality, when `--path_corpus_out` is used to to save the model, script will automatically generate the path name by increasing the suffix value. The exact name is of form `corpus_{n}.tsv`, where n denotes which corpus it is in the sequence of their generation.

`prediction` subparser to be defined soon - once implemented.

### Exemplar scenarios of running the script
