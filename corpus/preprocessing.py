"""
    Contains classes and functions to manage the documents used as data. 
"""

from functools import reduce

import nltk
from nltk.tag import StanfordNERTagger
import pandas as pd

from corpus.persistance import Persistable
from utils import DataRow


class TextNormalizer:
    """Class that handles normalizing text.

    Args:
        text (list): list of texts to be processed. 
    """
    def __init__(self, text: list):
        self._normalized = self._normalize(text)

    def _lowercase(self, text: list) -> list:
        return [[token.strip().lower() for token in sentence]
                for sentence in self._tokenize(text)]

    def _removeStopWords(self, text: list) -> list:
        stopwords = nltk.corpus.stopwords.words("english")

        return [[token for token in sentence if token not in stopwords]
                for sentence in self._lowercase(text)]

    def _normalize(self, text: list) -> list:
        return [xi for x in self._removeStopWords(text) for xi in x]

    def getNormalized(self) -> list:
        return self._normalized

    @staticmethod
    def _tokenize(text: list) -> list:
        return [nltk.word_tokenize(sentence) for sentence in text]

class FeaturesGenerator:
    """Generate various features from corpus.

    Args:
        df (pd.DataFrame): Data. 
        schema (dict): Contains information about columns.
    """
    def __init__(self, df: pd.DataFrame, schema: dict):
        self._df              = df
        self._is_df_numeric   = False  # for conversion
        self._converted_df    = pd.DataFrame()
        self._schema          = schema
        self._lemmatizer      = nltk.stem.wordnet.WordNetLemmatizer()
        self._list_features   = [k for k, v in self._schema.items() if v == "list"]
        self._target_features = ["category", "group"]

        # denotes absence of lemmas
        self._fake = "<FAKE_TOKEN>"

        # implemented by self._generate_dictionaries
        # which is triggered with self.mutate
        self._lemmas       = set()
        self._vocabulary   = set()
        self._token2idx    = dict()
        self._idx2token    = dict()
        self._category2idx = dict()
        self._idx2category = dict()
        self._group2idx    = dict()
        self._idx2group    = dict()

    def get_df(self) -> pd.DataFrame:
        self._is_df_numeric = not self._is_df_numeric  # toggle flag

        return self._converted_df if self._is_df_numeric else self._df

    def mutate(self):
        """Process the data.

        Ref: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        """
        tag_rules = {
                 "VERB": ("V",  "v"),
                 "NOUN": ("N",  "n"),
               "ADVERB": ("RB", "r"),
            "ADJECTIVE": ("J",  "a"),
        }

        self._preserve_alphas()
        self._pos_annotate("POS")
        self._filter_pos_and_lemmatize("POS", tag_rules)
        self._generate_dictionaries()
    
    def toggle_df_representation(self) -> pd.DataFrame:
        """Toggle str of text to numerical representation and vice versa.

        Returns:
            pd.DataFrame: Data.
        """
        if not self._converted_df.empty:
            return self.get_df()

        self._converted_df["title"] = self._df["title"]

        colsToConvertTokens = filter(
            lambda l: l not in ["title"] + self._target_features,
            self._df.columns
        )

        for col in colsToConvertTokens:
            self._converted_df[col]    = self._df[col].apply(self._convert_tokens)
        self._converted_df["category"] = self._df["category"].apply(self._convert_category)
        self._converted_df["group"]    = self._df["group"].apply(self._convert_group)

        assert self._df.shape == self._converted_df.shape, "shapes' missmatch"

        self._is_df_numeric = not self._is_df_numeric
        return self._converted_df

    def _preserve_alphas(self) -> None:
        for feature in self._list_features:
            self._df[feature] = self._df[feature].apply(self._is_alpha)

    @staticmethod
    def _is_alpha(l: list) -> list:
        return list(filter(lambda x: x.isalpha(), l))

    def _pos_annotate(self, suffix: str) -> None:
        for feature in self._list_features:
            self._df[f"{feature}_{suffix}"] = self._df[feature].apply(
                nltk.pos_tag)

    def _filter_pos_and_lemmatize(self, suffix: str, tag_rules: dict) -> None:
        """Filter by specific POS, then lemmatize the token.

        Args:
            suffix (str): suffix used in column to contain POS.
            tag_rules (dict): maps certain Universal POS to Penn Treebanks POS.
        """
        for feature in self._list_features:
            col_in = f"{feature}_{suffix}"
            for pos, (rule, lemma) in tag_rules.items():
                col_out = f"{feature}_{pos}"
                self._df[col_out] = self._df[col_in].apply(
                    lambda l: self._lemmatize_from_pos(l, lemma, rule)
                )
                self._lemmas |= {xi for x in self._df[col_out].tolist() for xi in x if x}
            self._df.drop(col_in, axis=1, inplace=True)

    def _lemmatize_from_pos(self, record: list, lemma: str, rule: str) -> list:
        if not record:
            return [self._fake]
        return list(
            map(
                lambda m: self._lemmatizer.lemmatize(m[0], lemma),
                filter(lambda f: f[1].startswith(rule), record)
            )
        )

    def _generate_dictionaries(self) -> None:
        vocabs = (
            {xi for x in self._df[col].values.tolist() for xi in x}
            # for col in self._list_features
            for col in filter(
                lambda l: l not in ["title"] + self._target_features,
                self._df.columns
            )
        )

        self._vocabulary   = reduce(lambda x, y: x | y, vocabs)
        self._token2idx    = {token: i for i, token in enumerate(self._vocabulary)}
        self._idx2token    = {v: k for k, v in self._token2idx.items()}
        self._category2idx = {cat: i for i, cat in enumerate(set(self._df.category))}
        self._idx2category = {v: k for k, v in self._category2idx.items()}
        self._group2idx    = {group: i for i, group in enumerate(set(self._df.group))}
        self._idx2group    = {v: k for k, v in self._group2idx.items()}
    
    def _convert_tokens(self, row: list) -> list:
        return list(map(self._token2idx.get, row))

    def _convert_category(self, row: str) -> int:
        return self._category2idx[row]

    def _convert_group(self, row: str) -> int:
        return self._group2idx[row]


class DataBuilder(Persistable):
    """Class to build collected texts to formatted data.

    Args:
        data (list): Contains raw text data. Pass None is only loading the data from saved ones.
        path (str): Path to a directory to save/saved data.
        load (bool, optional): Whether to load the saved data or not. Defaults to False.

    """
    def __init__(self, data: list, path: str, load: bool=False):
        super(DataBuilder, self).__init__(path, load)

        self._df = None
        self._schema = None

        if not load:
            self._data = data
            self._columns = data[0][0]._fields
            self._df = self._getNormalizedDataFrame()
            self._columns_str = "\t".join(self._df.columns)
            self._schema = self._generate_schema(self._df)

    @staticmethod
    def _normalizeRow(row: DataRow) -> (str, list, list, str, str):
        title, description, content, category, group = row

        return (
            title,
            TextNormalizer(description).getNormalized(),
            TextNormalizer(content).getNormalized(),
            category,
            group
        )

    def _flattenRows(self) -> list:
        return [xi for x in self._data for xi in x]

    def _getNormalizedDataFrame(self) -> pd.DataFrame:
        rows = list(map(self._normalizeRow, self._flattenRows()))

        return pd.DataFrame(rows, columns=self._columns)

    def get_schema(self) -> dict:
        return self._schema

    def get_df(self) -> (pd.DataFrame, dict):
        return self._df, self._schema

    def save(self) -> None:
        self._cache(self._schema, self._columns_str, self._df.values)

    def load(self) -> pd.DataFrame:
        self._df = self._load_cache()
        self._schema = self._generate_schema(self._df)

        return self._df, self._schema
