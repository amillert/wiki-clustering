from utils import DataRow
from corpus.persistance import Persistable

from functools import reduce
import nltk
from nltk.tag import StanfordNERTagger
import pandas as pd


class TextNormalizer:
    def __init__(self, text: list):
        self._normalized = self._normalize(text)

    @staticmethod
    def _tokenize(text: list) -> list:
        return [nltk.word_tokenize(sentence) for sentence in text]

    def _lowercase(self, text: list) -> list:
        return [[token.strip().lower() for token in sentence]
                for sentence in self._tokenize(text)]

    def _removeStopWords(self, text: list) -> list:
        stopwords = nltk.corpus.stopwords.words("english")

        return [[token for token in sentence if token not in stopwords]
                for sentence in self._lowercase(text)]

    def _normalize(self, text: list) -> list:
        # return self._removeStopWords(text)
        return [xi for x in self._removeStopWords(text) for xi in x]

    def getNormalized(self) -> list:
        return self._normalized


class FeaturesGenerator:
    """
    Get NERs, POSs, etc.
    """

    def __init__(self, df: pd.DataFrame, schema: dict):
        self._df = df
        self._schema = schema
        self._lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        self._list_features = [k for k, v in self._schema.items() if v == "list"]

        # implemented by self._generate_dictionaries
        # which is triggered with self.mutate
        self._vocabulary = None
        self._token2idx = None
        self._idx2token = None


    def get_df(self) -> pd.DataFrame:
        return self._df

    def mutate(self):
        """
        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
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

    def _filter_pos_and_lemmatize(self, suffix: str, tag_rules: list) -> None:
        for feature in self._list_features:
            for pos, (rule, lemma) in tag_rules.items():
                col_in, col_out = f"{feature}_{suffix}", f"{feature}_{pos}"
                self._df[col_out] = self._df[col_in].apply(
                    lambda l: self._lemmatize_from_pos(l, lemma, rule)
                )

    def _lemmatize_from_pos(self, record: list, lemma: str, rule: str) -> list:
        return list(
            map(
                lambda m: self._lemmatizer.lemmatize(m[0], lemma),
                filter(lambda f: f[1].startswith(rule), record)
            )
        )

    def _generate_dictionaries(self) -> None:
        vocabs = (
            {xi for x in self._df[listCol].values.tolist() for xi in x}
            for listCol in self._list_features
        )

        self._vocabulary = reduce(lambda x, y: x | y, vocabs)
        self._token2idx = {token: i for i, token in enumerate(self._vocabulary)}
        self._idx2token = {v: k for k, v in self._token2idx.items()}


class DataBuilder(Persistable):
    def __init__(self, data: list, args):
        super(DataBuilder, self).__init__(args.path_corpus_out, args.load_data)

        # implemented if args.load_data flag set or when loading data with self.load
        self._df = None
        self._schema = None

        if not args.load_data:
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
