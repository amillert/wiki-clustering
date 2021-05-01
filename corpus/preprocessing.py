from utils import DataRow
from corpus.persistance import Persistable

import nltk
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

    def getNormalized(self):
        return self._normalized


class DataBuilder(Persistable):
    def __init__(self, data: list, args: list):
        super(DataBuilder, self).__init__(args.path_corpus_out)
        self._data = data
        self._columns = data[0][0]._fields
        self._df = self._getNormalizedDataFrame()
        self._columns_str = "\t".join(self._df.columns)
        self._schema = self._generate_schema(self._df)

    @staticmethod
    def _normalizeRow(row: DataRow) -> tuple:
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

    def get_df(self):
        return self._df
    
    def save(self) -> None:
        self._cache(self._schema, self._columns_str, self._df.values)

    def load(self):
        return self._load_cache()
