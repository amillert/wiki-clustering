from utils import DataRow

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


class DataBuilder:
    def __init__(self, data: list):
        self._data = data
        self._columns = data[0][0]._fields

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

    def getNormalizedDataFrame(self) -> pd.DataFrame:
        rows = list(map(self._normalizeRow, self._flattenRows()))

        return pd.DataFrame(rows, columns=self._columns)
