import json
import numpy as np
import os
import pandas as pd


class Persistable:
    """Class to handle loading/saving corpus.

    Args:
        path (str): Path of the directory to save/saved corpus.
        toLoad (bool): Whether to use saved corpus.
    """
    def __init__(self, path: str, toLoad: bool):
        self.pardir = path.strip()
        self.toLoad = toLoad

        if self.pardir.find(".") == -1:  # if not file-like path...
            os.makedirs(self.pardir, exist_ok=True)

            self._suffix = self._get_suffix(self.pardir)
            self._corpus_path = os.path.join(self.pardir, f"corpus_{self._suffix}.tsv")
        else:
            # TODO(amillert) test if works
            self._suffix = self._get_suffix(os.path.dirname(self.pardir))
            self._corpus_path = self.pardir

        self._schema_path = os.path.join(self.pardir, f"schema_{self._suffix}.json")

    def _generate_schema(self, df: pd.DataFrame) -> dict:
        """Function to generate schema for data.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            dict: Have key as column name and value as type of the data stored in the column.
        """
        return {
            col: type(df[col][0]).__name__
            for col in df.columns
        }

    def _get_suffix(self, path: str) -> int:
        return len(list(filter(lambda l: "corpus" in l, os.listdir(path)))) - int(self.toLoad)

    def _cache(self, schema: dict, columns: str, data: np.array) -> None:
        with open(self._schema_path, "w") as fout:
            fout.write(json.dumps(schema, indent=2))

        with open(self._corpus_path, "w") as fout:
            fout.write(f"{columns}\n")
            for row in data:
                line = '\t'.join(map(str, row))
                fout.write(f"{line}\n")

    @staticmethod
    def _build_dataframe(cols: list, data: list, schema: dict) -> pd.DataFrame:
        df = pd.DataFrame(data, columns = cols)

        for col_name in df.columns:
            col_type = schema[col_name]
            if col_type == "list":
                df[col_name] = df[col_name].apply(eval)

        return df

    def _load_cache(self) -> pd.DataFrame:
        """Loads the most recent (number-wise) df and schema

        Returns:
            pd.DataFrame: Data.
        """
        with open(os.path.join(self.pardir, f"corpus_{self._suffix}.tsv"), "r") as fin:
            cols = list(map(str.strip, fin.readline().split("\t")))
            data = list(
                filter(
                    lambda l: l[0],
                    map(
                        lambda l: l.split("\t"),
                        fin.read().split("\n")
                    )
                )
            )

        with open(os.path.join(self.pardir, f"schema_{self._suffix}.json"), "r") as fin:
            schema = json.loads(fin.read())

        return self._build_dataframe(cols, data, schema)
