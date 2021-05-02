import ast
import json
import numpy as np
import os
import pandas as pd


class Persistable:
    """
    Makes sense only if --path_corpus_out flag provided; otherwise methods should fail
    """

    def __init__(self, path_corpus_out: str, toLoad: bool):
        self.pardir = path_corpus_out.strip()
        self.toLoad = toLoad

        if os.path.isdir(self.pardir):
            os.makedirs(self.pardir, exist_ok=True)

            self._suffix = self._get_suffix(self.pardir)
            self._corpus_path = os.path.join(self.pardir, f"corpus_{self._suffix}.tsv")
        else:
            # TODO(albert) test if works
            self._suffix = self._get_suffix(os.path.dirname(self.pardir))
            self._corpus_path = self.pardir

        self._schema_path = os.path.join(self.pardir, f"schema_{self._suffix}.json")

    def _generate_schema(self, df: pd.DataFrame) -> dict:
        # return json.dumps({
        return {
            col: type(df[col][0]).__name__
            for col in df.columns
        } #)
        # }, indent=2)

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
                # df[col_name].apply(ast.literal_eval)
                df[col_name] = df[col_name].apply(eval)
                print("check")
            # df[col_name].apply(eval(col_type))

        return df

    def _load_cache(self) -> pd.DataFrame:
        """
        Loads the most recent (number-wise) df and schema
        """

        # self._suffix = len(os.listdir(self.pardir)) - 1 
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
