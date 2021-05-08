from collections import defaultdict
import numpy as np
import os
import pandas as pd
import random
from scipy.sparse import hstack, coo_matrix
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import DataLoader

from .model.custom_dataset import VectorizedDataset
from .model.classifier import LogisticRegressor
from utils.funs import reduce, reduce_tensors


class Predictor:
    def __init__(self, df: pd.DataFrame, targets: list, args: list):
        # clustering
        self._df             = df  # one from cluster
        self._num_clusters   = args.num_clusters
        self._features_cols  =  list(
            filter(
                lambda l: l not in ["title", "category", "group"],
                self._df.columns
            )
        )
        self._vectors        = self._vectorize_all(args.keep_top_tokens)
        self.stacked_vectors = hstack(self._vectors)

        # classification
        self._data        = self.stacked_vectors.toarray()
        self._targets     = targets
        self._data_size   = len(self._data)
        self._batch_size  = args.batch_size
        self._num_batches = self._data_size // self._batch_size
        self._epochs      = args.epochs

        # produces X_train, y_train, X_test, y_test
        self._balanced_split(
            int(self._data_size // 6 * 0.1)  # 10% per category in test data
        )

        self.model = LogisticRegressor(
            in_features=self._data.shape[1],  # vector size
            n_hidden=args.n_hidden,
            n_out_classes=6  # writer, singer, painter, politician, mathematician, architect
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.eta, momentum=0.9)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.eta)

    def cluster(self) -> None:
        kmeans = KMeans(
            n_clusters=self._num_clusters,
            random_state=77,
            max_iter=300,
            init="k-means++",
        )
        cluster_res = kmeans.fit(self.stacked_vectors)

        # TODO(amillert): Grid search to find best set of features + hyper-params settings

        print(f"Homogeneity: {metrics.homogeneity_score(self._targets, cluster_res.labels_):0.4f}")
        print(f"Completeness: {metrics.completeness_score(self._targets, cluster_res.labels_):0.4f}")
        print(f"V-measure: {metrics.v_measure_score(self._targets, cluster_res.labels_):0.4f}")
        print(f"Adjusted Rand-Index: {metrics.adjusted_rand_score(self._targets, cluster_res.labels_):.4f}")
        for i, col in enumerate(self._features_cols):
            print(f"Silhouette Coefficient for {col}: {metrics.silhouette_score(self._vectors[i], cluster_res.labels_, sample_size=1000):0.4f}")
        print()

    def classify(self) -> None:
        # TODO(amillert): 1. Probably refactor and put logic into a class
        # TODO(amillert): 2. Data should probably come from this class, not from cluster itself
        # TODO(amillert): 3. Lower eta each few epochs

        self._training()
        self._testing()

    def _training(self):
        batches = DataLoader(
            dataset=VectorizedDataset(self.X_train, self.y_train),
            drop_last=True,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=os.cpu_count()
        )

        for epoch in range(1, self._epochs + 1):
            loss_total = 0.0
            batch_count = 0
            y_pred, y_gold = [], []

            for X, y in batches:
                batch_count += 1

                output = self.model(X)

                loss = self.criterion(output, y)
                loss_total += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                y_pred.append(torch.argmax(output, dim=1))
                y_gold.append(y)

                if not batch_count % 10:
                    print(f"Batch number {batch_count}, avg loss: {loss_total / batch_count:.4f}")

            print("—"*100)
            print(f"Epoch: {epoch} out of {self._epochs}")
            print(f"Mean loss:  {loss_total / self._num_batches:.4f}")
            print(f"Total loss: {loss_total:.4f}")
            self._evaluate(y_gold, y_pred)
            print("—"*100)

    def _testing(self):
        for X, y in DataLoader(
            dataset=VectorizedDataset(self.X_test, self.y_test),
            drop_last=True,
            batch_size=len(self.X_test),
            shuffle=True,
            num_workers=os.cpu_count()
        ): break

        y_pred = torch.argmax(self.model(X), dim=1)

        print("Evaluation of the model")
        self._evaluate(y, y_pred, False)
        print("—"*100)

        print()

    def _vectorize_all(self, keep_top: float) -> list:
        def _vectorize(series: pd.Series) -> coo_matrix:
            tfidf = TfidfVectorizer(
                max_features=int(len({xi for x in series.tolist() for xi in x}) * keep_top),
                use_idf=True,
                ngram_range=(1, 3),
            )

            return tfidf.fit_transform(series.values)

        return [
            _vectorize(self._df[col].apply(" ".join))
            # TODO(amillert): Preferably fix it so there's no need for " ".join
            for col in self._features_cols
        ]

    @staticmethod
    def _evaluate(y_gold: list, y_pred: list, is_train: bool=True) -> None:
        golds = reduce_tensors(y_gold) if is_train else y_gold.tolist()
        preds = reduce_tensors(y_pred) if is_train else y_pred.tolist()

        prec, rec, f1, _ = metrics.precision_recall_fscore_support(golds, preds, average="macro", zero_division=0)
        acc = metrics.accuracy_score(golds, preds)
        print("Macro averaging")
        print(f"precision: {prec:.4f}")
        print(f"recall:    {rec:.4f}")
        print(f"f1 score:  {f1:.4f}")
        print(f"accuracy:  {acc:.4f}")

    def _balanced_split(self, how_many: int):
        splitter = {
            i: defaultdict(list)
            for i in range(6)  # categories
        }

        zipped = list(zip(self._data, self._targets))
        random.shuffle(zipped)  # inplace

        # self._target must be converted to numeric
        for d, t in zipped:
            if len(splitter[t]["X_test"]) < how_many:
                splitter[t]["X_test"].append(d)
                splitter[t]["y_test"].append(t)
            else:
                splitter[t]["X_train"].append(d)
                splitter[t]["y_train"].append(t)
        
        self.X_train = np.array(reduce([splitter[i]["X_train"] for i in range(6)]))
        self.y_train = reduce([splitter[i]["y_train"] for i in range(6)])
        self.X_test  = np.array(reduce([splitter[i]["X_test"] for i in range(6)]))
        self.y_test  = reduce([splitter[i]["y_test"] for i in range(6)])
        