"""
    Contains methods to perform the machine learning algorithms. 
"""

from collections import defaultdict, namedtuple
import os
import random

import numpy as np
import pandas as pd
from scipy.sparse import hstack, coo_matrix
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import torch
from torch.utils.data import DataLoader

from .model.custom_dataset import VectorizedDataset
from .model.classifier import LogisticRegressor
from utils import ClusterMetrics
from utils.flatten import flatten, flatten_tensors
from utils.visualization import visualize_confussion_matrix


class Predictor:
    def __init__(self, df: pd.DataFrame, targets: pd.DataFrame, conversion_dics: tuple, args: list):
        """Class contains classifier and cluster

        Args:
            df (pd.DataFrame): Input data.
            targets (pd.DataFrame): Correct output (both categories and groups).
            args (namedtuple):
        """
        # clustering
        self._df             = self._drop_empty(df)  # one from cluster
        self._idx2category, self._idx2group = conversion_dics
        self._num_clusters   = args.num_clusters
        self._features_cols  =  list(
            filter(
                lambda l: l not in ["title", "category", "group"],
                self._df.columns
            )
        )
        self._keep_top        = args.keep_top_tokens
        self._vectors         = self._vectorize_all()
        self._tokens          = pd.get_dummies(self._df.content.apply(pd.Series))
        self._token_freqs     = self._vectorize(self._df.content.apply(" ".join), idf=False)
        self._stacked_vectors = hstack(self._vectors)

        # classification
        self._data              = self._stacked_vectors.toarray()
        self._target_groups     = targets.group.values
        self._target_categories = targets.category.values
        self._data_size         = len(self._data)
        self._batch_size        = args.batch_size
        self._num_batches       = self._data_size // self._batch_size
        self._epochs            = args.epochs

        test_size = int(self._data_size // 6 * 0.1)
        # produces X_train, y_train, X_test, y_test for categories
        self._balanced_split(
            test_size,  # 10% per category in test data
            categories=True
        )
        # produces X_train, y_train, X_test, y_test for groups
        self._balanced_split(
            test_size,  # 10% per group in test data
            categories=False
        )

        self.model_categories     = LogisticRegressor(
            in_features=self._data.shape[1],  # vector size
            n_hidden=args.n_hidden,
            n_out_classes=6  # writer, singer, painter, politician, mathematician, architect
        )
        self.criterion_categories = torch.nn.CrossEntropyLoss()
        self.optimizer_categories = torch.optim.SGD(
            self.model_categories.parameters(),
            lr=args.eta, momentum=0.9
        )

        self.model_groups     = LogisticRegressor(
            in_features=self._data.shape[1],  # vector size
            n_hidden=args.n_hidden,
            n_out_classes=2  # artists, non-artists
        )
        self.criterion_groups = torch.nn.CrossEntropyLoss()
        self.optimizer_groups = torch.optim.SGD(
            self.model_groups.parameters(),
            lr=args.eta, momentum=0.9
        )

        # for visualization
        self._cluster_res  = []
        self._classify_res = []

    def cluster_all(self) -> None:
        self.cluster(categories=False)
        self.cluster(categories=True)

    def cluster(self, categories: bool) -> None:
        clusters = 6 if categories else 2
        targets = self._target_categories if categories else self._target_groups
        # 3 models since there are 3 methods
        kmeans1 = KMeans(n_clusters=clusters, random_state=7, max_iter=300, init="k-means++")
        kmeans2 = KMeans(n_clusters=clusters, random_state=7, max_iter=300, init="k-means++")
        kmeans3 = KMeans(n_clusters=clusters, random_state=7, max_iter=300, init="k-means++")

        cluster_res_all = [
            kmeans1.fit(self._tokens),
            kmeans2.fit(self._token_freqs),
            kmeans3.fit(self._stacked_vectors)
        ]

        for cluster_res in cluster_res_all:
            homogeneity       = metrics.homogeneity_score(targets, cluster_res.labels_)
            completeness      = metrics.completeness_score(targets, cluster_res.labels_)
            vMeasure          = metrics.v_measure_score(targets, cluster_res.labels_)
            adjustedRandIndex = metrics.adjusted_rand_score(targets, cluster_res.labels_)
            silhouette        = np.mean([metrics.silhouette_score(
                self._vectors[i],
                cluster_res.labels_,
                sample_size=1000
                ) for i, col in enumerate(self._features_cols)
            ])

            self._cluster_res.append(
                ClusterMetrics(
                    homogeneity,
                    completeness,
                    vMeasure,
                    adjustedRandIndex,
                    silhouette
                )
            )

    def classify_all(self) -> None:
        self.classify(categories=False)
        self.classify(categories=True)

    def classify(self, categories: bool) -> None:
        self._training(categories)
        self._testing(categories)

    def get_clustering_results(self):
        def merge_metrics(xs):
            merged = defaultdict(list)

            for d in map(lambda l: l._asdict(), xs):
                for k, v in d.items():
                    merged[k].append(v)

            return merged

        return {
            "2cluster": merge_metrics(self._cluster_res[:3]), # first three (token, token frequency, tf-idf) belong to group evaluation
            "6cluster": merge_metrics(self._cluster_res[3:]) # rest belong to category evaluation
        }

    def get_classification_results(self):
        return {
            "Group": self._classify_res[:2],  # first two belong to group evaluation
            "Category": self._classify_res[2:]  # next six belong to category evaluation
        }

    def _training(self, categories: bool):
        model     = self.model_categories if categories else self.model_groups
        criterion = self.criterion_categories if categories else self.criterion_groups
        optimizer = self.optimizer_categories if categories else self.optimizer_groups
        X_y_pair  = (
            (self.X_train_categories, self.y_train_categories) if categories
            else (self.X_train_groups, self.y_train_groups)
        )

        batches = DataLoader(
            dataset=VectorizedDataset(*X_y_pair),
            drop_last=True,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=os.cpu_count()
        )

        print(f"Model training for {'Category' if categories else 'Group'}")

        for epoch in range(1, self._epochs + 1):
            loss_total = 0.0
            batch_count = 0
            y_pred, y_gold = [], []

            for X, y in batches:
                batch_count += 1

                output = model(X)

                loss = criterion(output, y)
                loss_total += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                y_pred.append(torch.argmax(output, dim=1))
                y_gold.append(y)

            print("—"*100)
            print(f"Epoch: {epoch} out of {self._epochs}")
            print(f"Mean loss:  {loss_total / self._num_batches:.4f}")
            print(f"Total loss: {loss_total:.4f}")
            self._evaluate(y_gold, y_pred)

        print("—"*100)

    def _testing(self, categories: bool):
        model = self.model_categories if categories else self.model_groups
        X_y_pair = (
            (self.X_train_categories, self.y_train_categories) if categories
            else (self.X_train_groups, self.y_train_groups)
        )

        for X, y in DataLoader(
            dataset=VectorizedDataset(*X_y_pair),
            drop_last=True,
            batch_size=len(X_y_pair[1]),
            shuffle=True,
            num_workers=os.cpu_count()
        ): break

        y_pred = torch.argmax(model(X), dim=1)

        print(f"Evaluation of the model for {'Category' if categories else 'Group'} in testing")
        self._evaluate(y, y_pred, False)
        print("—"*100)

        print()

    def _vectorize(self, series: pd.Series, idf: bool) -> coo_matrix:
        tfidf = TfidfVectorizer(
            max_features=int(len({xi for x in series.tolist() for xi in x}) * self._keep_top),
            use_idf=idf,
            ngram_range=(1, 3),
            lowercase=False,
        )

        return tfidf.fit_transform(series.values)

    def _vectorize_all(self) -> list:
        return [
            # TODO(amillert): Preferably fix it so there's no need for " ".join
            self._vectorize(self._df[col].apply(" ".join), idf=True)
            for col in self._features_cols
            # if list(filter(lambda l: l, self._df[col].tolist()))
        ]

    def _evaluate(self, y_gold: list, y_pred: list, is_train: bool=True) -> None:
        # check if list of tensors
        golds = flatten_tensors(y_gold) if is_train else y_gold.tolist()
        preds = flatten_tensors(y_pred) if is_train else y_pred.tolist()

        prec, rec, f1, _ = metrics.precision_recall_fscore_support(
            golds,
            preds,
            average="macro",
            zero_division=0
        )
        acc = metrics.accuracy_score(golds, preds)

        print()
        print("Macro averaging")
        print(f"precision: {prec:.4f}")
        print(f"recall:    {rec:.4f}")
        print(f"f1 score:  {f1:.4f}")
        print(f"accuracy:  {acc:.4f}")

        if not is_train:
            folds = set(golds)
            convert = self._idx2category if len(folds) == 6 else self._idx2group
            for fold in folds:
                # gpf -> gold_positions_per_fold
                # ppf -> pred_positions_per_fold
                gpf = self._groupby(fold, golds)
                ppf = self._groupby(fold, preds)

                acc = len(gpf & ppf) / len(gpf)  # accuracy or recall...?
                self._classify_res.append((convert[fold], acc))

            visualize_confussion_matrix(golds, preds)

    @staticmethod
    def _groupby(fold: int, xs: list) -> set:
        return set(
            map(
                lambda l: l[0],
                filter(
                    lambda l: l[1] == fold,
                    map(
                        lambda l: l,
                        enumerate(xs)
                    )
                )
            )
        )

    def _balanced_split(self, how_many: int, categories: bool):
        """Split data.

        Args:
            how_many (int): Number of test data to have.
            categories (bool): True if 6 categories else 2 groups.
        """
        uniq_vals = 6 if categories else 2
        targets   = self._target_categories if categories else self._target_groups

        splitter = {
            i: defaultdict(list)
            for i in range(uniq_vals)
        }

        zipped = list(zip(self._data, targets))
        random.shuffle(zipped)  # inplace

        for d, t in zipped:  # data, target
            if len(splitter[t]["X_test"]) < how_many:
                splitter[t]["X_test"].append(d)
                splitter[t]["y_test"].append(t)
            else:
                splitter[t]["X_train"].append(d)
                splitter[t]["y_train"].append(t)

        if categories:
            self.X_train_categories = np.array(flatten([splitter[i]["X_train"] for i in range(uniq_vals)]))
            self.X_test_categories  = np.array(flatten([splitter[i]["X_test"] for i in range(uniq_vals)]))
            self.y_train_categories = flatten([splitter[i]["y_train"] for i in range(uniq_vals)])
            self.y_test_categories  = flatten([splitter[i]["y_test"] for i in range(uniq_vals)])
        else:
            self.X_train_groups = np.array(flatten([splitter[i]["X_train"] for i in range(uniq_vals)]))
            self.X_test_groups  = np.array(flatten([splitter[i]["X_test"] for i in range(uniq_vals)]))
            self.y_train_groups = flatten([splitter[i]["y_train"] for i in range(uniq_vals)])
            self.y_test_groups  = flatten([splitter[i]["y_test"] for i in range(uniq_vals)])

    def _drop_empty(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [col for col in df.columns if list(filter(lambda l: not l, df[col].tolist()))]

        return df.drop(cols_to_drop, axis=1, inplace=False)
