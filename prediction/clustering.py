import numpy as np
import os
import pandas as pd
from scipy.sparse import hstack
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import DataLoader

from utils.model.custom_dataset import VectorizedDataset
from utils.model.classifier import LogisticRegressor
from utils.funs import reduce


def cluster(df: pd.DataFrame, num_clusters: int, col: str, keep_top: float) -> None:
    # TODO(amillert): Update return type
    def _vectorize(series: pd.Series):
        tfidf = TfidfVectorizer(
            max_features=int(len({xi for x in series.tolist() for xi in x}) * keep_top),
            use_idf=True,
            ngram_range=(1, 3),
        )

        return tfidf.fit_transform(series.values)

    cols = list(filter(lambda l: l not in ["title", "category", "group"], df.columns))
    tmp_vectors = [
        _vectorize(df[col].apply(" ".join))
        # _vectorize(df[col])
        for col in cols
    ]
    stacked_vectors = hstack(tmp_vectors)

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=77,
        max_iter=300,
        init="k-means++",
    )
    cluster_res = kmeans.fit(stacked_vectors)

    # TODO(amillert): Grid search to find best set of features + hyper-params settings

    print(f"Homogeneity: {metrics.homogeneity_score(df.category, cluster_res.labels_):0.4f}")
    print(f"Completeness: {metrics.completeness_score(df.category, cluster_res.labels_):0.4f}")
    print(f"V-measure: {metrics.v_measure_score(df.category, cluster_res.labels_):0.4f}")
    print(f"Adjusted Rand-Index: {metrics.adjusted_rand_score(df.category, cluster_res.labels_):.4f}")
    for i, col in enumerate(cols):
        print(f"Silhouette Coefficient for {col}: {metrics.silhouette_score(tmp_vectors[i], cluster_res.labels_, sample_size=1000):0.4f}")
    
    return stacked_vectors

def evaluate(y_gold: list, y_pred: list) -> None:
    golds = reduce(y_gold)
    preds = reduce(y_pred)

    prec, rec, f1, _ = metrics.precision_recall_fscore_support(golds, preds, average="macro", zero_division=0)
    acc = metrics.accuracy_score(golds, preds)
    print(f"accuracy:  {acc:.4f}")
    print()
    print("Macro averaging")
    print(f"precision: {prec:.4f}")
    print(f"recall:    {rec:.4f}")
    print(f"f1 score:  {f1:.4f}")
    # print()
    # print("Micro averaging")  # all equal to accuracy
    # prec, rec, f1, _ = metrics.precision_recall_fscore_support(golds, preds, average="micro", zero_division=0)
    # print(f"precision: {prec:.4f}")
    # print(f"recall:    {rec:.4f}")
    # print(f"f1 score:  {f1:.4f}")

def classify(data: np.array, targets: pd.DataFrame, args: list) -> None:
    # TODO(amillert): So far it's train data only (implement testing?)
    # TODO(amillert): Probably refactor and put logic into a class
    # TODO(amillert): data should probably come from this class, not from cluster itself
    data_size   = len(data)
    n_features  = data.shape[1]
    num_batches = data_size // args.batch_size

    batches = DataLoader(
        dataset=VectorizedDataset(data, targets.category.values),
        drop_last=True,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count()
    )
    model = LogisticRegressor(
        in_features=n_features,
        n_hidden=args.n_hidden,
        n_out_classes=6  # writer, singer, painter, politician, mathematician, architect
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.eta)

    for epoch in range(1, args.epochs + 1):
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

            if not batch_count % 10:
                print(f"Batch number {batch_count}, avg loss: {loss_total / batch_count:.4f}")

        print("—"*100)
        print(f"Epoch: {epoch} out of {args.epochs}")
        print(f"Mean loss: {loss_total / num_batches:.4f}")
        print(f"Total loss: {loss_total:.4f}")
        evaluate(y_gold, y_pred)
        print("—"*100)

    print()