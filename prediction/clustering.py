import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster(df: pd.DataFrame, num_clusters: int, col: str) -> None: 
    # expected (n_samples, n_features)
    
    tfidf = TfidfVectorizer(
        max_features=int(len({xi for x in df[col].values for xi in x}) * 0.45),
        use_idf=True,
        ngram_range=(1, 3),
    )
    fitted = tfidf.fit_transform(df[col].apply(" ".join))

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=77,
        max_iter=300,
        init="k-means++",
    )
    res = kmeans.fit(fitted)

    print(f"Homogeneity: {metrics.homogeneity_score(df.category, res.labels_):0.4f}")
    print(f"Completeness: {metrics.completeness_score(df.category, res.labels_):0.4f}")
    print(f"V-measure: {metrics.v_measure_score(df.category, res.labels_):0.4f}")
    print(f"Adjusted Rand-Index: {metrics.adjusted_rand_score(df.category, res.labels_):.4f}")
    print(f"Silhouette Coefficient: {metrics.silhouette_score(fitted, res.labels_, sample_size=1000):0.4f}")
