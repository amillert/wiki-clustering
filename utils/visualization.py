import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics


def visualize_confussion_matrix(golds, preds) -> None:
    cm = metrics.confusion_matrix(golds, preds)
    cnt_unique = len(set(golds))
    plt.figure(figsize=(cnt_unique, cnt_unique))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap="Blues_r")
    plt.ylabel("Actual group")
    plt.xlabel("Predicted group")
    all_sample_title = "Confusion Matrix"
    plt.title(all_sample_title, size=15)

    plt.show()

def cluster_visualize(data: dict) -> None:
    """Visualization of evaluation methods among different number of clusters.

    Args:
        data (dict): Contains result of evaluation scores.
        labels (list(str)): Ways of representing text.
    """
    labels = ["Tokens", "Token Frequencies", "TF-IDF"]

    x_pos =  np.arange(len(labels)) 
    bat_width = 0.35

    cluster2 = data["2cluster"]
    cluster6 = data["6cluster"]

    for metric in cluster2:
        fig, ax = plt.subplots()
        rects1 = ax.bar(x_pos - bat_width/2, cluster2[metric], bat_width, label="2-clusters")
        rects2 = ax.bar(x_pos + bat_width/2, cluster6[metric], bat_width, label="6-clusters")

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Scores")
        ax.set_ylim(top=1)
        ax.set_title(f"{metric} scores by ways of representing text and number of clusters")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

    plt.show()

def classify_visualize(accs: dict, labels: list) -> None:
    colors = iter(["green", "red"])

    for classification_type, res in accs.items():
        x_pos = np.arange(len(res)) 
        plt.bar(x_pos, list(map(lambda l: l[1], res)), color=next(colors))
        plt.ylabel("Accuracy scores")
        plt.xlabel(classification_type)
        plt.title(f"Accuracy by {classification_type.lower()}")
        plt.xticks(x_pos, list(map(lambda l: l[0], res)))
        plt.show()
