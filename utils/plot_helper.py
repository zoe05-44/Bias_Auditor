import matplotlib.pyplot as plt
import numpy as np

def plot_metric_comparison(results, metric_name, title):
    model_names = list(set(r["label"].split(" (")[0] for r in results))
    unweighted = [r[metric_name] for r in results if "Unweighted" in r["label"]]
    reweighted = [r[metric_name] for r in results if "Reweighted" in r["label"]]

    x = np.arange(len(model_names))
    bar_width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - bar_width/2, unweighted, width=bar_width, label='Unweighted')
    plt.bar(x + bar_width/2, reweighted, width=bar_width, label='Reweighted')
    plt.xticks(x, model_names)
    plt.ylabel(metric_name.capitalize())
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()