import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

def plot_count_distribution(data, column, title, xlabel):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x=column, palette='Set2')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.show()


def plot_crosstab_heatmap(index_col, col_col, df, normalize='index', cmap='coolwarm'):
    # Handle index columns
    if isinstance(index_col, list):
        index_vals = [df[col] for col in index_col]  
    else:
        index_vals = df[index_col]

    # Handle column columns
    if isinstance(col_col, list):
        col_vals = [df[col] for col in col_col]  
    else:
        col_vals = df[col_col]

    # Create crosstab
    table = pd.crosstab(index=index_vals, columns=col_vals, normalize=normalize)
    print(table)

    # Plot
    plt.figure(figsize=(8, 5))
    sns.heatmap(table, annot=True, fmt=".2f", cmap=cmap)
    plt.title(f"{index_col} vs {col_col} (normalized by {normalize})")
    plt.show()