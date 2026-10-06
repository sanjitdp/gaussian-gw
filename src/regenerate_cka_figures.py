"""Regenerate the CKA-versus-IGW figures from cached experiment summaries.

The CKA scores are fixed outputs of the representation experiment. The IGW
coordinates are always read from the current IGW cache, and model
sizes are associated by model name rather than by array position.
"""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "images"
OUTPUT.mkdir(exist_ok=True)
plt.style.use(HERE / "math.mplstyle")


with (HERE / "cache" / "bert_igw_results.pkl").open("rb") as stream:
    igw_results = pickle.load(stream)
cka_scores = pd.read_csv(HERE / "cache" / "cka_scores.csv")


def plot_dataset(dataset_name: str) -> None:
    model_data = igw_results["datasets"][dataset_name]["model_data"]
    frame = cka_scores.loc[cka_scores["dataset"] == dataset_name].copy()
    frame["param_count"] = frame["model_name"].map(
        lambda name: model_data[name]["param_count"]
    )
    frame["igw_distance"] = frame["model_name"].map(
        lambda name: (
            model_data[name]["estimate"]
            if np.isfinite(model_data[name]["estimate"])
            else model_data[name]["upper_bound"]
        )
    )

    x = frame["igw_distance"].to_numpy()
    y = frame["cka_score"].to_numpy()
    sizes = frame["param_count"].to_numpy()
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        x,
        y,
        c=sizes,
        s=60,
        alpha=0.7,
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5,
    )
    trend = np.poly1d(np.polyfit(x, y, 1))
    trend_x = np.linspace(x.min(), x.max(), 100)
    ax.plot(trend_x, trend(trend_x), "r--", linewidth=2, alpha=0.8, label="Trendline")
    ax.text(
        0.05,
        0.95,
        f"Pearson $r$: {pearson.statistic:.3f} ($p={pearson.pvalue:.3f}$)\n"
        f"Spearman $\\rho$: {spearman.statistic:.3f} ($p={spearman.pvalue:.3f}$)",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Model size (millions of parameters)", fontsize=12)
    ax.set_xlabel("IGW distance from bert-base (RGD upper bound)", fontsize=12)
    ax.set_ylabel("CKA similarity with bert-base", fontsize=12)
    ax.set_title(f"IGW distance vs. CKA similarity ({dataset_name})", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(top=0.7)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"cka_vs_igw_{dataset_name}.pdf")
    plt.close(fig)
    print(
        f"{dataset_name}: Pearson r={pearson.statistic:.6f}, "
        f"Spearman rho={spearman.statistic:.6f}"
    )


for dataset in ("amazon_polarity", "ag_news"):
    plot_dataset(dataset)
