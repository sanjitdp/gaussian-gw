"""Run the notebook representation pipeline for amazon_polarity and ag_news.
"""

import json
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

os.makedirs("images", exist_ok=True)
os.makedirs("cache", exist_ok=True)


def cell_source(nb_path, marker):
    """Return source of the first code cell containing `marker`."""
    with open(nb_path) as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and marker in src:
            return src
    raise ValueError(f"cell with marker {marker!r} not found")


ns = {}

# Load definitions without running the notebook driver.
src = cell_source("distillations.ipynb", "def compute_igw_bounds")
defs = src[: src.index("\nresults = []")]
exec(defs, ns)


results = []
for dataset_name in ["amazon_polarity", "ag_news"]:
    dataset_config = ns["DATASETS"][dataset_name]
    result = ns["analyze_dataset"](dataset_name, dataset_config)
    if result is not None:
        results.append(result)

if results:
    ns["save_results"](results)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("math.mplstyle")

plot_ns = {}
plot_src = cell_source("distillations.ipynb", "def plot_single_dataset_igw")
plot_defs = plot_src[: plot_src.index("\nimport pickle")]
exec(plot_defs, plot_ns)

for result in results:
    fig_result = {
        "lower_bounds": result["lower_bounds"],
        "upper_bounds": result["upper_bounds"],
        "estimates": result["estimates"],
        "dataset_name": result["dataset_name"],
    }
    plot_ns["plot_single_dataset_igw"](fig_result)
    print(f"wrote images/igw_distance_{result['dataset_name']}.pdf")

print("DONE")
