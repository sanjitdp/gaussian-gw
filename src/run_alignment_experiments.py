"""Compute Gaussian alignments and bounds from cached moments."""

from pathlib import Path
import pickle, json, time
import numpy as np
import pandas as pd
from igw_numerics import gaussian_igw, transport
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
(HERE / "cache").mkdir(parents=True, exist_ok=True)
OUTPUT = HERE / "images"
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.style.use(HERE / "math.mplstyle")
source = HERE / "cache" / "bert_igw_results.pkl"
results = pickle.load(source.open("rb"))
rows = []
for name, dataset in results["datasets"].items():
    for model, record in dataset["model_data"].items():
        start = time.perf_counter()
        r = gaussian_igw(
            record["mean"],
            record["covariance"],
            dataset["reference_mean"],
            dataset["reference_covariance"],
        )
        elapsed = time.perf_counter() - start
        for key in ["lower_bound", "upper_bound", "estimate"]:
            record[key] = r[key]
        rows.append(
            dict(
                dataset=name,
                model=model,
                **{
                    k: r[k]
                    for k in [
                        "lower_bound",
                        "upper_bound",
                        "estimate",
                        "orthogonality_residual",
                    ]
                },
                runtime=elapsed,
            )
        )
    records = list(dataset["model_data"].values())
    dataset["summary_stats"] = {
        k: [record[v] for record in records]
        for k, v in [
            ("model_sizes", "param_count"),
            ("lower_bounds", "lower_bound"),
            ("upper_bounds", "upper_bound"),
            ("estimates", "estimate"),
        ]
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    records = sorted(records, key=lambda r: r["param_count"])
    for key, label, marker in [
        ("upper_bound", "Analytic upper bound", "^"),
        ("estimate", "Feasible RGD upper bound", "o"),
        ("lower_bound", "Analytic lower bound", "s"),
    ]:
        ax.plot(
            [r["param_count"] for r in records],
            [r[key] for r in records],
            marker=marker,
            label=label,
            linewidth=1,
            markersize=4,
        )
    ax.set(
        xlabel="Model size (millions of parameters)", ylabel="IGW distance", title=name
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"igw_distance_{name}.pdf")
    plt.close(fig)
    print(name, "done", flush=True)
results["analysis_metadata"][
    "formula_version"
] = "ordered_spectral_diagonal_sign_initialization"
pickle.dump(results, source.open("wb"))
pd.DataFrame(rows).to_csv(HERE / "cache" / "alignment_diagnostics.csv", index=False)
# Check map covariances for square and rectangular alignments.
rng = np.random.default_rng(738)
audits = []
for d1, d2 in [(2, 2), (5, 3), (7, 7)]:
    for _ in range(3):
        b1 = rng.normal(size=(d1, d1))
        b2 = rng.normal(size=(d2, d2))
        _, r = transport(
            rng.normal(size=d1),
            rng.normal(size=d2),
            b1 @ b1.T + 0.1 * np.eye(d1),
            b2 @ b2.T + 0.1 * np.eye(d2),
        )
        audits.append(
            {k: r[k] for k in ["target_covariance_error", "cross_covariance_error"]}
        )
(HERE / "cache" / "map_audit.json").write_text(json.dumps(audits, indent=2))
print(
    "Map audit max errors",
    {key: max(a[key] for a in audits) for key in audits[0]},
    flush=True,
)
