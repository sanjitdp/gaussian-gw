from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
(HERE / "cache").mkdir(parents=True, exist_ok=True)
OUTPUT = HERE / "images"
OUTPUT.mkdir(parents=True, exist_ok=True)
x = pd.read_csv(HERE / "cache" / "optimization_comparison.csv")
plt.style.use(HERE / "math.mplstyle")
plt.rcParams.update(
    {
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "black",
    }
)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
families = [
    (x[(x.d == 3) & x.condition.isna()], "p", "Number of marginals $p$"),
    (x[(x.p == 10) & x.condition.isna() & (x.d != 3)], "d", "Dimension $d$"),
    (x[x.condition.notna()], "condition", "Condition number"),
]
for ax, (data, key, label) in zip(axes, families):
    for name, style, color in [
        ("RGD", "o-", "tab:blue"),
        ("Fixed point", "s--", "tab:orange"),
    ]:
        g = data[data.method == name].groupby(key).runtime.agg(["median", "min", "max"])
        ax.plot(g.index, g["median"], style, color=color, markersize=4, label=name)
        ax.fill_between(g.index, g["min"], g["max"], color=color, alpha=0.08)
    ax.set(xlabel=label, ylabel="Time (seconds)", yscale="log")
    if key == "condition":
        ax.set_xscale("log")
    ax.legend(fontsize=9, labelcolor="black")
fig.tight_layout()
fig.savefig(OUTPUT / "optimization_scaling.pdf")
plt.close(fig)
rows = []
for method, g in x.groupby("method"):
    rows.append(
        dict(
            method=method,
            runs=len(g),
            max_feasibility=g.feasibility.max(),
            max_gradient=g.gradient_norm.max(),
            max_kkt=g.kkt_residual.max(),
            max_dual_gap=g.relative_dual_gap.max(),
            max_complementarity=g.complementarity.max(),
            min_slack_eigenvalue=g.dual_min_eigenvalue.min(),
            rank_d=int((g.numerical_rank == g.d).sum()),
        )
    )
pd.DataFrame(rows).to_csv(HERE / "cache" / "optimization_summary.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
