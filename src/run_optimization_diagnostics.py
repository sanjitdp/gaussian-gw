"""Dimension, conditioning, stationarity, rank, and global-value diagnostics.

The independent global objective is computed from the Gaussian Wasserstein
barycenter identity for the pairwise quadratic multimarginal objective. This
avoids treating an SDP solver's numerical output as ground truth.
"""

from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "images"
OUTPUT.mkdir(exist_ok=True)
plt.style.use(HERE / "math.mplstyle")


def matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T


def gaussian_barycenter(covariances, tolerance=1e-12, max_iter=10000):
    barycenter = sum(covariances) / len(covariances)
    for _ in range(max_iter):
        root = matrix_sqrt(barycenter)
        updated = sum(matrix_sqrt(root @ sigma @ root) for sigma in covariances)
        updated /= len(covariances)
        residual = np.linalg.norm(updated - barycenter, "fro") / max(
            np.linalg.norm(barycenter, "fro"), 1e-15
        )
        barycenter = updated
        if residual < tolerance:
            return barycenter
    raise RuntimeError("Gaussian barycenter fixed-point iteration did not converge")


def global_trace_objective(covariances) -> float:
    """Maximum sum_{i<j} tr(C_ij), via the equivalent barycenter problem."""
    p = len(covariances)
    barycenter = gaussian_barycenter(covariances)
    root = matrix_sqrt(barycenter)
    barycenter_cost = sum(
        np.trace(barycenter)
        + np.trace(sigma)
        - 2 * np.trace(matrix_sqrt(root @ sigma @ root))
        for sigma in covariances
    )
    pairwise_cost = p * barycenter_cost
    constant = (p - 1) * sum(np.trace(sigma) for sigma in covariances)
    return float((constant - pairwise_cost) / 2)


def solve_burer_monteiro(covariances, seed, max_iter=3000, tolerance=1e-8):
    rng = np.random.default_rng(seed)
    p = len(covariances)
    d = covariances[0].shape[0]
    roots = [matrix_sqrt(sigma) for sigma in covariances]
    frames = []
    for _ in range(p):
        q, _ = np.linalg.qr(rng.standard_normal((d + 1, d)))
        frames.append(q.T)

    def project(frame, gradient):
        return gradient - ((frame @ gradient.T + gradient @ frame.T) / 2) @ frame

    def retract(frame):
        q, r = np.linalg.qr(frame.T)
        signs = np.sign(np.diag(r))
        signs[signs == 0] = 1
        q = q @ np.diag(signs)
        return q.T

    def objective(candidate_frames):
        factors = [root @ frame for root, frame in zip(roots, candidate_frames)]
        total = sum(factors)
        return -0.5 * (
            np.linalg.norm(total, "fro") ** 2
            - sum(np.linalg.norm(factor, "fro") ** 2 for factor in factors)
        )

    step = 0.01
    start = time.perf_counter()
    for _ in range(max_iter):
        factors = [root @ frame for root, frame in zip(roots, frames)]
        total = sum(factors)
        gradients = [
            project(frame, root.T @ (-(total - factor)))
            for root, frame, factor in zip(roots, frames, factors)
        ]
        gradient_norm = np.sqrt(sum(np.linalg.norm(g, "fro") ** 2 for g in gradients))
        if gradient_norm < tolerance:
            break
        current = objective(frames)
        accepted = False
        trial_step = step
        for _ in range(30):
            candidate = [
                retract(frame - trial_step * gradient)
                for frame, gradient in zip(frames, gradients)
            ]
            if objective(candidate) < current:
                frames = candidate
                step = min(1.5 * trial_step, 1.0)
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            break
    runtime = time.perf_counter() - start

    factors = [root @ frame for root, frame in zip(roots, frames)]
    total = sum(factors)
    gradients = [
        project(frame, root.T @ (-(total - factor)))
        for root, frame, factor in zip(roots, frames, factors)
    ]
    gradient_norm = np.sqrt(sum(np.linalg.norm(g, "fro") ** 2 for g in gradients))
    stacked = np.vstack(factors)
    singular_values = np.linalg.svd(stacked, compute_uv=False)
    feasibility = max(
        np.linalg.norm(factor @ factor.T - sigma, "fro") / np.linalg.norm(sigma, "fro")
        for factor, sigma in zip(factors, covariances)
    )
    return {
        "objective": -objective(frames),
        "runtime": runtime,
        "gradient_norm": gradient_norm,
        "rank_ratio": singular_values[-1] / singular_values[0],
        "feasibility": feasibility,
    }


def random_covariances(p, d, seed):
    rng = np.random.default_rng(seed)
    return [
        (lambda a: a @ a.T + 0.1 * np.eye(d))(rng.standard_normal((d, d)))
        for _ in range(p)
    ]


def conditioned_covariances(p, d, condition_number, seed):
    rng = np.random.default_rng(seed)
    eigenvalues = np.geomspace(1.0, 1.0 / condition_number, d)
    eigenvalues *= d / eigenvalues.sum()
    covariances = []
    for _ in range(p):
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        covariances.append((q * eigenvalues) @ q.T)
    return covariances


def run_family(kind, values, covariance_factory, p, restarts=5, trials=3):
    rows = []
    for value in values:
        for trial in range(trials):
            covariances = covariance_factory(p, value, 1000 + trial)
            optimum = global_trace_objective(covariances)
            for restart in range(restarts):
                result = solve_burer_monteiro(
                    covariances, seed=100000 * trial + 1000 * int(value) + restart
                )
                rows.append(
                    {
                        "family": kind,
                        "value": value,
                        "trial": trial,
                        "restart": restart,
                        "relative_objective_gap": abs(optimum - result["objective"])
                        / max(abs(optimum), 1.0),
                        "normalized_kkt_residual": result["gradient_norm"]
                        / max(abs(optimum), 1.0),
                        **result,
                    }
                )
        print(f"completed {kind}={value}")
    return rows


dimension_values = [2, 3, 5, 8, 12, 16]
condition_values = [1, 10, 100, 1000, 10000]
rows = run_family(
    "dimension",
    dimension_values,
    lambda p, d, seed: random_covariances(p, d, seed),
    p=10,
)
rows += run_family(
    "condition",
    condition_values,
    lambda p, condition, seed: conditioned_covariances(p, 5, condition, seed),
    p=10,
)
frame = pd.DataFrame(rows)
frame.to_csv(HERE / "cache" / "optimization_diagnostics.csv", index=False)


def summarize(family):
    subset = frame.loc[frame["family"] == family]
    return subset.groupby("value").agg(
        runtime=("runtime", "median"),
        objective_gap=("relative_objective_gap", "max"),
        kkt=("normalized_kkt_residual", "max"),
        rank_ratio=("rank_ratio", "max"),
        feasibility=("feasibility", "max"),
    )


dimension = summarize("dimension")
condition = summarize("condition")
fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
axes[0, 0].plot(dimension.index, dimension["runtime"], "o-")
axes[0, 0].set_xlabel("Dimension ($d$)")
axes[0, 0].set_ylabel("Median runtime (seconds)")
axes[0, 0].set_title("Dimension scaling ($p=10$)")

for column, label in [
    ("objective_gap", "Global objective gap"),
    ("kkt", "Riemannian KKT residual"),
    ("rank_ratio", "Rank ratio $\\sigma_{d+1}/\\sigma_1$"),
    ("feasibility", "Feasibility residual"),
]:
    axes[0, 1].semilogy(dimension.index, dimension[column], "o-", label=label)
axes[0, 1].set_xlabel("Dimension ($d$)")
axes[0, 1].set_ylabel("Maximum residual over 15 runs")
axes[0, 1].set_title("Accuracy and stationarity")
axes[0, 1].legend(fontsize=8)

axes[1, 0].semilogx(condition.index, condition["runtime"], "o-")
axes[1, 0].set_xlabel("Covariance condition number")
axes[1, 0].set_ylabel("Median runtime (seconds)")
axes[1, 0].set_title("Condition-number sensitivity ($p=10,d=5$)")

for column, label in [
    ("objective_gap", "Global objective gap"),
    ("kkt", "Riemannian KKT residual"),
    ("rank_ratio", "Rank ratio $\\sigma_{d+1}/\\sigma_1$"),
    ("feasibility", "Feasibility residual"),
]:
    axes[1, 1].loglog(condition.index, condition[column], "o-", label=label)
axes[1, 1].set_xlabel("Covariance condition number")
axes[1, 1].set_ylabel("Maximum residual over 15 runs")
axes[1, 1].set_title("Accuracy and stationarity")
axes[1, 1].legend(fontsize=8)

for axis in axes.ravel():
    axis.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT / "optimization_diagnostics.pdf")
plt.close(fig)

print("worst relative objective gap", frame["relative_objective_gap"].max())
print("worst normalized KKT residual", frame["normalized_kkt_residual"].max())
print("worst rank ratio", frame["rank_ratio"].max())
print("worst feasibility residual", frame["feasibility"].max())
