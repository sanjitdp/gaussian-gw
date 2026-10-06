"""Product row-Stiefel RGD, Gaussian barycenter iteration, and SDP diagnostics.

The equivalent minimized objective is -.5*||sum_i rho_i U_i||_F^2.
All factors satisfy U_i U_i.T = Sigma_i to floating-point precision.
The dual bound is built independently of numerical rank or stationarity.
"""

import time
import numpy as np
from scipy.linalg import block_diag


def sym(x):
    return (x + np.swapaxes(x, -1, -2)) / 2


def powers(x):
    w, q = np.linalg.eigh(sym(x))
    if np.min(w) <= 0:
        raise ValueError("Positive definite covariance required")
    return (q * np.sqrt(w)[..., None, :]) @ np.swapaxes(q, -1, -2), (
        q / np.sqrt(w)[..., None, :]
    ) @ np.swapaxes(q, -1, -2)


def project(v, g):
    return g - sym(g @ np.swapaxes(v, -1, -2)) @ v


def retract(v):
    q, r = np.linalg.qr(np.swapaxes(v, -1, -2), mode="reduced")
    signs = np.where(np.diagonal(r, axis1=-2, axis2=-1) < 0, -1.0, 1.0)
    return np.swapaxes(q * signs[..., None, :], -1, -2)


def diagnostics(covariances, factors, rho, roots=None, full=False):
    p, d, k = factors.shape
    if roots is None:
        roots = powers(covariances)[0]
    total = np.einsum("i,ijk->jk", rho, factors)
    primal = -0.5 * np.linalg.norm(total) ** 2
    scale = max(abs(primal), 1.0)
    # Symmetric first-order multipliers for the original factor constraints.
    right = rho[:, None, None] * (total @ np.swapaxes(factors, -1, -2))
    multipliers = sym(
        np.swapaxes(np.linalg.solve(covariances, np.swapaxes(right, -1, -2)), -1, -2)
    )
    raw_eigenvalues = np.linalg.eigvalsh(multipliers)
    margin = 1e-12 * max(1.0, np.max(np.abs(raw_eigenvalues)))
    shifts = np.maximum(0.0, margin - raw_eigenvalues[:, 0])
    positive = multipliers + shifts[:, None, None] * np.eye(d)
    schur = np.einsum("i,ijk->jk", rho * rho, np.linalg.inv(positive))
    inflation = max(1.0, np.linalg.eigvalsh(sym(schur))[-1]) * (1 + 1e-12)
    dual_multipliers = inflation * positive
    dual = -0.5 * np.einsum("ijk,ikj->", dual_multipliers, covariances)
    gap = max(0.0, primal - dual) / scale
    residual = multipliers @ factors - rho[:, None, None] * total
    inverse_roots = powers(covariances)[1]
    frames = inverse_roots @ factors
    gradients = project(frames, -rho[:, None, None] * (roots @ total))
    feasibility = np.max(
        np.linalg.norm(
            factors @ np.swapaxes(factors, -1, -2) - covariances, axis=(1, 2)
        )
        / np.linalg.norm(covariances, axis=(1, 2))
    )
    result = dict(
        primal=primal,
        dual=dual,
        relative_dual_gap=gap,
        gradient_norm=np.linalg.norm(gradients),
        kkt_residual=np.linalg.norm(residual)
        / max(1.0, np.linalg.norm(rho[:, None, None] * total)),
        feasibility=feasibility,
    )
    if full:
        singular = np.linalg.svd(factors.reshape(p * d, k), compute_uv=False)
        result.update(
            numerical_rank=int(np.sum(singular > 1e-6 * singular[0])),
            rank_threshold=1e-6,
            rank_ratio=singular[-1] / singular[0],
        )
        slack = block_diag(*dual_multipliers) - np.kron(np.outer(rho, rho), np.eye(d))
        raw_slack = block_diag(*multipliers) - np.kron(np.outer(rho, rho), np.eye(d))
        result.update(
            dual_min_eigenvalue=float(np.linalg.eigvalsh(sym(slack))[0]),
            unshifted_dual_min_eigenvalue=float(np.linalg.eigvalsh(sym(raw_slack))[0]),
            complementarity=np.linalg.norm(slack @ factors.reshape(p * d, k))
            / max(1.0, np.linalg.norm(factors)),
        )
    return result


def solve_rgd(covariances, seed=0, rho=None, tolerance=1e-7, max_iter=10000):
    start = time.perf_counter()
    p, d, _ = covariances.shape
    if rho is None:
        rho = np.full(p, 1 / p)
    roots, _ = powers(covariances)
    frames = retract(np.random.default_rng(seed).normal(size=(p, d, d + 1)))
    step = p / max(np.linalg.eigvalsh(covariances).max(), 1.0)

    def objective(v):
        return -0.5 * np.linalg.norm(np.einsum("i,ijk->jk", rho, roots @ v)) ** 2

    value = objective(frames)
    reason = "iteration_limit"
    for iteration in range(max_iter + 1):
        factors = roots @ frames
        total = np.einsum("i,ijk->jk", rho, factors)
        gradients = project(frames, -rho[:, None, None] * (roots @ total))
        norm_sq = np.sum(gradients * gradients)
        if iteration % 25 == 0:
            info = diagnostics(covariances, factors, rho, roots)
            if info["relative_dual_gap"] <= tolerance and info["feasibility"] <= 1e-10:
                reason = "dual_gap"
                break
        if iteration == max_iter:
            break
        for backtrack in range(40):
            candidate = retract(frames - step * gradients)
            candidate_value = objective(candidate)
            if candidate_value <= value - 1e-4 * step * norm_sq:
                frames, value = candidate, candidate_value
                step = min(1.5 * step, 1e6)
                break
            step *= 0.5
        else:
            reason = "line_search"
            break
    factors = roots @ frames
    info = diagnostics(covariances, factors, rho, roots)
    runtime = time.perf_counter() - start
    info.update(diagnostics(covariances, factors, rho, roots, full=True))
    return factors, dict(
        **info, runtime=runtime, iterations=iteration, stop_reason=reason, method="RGD"
    )


def solve_fixed_point(covariances, seed=0, rho=None, tolerance=1e-7, max_iter=10000):
    start = time.perf_counter()
    p, d, _ = covariances.shape
    if rho is None:
        rho = np.full(p, 1 / p)
    # Random positive-definite starts test reliability for both methods.
    rng = np.random.default_rng(seed)
    base = np.einsum("i,ijk->jk", rho, covariances)
    a = rng.normal(size=(d, d))
    b = a @ a.T / d + 0.1 * np.eye(d)
    root = powers(base)[0]
    b = root @ b @ root
    reason = "iteration_limit"
    for iteration in range(max_iter + 1):
        root, inverse = powers(b)
        mids = powers(root @ covariances @ root)[0]
        # U_i = T_i B^.5 = B^(-.5) (B^.5 Sigma_i B^.5)^.5.
        factors = inverse @ mids
        info = diagnostics(covariances, factors, rho)
        if info["relative_dual_gap"] <= tolerance and info["feasibility"] <= 1e-10:
            reason = "dual_gap"
            break
        if iteration == max_iter:
            break
        average = np.einsum("i,ijk->jk", rho, mids)
        b = sym(inverse @ average @ average @ inverse)
    runtime = time.perf_counter() - start
    info.update(diagnostics(covariances, factors, rho, full=True))
    return factors, dict(
        **info,
        runtime=runtime,
        iterations=iteration,
        stop_reason=reason,
        method="Fixed point"
    )


def random_covariances(p, d, condition, seed):
    rng = np.random.default_rng(seed)
    if condition is None:
        a = rng.normal(size=(p, d, d))
        return a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(d)
    q, _ = np.linalg.qr(rng.normal(size=(p, d, d)))
    spectrum = np.geomspace(1, 1 / condition, d)
    spectrum *= d / spectrum.sum()
    return (q * spectrum) @ np.swapaxes(q, -1, -2)
