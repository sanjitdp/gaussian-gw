"""Gaussian IGW bounds and feasible alignment (positive spectra sorted together).

Alignment uses the column-Stiefel convention: C has shape (d1,d2), d1>=d2.
All reported bounds are distances, rather than squared distances.
"""

import numpy as np


def spectral(sigma, mean):
    values, vectors = np.linalg.eigh((sigma + sigma.T) / 2)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if values.min() < -1e-9 * max(1.0, values.max()):
        raise ValueError("Covariance is not positive semidefinite")
    values = np.maximum(values, 0.0)
    return values, vectors, np.sqrt(values) * (vectors.T @ mean)


def retract(matrix):
    q, r = np.linalg.qr(matrix, mode="reduced")
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return q * signs


def optimize_spectral(l1, l2, w1, w2, max_iter=50, tol=1e-2):
    d1, d2 = len(l1), len(l2)
    if d1 < d2:
        raise ValueError("Expected d1 >= d2")
    c = np.eye(d1, d2)
    c[np.arange(d2), np.arange(d2)] = np.where(w1[:d2] * w2 < 0, -1.0, 1.0)
    weights = l1[:, None] * l2[None, :]
    linear = np.outer(w1, w2)

    def objective(x):
        return float(np.sum(weights * x * x) + 2 * np.sum(linear * x))

    value = objective(c)
    history = [value]
    step = 1 / max(2 * l1[0] * l2[0], 1.0)
    for iteration in range(max_iter):
        euclidean = 2 * weights * c + 2 * linear
        cg = c.T @ euclidean
        gradient = euclidean - c @ ((cg + cg.T) / 2)
        norm = np.linalg.norm(gradient)
        if norm < tol:
            break
        for backtrack in range(40):
            candidate = retract(c + step * gradient)
            candidate_value = objective(candidate)
            if candidate_value >= value + 1e-4 * step * norm * norm:
                c, value = candidate, candidate_value
                history.append(value)
                step *= 1.5
                break
            step *= 0.5
        else:
            break
    return c, history


def gaussian_igw(mean1, cov1, mean2, cov2, max_iter=50, tol=1e-2):
    swapped = len(mean1) < len(mean2)
    if swapped:
        mean1, mean2, cov1, cov2 = mean2, mean1, cov2, cov1
    l1, q1, w1 = spectral(cov1, mean1)
    l2, q2, w2 = spectral(cov2, mean2)
    paired = float(l1[: len(l2)] @ l2)
    constant = float(
        l1 @ l1
        + l2 @ l2
        + 2 * (w1 @ w1 + w2 @ w2)
        + (mean1 @ mean1 - mean2 @ mean2) ** 2
    )
    xi = constant - 2 * paired
    lower_sq = xi - 4 * np.linalg.norm(w1) * np.linalg.norm(w2)
    upper_sq = xi - 4 * np.abs(w1[: len(w2)] * w2).sum()
    c, history = optimize_spectral(l1, l2, w1, w2, max_iter, tol)
    estimate_sq = constant - 2 * history[-1]
    result = dict(
        lower_bound=np.sqrt(max(0.0, lower_sq)),
        upper_bound=np.sqrt(max(0.0, upper_sq)),
        estimate=np.sqrt(max(0.0, estimate_sq)),
        alignment=c,
        history=history,
        swapped=swapped,
    )
    k = (q1 * np.sqrt(l1)) @ c @ (q2 * np.sqrt(l2)).T
    result["cross_covariance"] = k.T if swapped else k
    result["orthogonality_residual"] = np.linalg.norm(c.T @ c - np.eye(len(l2)))
    return result


def transport(mean1, mean2, cov1, cov2, **kwargs):
    if len(mean1) < len(mean2) or np.linalg.eigvalsh(cov1).min() <= 0:
        raise ValueError("A nondegenerate source with d1 >= d2 is required")
    result = gaussian_igw(mean1, cov1, mean2, cov2, **kwargs)
    matrix = np.linalg.solve(cov1, result["cross_covariance"]).T
    target_error = np.linalg.norm(matrix @ cov1 @ matrix.T - cov2) / max(
        1.0, np.linalg.norm(cov2)
    )
    cross_error = np.linalg.norm(cov1 @ matrix.T - result["cross_covariance"]) / max(
        1.0, np.linalg.norm(result["cross_covariance"])
    )
    if target_error > 1e-8 or cross_error > 1e-8:
        raise ArithmeticError((target_error, cross_error))
    result["target_covariance_error"] = target_error
    result["cross_covariance_error"] = cross_error
    return matrix, result
