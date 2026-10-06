"""Independent identities, weighted SDP comparison, and map/certificate checks."""

from pathlib import Path
import json
import numpy as np
import cvxpy as cp
from igw_numerics import gaussian_igw, transport, spectral
from multimarginal_numerics import (
    solve_rgd,
    solve_fixed_point,
    random_covariances,
    project,
    powers,
    retract,
)

rng = np.random.default_rng(91)
# Direct empirical pairwise distortion equals its cross-moment expansion.
x = rng.normal(size=(40, 3)) + 1
y = rng.normal(size=(40, 2)) - 0.3
actual = np.mean((x @ x.T - y @ y.T) ** 2)
identity = (
    np.linalg.norm(x.T @ x / len(x)) ** 2
    + np.linalg.norm(y.T @ y / len(y)) ** 2
    - 2 * np.linalg.norm(x.T @ y / len(x)) ** 2
)
assert np.isclose(actual, identity, rtol=1e-12)
# The sign bound is unchanged by arbitrary eigenvector signs.
covs = random_covariances(2, 5, None, 73)
means = rng.normal(size=(2, 5))
l1, q1, w1 = spectral(covs[0], means[0])
l2, q2, w2 = spectral(covs[1], means[1])
signs = rng.choice([-1, 1], size=(2, 5))
assert np.isclose(
    np.abs(w1 * w2).sum(), np.abs((w1 * signs[0]) * (w2 * signs[1])).sum()
)
# Centered singular and kernel-mean cases: the exact spectral formula.
s1 = np.diag([3.0, 1.0, 0.0])
s2 = np.diag([2.0, 0.0, 0.0])
m1 = np.array([0.0, 0.0, 4.0])
m2 = np.zeros(3)
r = gaussian_igw(m1, s1, m2, s2)
assert np.isclose(r["estimate"] ** 2, 2 + 16**2)
assert np.isclose(r["estimate"], r["lower_bound"])
# One-dimensional formula is exact for arbitrary signs, including degenerate inputs.
for a, b, u, v in [(2, 3, -1, 4), (0, 2, 3, 0), (1, 1, 0.5, -0.5)]:
    r = gaussian_igw(
        np.array([u]),
        np.array([[float(a * a)]]),
        np.array([v]),
        np.array([[float(b * b)]]),
    )
    exact = (
        (a * a - b * b) ** 2 + (u * u - v * v) ** 2 + 2 * (a * abs(u) - b * abs(v)) ** 2
    )
    assert np.isclose(r["estimate"] ** 2, exact)
# Finite-dimensional target and cross covariance audit for random rectangular maps.
map_errors = []
for d1, d2 in [(2, 2), (5, 3), (7, 7)]:
    a = rng.normal(size=(d1, d1))
    b = rng.normal(size=(d2, d2))
    matrix, result = transport(
        rng.normal(size=d1),
        rng.normal(size=d2),
        a @ a.T + 0.1 * np.eye(d1),
        b @ b.T + 0.1 * np.eye(d2),
    )
    map_errors.append(result["target_covariance_error"])
# Verify the row-Stiefel gradient against an independent directional derivative.
cov = random_covariances(3, 2, None, 52)
rho = np.array([0.2, 0.3, 0.5])
root = powers(cov)[0]
v = retract(rng.normal(size=(3, 2, 3)))
h = project(v, rng.normal(size=v.shape))
S = np.einsum("i,ijk->jk", rho, root @ v)
gradient = project(v, -rho[:, None, None] * (root @ S))


def objective(z):
    return -0.5 * np.linalg.norm(np.einsum("i,ijk->jk", rho, root @ z)) ** 2


finite = (objective(retract(v + 1e-6 * h)) - objective(retract(v - 1e-6 * h))) / 2e-6
assert np.isclose(finite, np.sum(gradient * h), rtol=1e-6, atol=1e-8)
# Nonuniform-weight SDP: direct optimization is independent of either solver.
xvar = cp.Variable((6, 6), symmetric=True)
constraints = [xvar >> 0]
for i in range(3):
    constraints.append(xvar[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] == cov[i])
W = np.kron(np.outer(rho, rho), np.eye(2))
problem = cp.Problem(cp.Minimize(-0.5 * cp.trace(W @ xvar)), constraints)
problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
rows = []
for solve in [solve_rgd, solve_fixed_point]:
    _, info = solve(cov, rho=rho, tolerance=1e-9)
    assert info["dual"] <= problem.value + 1e-8
    assert info["primal"] >= problem.value - 1e-8
    assert info["relative_dual_gap"] <= 1e-9
    rows.append(dict(sdp_objective=float(problem.value), **info))
# Strict suboptimal local maximum in two dimensions.
g = lambda t: 5 - np.sin(t) ** 2 - np.cos(t)
assert g(0) == 4 and np.isclose(g(np.pi), 6)
assert (g(1e-4) - 2 * g(0) + g(-1e-4)) / 1e-8 < -0.99
out = dict(
    empirical_identity_error=float(abs(actual - identity)),
    map_error_max=float(max(map_errors)),
    gradient_derivative_error=float(abs(finite - np.sum(gradient * h))),
    weighted_sdp_checks=rows,
)
(Path(__file__).parent / "cache").mkdir(parents=True, exist_ok=True)
(Path(__file__).parent / "cache" / "numerical_validation.json").write_text(
    json.dumps(out, indent=2)
)
print(json.dumps(out, indent=2))
