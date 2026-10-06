"""Repeated scaling and conditioning experiments with matched dual-gap targets."""

from pathlib import Path
import argparse
import pandas as pd
from multimarginal_numerics import random_covariances, solve_rgd, solve_fixed_point

HERE = Path(__file__).resolve().parent
(HERE / "cache").mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--pilot", action="store_true")
args = parser.parse_args()
settings = (
    [(10, 3, 10), (10, 5, 1000)]
    if args.pilot
    else [(p, 3, None) for p in [3, 10, 25, 50, 100]]
    + [(10, d, None) for d in [2, 5, 8, 12, 16, 32]]
    + [(10, 5, c) for c in [1, 10, 100, 1000, 10000]]
)
rows = []
path = (
    HERE
    / "cache"
    / ("optimization_pilot.csv" if args.pilot else "optimization_comparison.csv")
)
for p, d, condition in settings:
    for trial in range(1 if args.pilot else 3):
        covariances = random_covariances(p, d, condition, 1381 + trial)
        for restart in range(1 if args.pilot else 5):
            for solver in [solve_rgd, solve_fixed_point]:
                _, info = solver(
                    covariances, seed=100 * trial + restart, tolerance=1e-7
                )
                info.update(p=p, d=d, condition=condition, trial=trial, restart=restart)
                rows.append(info)
                print(
                    p,
                    d,
                    condition,
                    trial,
                    restart,
                    info["method"],
                    info["runtime"],
                    info["relative_dual_gap"],
                    info["iterations"],
                    info["stop_reason"],
                    flush=True,
                )
        pd.DataFrame(rows).to_csv(path, index=False)
