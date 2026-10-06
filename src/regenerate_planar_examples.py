"""Regenerate planar displacement interpolations and audit their actual maps."""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from igw_numerics import transport

HERE = Path(__file__).resolve().parent
(HERE / "cache").mkdir(parents=True, exist_ok=True)
OUTPUT = HERE / "images"
OUTPUT.mkdir(parents=True, exist_ok=True)
nb = json.loads((HERE / "plotting.ipynb").read_text())
plt.style.use(HERE / "math.mplstyle")
# Reuse the notebook examples with a repository-local output path.
records = []
for i in [2, 3, 4, 5]:
    source = "".join(nb["cells"][i]["source"])
    source = source.replace('"images/', f'"{OUTPUT}/')
    # Use unambiguous filenames for the translated and untranslated W2 rows.
    if i in [4, 5]:
        import re

        filename = "w2_interpolation" if i == 4 else "w2_interpolation_shifted"
        source = re.sub(r"w2_interpolation(?:_shifted)?_", filename + "_", source)
    source = source.replace("plt.show()", "plt.close()")
    ns = {}
    exec(source, ns)
    a, s1, s2 = ns["A"], ns["Sigma1"], ns["Sigma2"]
    error = np.linalg.norm(a @ s1 @ a.T - s2) / np.linalg.norm(s2)
    assert error < 1e-8, error
    record = dict(
        cell=i, kind="IGW" if i < 4 else "W2", target_covariance_error=float(error)
    )
    if i < 4:
        _, result = transport(ns["m1"], ns["m2"], s1, s2)
        record.update(
            lower_bound=float(result["lower_bound"]),
            estimate=float(result["estimate"]),
            cross_covariance_error=float(result["cross_covariance_error"]),
        )
    records.append(record)
(HERE / "cache" / "planar_map_audit.json").write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
