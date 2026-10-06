"""Checkpointed multimarginal benchmark.

Usage: python run_mm_chunk.py 3 5 8       # run these p values, merge into checkpoint
       python run_mm_chunk.py --plot      # generate figures from checkpoint
"""

import json
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
import pickle
import sys

os.makedirs("images", exist_ok=True)

import matplotlib

matplotlib.use("Agg")

CKPT = "cache/mm_benchmark.pkl"
os.makedirs("cache", exist_ok=True)


def cell_source(nb_path, marker):
    with open(nb_path) as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and marker in src:
            return src
    raise ValueError(f"cell with marker {marker!r} not found")


src = cell_source("multimarginal.ipynb", "def solve_burer_monteiro_manifold")
defs = src[: src.index("\np_values = [")]
defs = "\n".join(line for line in defs.split("\n") if "matplotlib_inline" not in line)

ns = {}
exec(defs, ns)

EMPTY = {
    "p_values": [],
    "sdp_times": [],
    "bm_times": [],
    "sdp_objectives": [],
    "bm_objectives": [],
    "variable_counts": [],
}


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT, "rb") as f:
            return pickle.load(f)
    return {k: dict() for k in EMPTY}  # keyed by p for easy merge


def save_ckpt(store):
    with open(CKPT, "wb") as f:
        pickle.dump(store, f)


if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print("Usage: python run_mm_chunk.py P [P ...] | --bm-only P [P ...] | --plot")
    raise SystemExit(0)

if sys.argv[1] == "--plot":
    store = load_ckpt()
    ps = sorted(store["p_values"].keys())
    results = {k: [store[k].get(p) for p in ps] for k in EMPTY}
    ns["plot_benchmark_results"](results)
    for p in ps:
        sdp, bm = store["sdp_objectives"].get(p), store["bm_objectives"][p]
        if sdp is not None:
            rel = abs(sdp - bm) / max(abs(sdp), 1e-12)
            print(
                f"p={p:3d}  SDP={sdp:.6f}  BM={bm:.6f}  rel.diff={rel:.2e}  "
                f"t_sdp={store['sdp_times'][p]:.2f}s  t_bm={store['bm_times'][p]:.2f}s"
            )
        else:
            print(
                f"p={p:3d}  SDP=--          BM={bm:.6f}  "
                f"t_bm={store['bm_times'][p]:.2f}s"
            )
    print("PLOTS DONE")
elif sys.argv[1] == "--bm-only":
    import time

    p_list = [int(a) for a in sys.argv[2:]]
    store = load_ckpt()
    for p in p_list:
        print(f"--- BM benchmark p={p} ---", flush=True)
        sigmas = ns["generate_random_covariances"](p, 3)
        t0 = time.time()
        bm_obj, _, _ = ns["solve_burer_monteiro_manifold"](
            sigmas, max_iter=1000, verbose=False
        )
        store["bm_times"][p] = time.time() - t0
        store["bm_objectives"][p] = bm_obj
        store["p_values"][p] = p
        store["variable_counts"][p] = (3**2 * p * (p - 1) // 2, p * 3 * 4)
        store["sdp_times"].setdefault(p, None)
        store["sdp_objectives"].setdefault(p, None)
        save_ckpt(store)
        sdp = store["sdp_objectives"].get(p)
        sdp_str = f"{sdp:.4f}" if sdp is not None else "--"
        print(
            f"p={p}: BM={bm_obj:.4f} SDP={sdp_str} "
            f"(t_bm={store['bm_times'][p]:.1f}s)",
            flush=True,
        )
    print("CHUNK DONE")
else:
    p_list = [int(a) for a in sys.argv[1:]]
    store = load_ckpt()
    for p in p_list:
        print(f"--- running p={p} ---", flush=True)
        res = ns["benchmark_methods"]([p], d=3, max_p_for_sdp=100)
        for k in EMPTY:
            store[k][p] = res[k][0]
        save_ckpt(store)
        print(
            f"p={p} done: BM={res['bm_objectives'][0]:.4f} "
            f"SDP={res['sdp_objectives'][0]:.4f} "
            f"(t_bm={res['bm_times'][0]:.1f}s, t_sdp={res['sdp_times'][0]:.1f}s)",
            flush=True,
        )
    print("CHUNK DONE")
