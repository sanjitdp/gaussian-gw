# Gaussian transport and alignment

We include code for *Optimal Transportation and Alignment Between Gaussian
Measures*, covering Gaussian IGW bounds and alignments, Gaussian multimarginal
optimal transport, and text-embedding experiments.

## Setup

We use Python 3.10 or newer. To install the dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd src
```

We use CMU Serif for plots when installed; Matplotlib falls back to an available
font otherwise.

We run the notebooks from `src/` and save script results to `src/cache/` and
figures to `src/images/`. We retain small CSV/JSON results in Git and exclude
model caches, embeddings, binary checkpoints, and generated figures.

For timing comparisons, we use one BLAS thread: set
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` before starting Python. We download
the datasets and models on the first inference run and reuse them afterward.
Hugging Face cache locations follow `HF_HOME` and `HF_DATASETS_CACHE`.

## Numerical methods

- In `igw_numerics.py`, we implement spectral bounds, Stiefel alignment with an
  Armijo line search, and affine transport maps. We return bounds as distances
  and represent alignments as `(d1, d2)` column frames with `d1 >= d2`, recording
  whether we swapped the input order.
- In `multimarginal_numerics.py`, we implement product-Stiefel RGD, a Gaussian
  barycenter fixed-point method, and primal/dual diagnostics. We use the relative
  dual gap as the stopping criterion for both solvers and perform full slack
  eigenvalue checks outside the timed solve.

We include numerical checks, including comparison with a nonuniform-weight SDP:

```sh
python validate_numerics.py
```

## Experiments

We provide the following scripts to run the experiments and generate figures:

| Experiment | Command | Output |
| --- | --- | --- |
| Representation comparison | `python run_distillations.py` | `cache/bert_igw_results.pkl`, IGW figures |
| Compute alignments from cached moments | `python run_alignment_experiments.py` | Moment cache and alignment diagnostics |
| CKA comparison plots | `python regenerate_cka_figures.py` | CKA/IGW figures |
| Planar transport examples | `python regenerate_planar_examples.py` | Interpolation figures and map checks |
| Repeated clustering | `python run_clustering_comparisons.py` | `cache/clustering_comparisons.csv` |
| Dense SDP comparison | `python run_mm_chunk.py 3 5 8` | `cache/mm_benchmark.pkl` |
| Plot dense SDP comparison | `python run_mm_chunk.py --plot` | Timing and objective figures |
| Repeated RGD/fixed-point comparison | `python run_optimization_experiments.py` | `cache/optimization_comparison.csv` |
| Plot repeated comparison | `python summarize_optimization_experiments.py` | Scaling figure and summary CSV |

We include a two-setting pilot run via
`python run_optimization_experiments.py --pilot`. In the full comparison, we use
16 settings, three covariance collections per setting, and five initializations
per solver, with a relative dual-gap target of `1e-7`. We also include a
barycenter-objective comparison in `run_optimization_diagnostics.py`, using a
separate stopping rule; its timings are not directly comparable.

We compute paired CKA scores in `distillations.ipynb` and use the moment cache
and `cache/cka_scores.csv` to generate the CKA plots.

For clustering, we use `all-MiniLM-L6-v2`. In each of ten
trials, we sample 16,000 documents per category from a pool of 20,000 and partition
them into 40 users of 400 documents. We compare mean, covariance-spectrum, and
combined features using the same users and embeddings, with 20 k-means
initializations. We download the scientific-papers arXiv archive on the first run,
which may take time.

We also include examples and plotting workflows in the notebooks.
