"""Repeated mean/spectrum/combined clustering with one shared embedding model.

Source pool: first 20,000 training documents in each category.
Each of 10 seeds samples 16,000 without replacement per category and randomly
partitions them into 40 disjoint synthetic users of 400 documents each.
All three representations use exactly the same users and embeddings.
"""

from pathlib import Path
import os, json, zipfile, time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "clustering"
CACHE.mkdir(parents=True, exist_ok=True)
POOL = 20000


def texts():
    target = CACHE / "texts.json"
    if target.exists():
        return json.loads(target.read_text())
    import pyarrow as pa
    import fsspec

    root = Path(
        os.environ.get(
            "HF_DATASETS_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            / "datasets",
        )
    )

    def arrow(category):
        source = next((root / category).rglob(f"{category}-train.arrow"), None)
        if source is None:
            from datasets import load_dataset

            return load_dataset(category, split=f"train[:{POOL}]")["text"]
        with pa.memory_map(str(source), "r") as f:
            table = pa.ipc.open_stream(f).read_all()
        return table.column("text")[:POOL].to_pylist()

    result = {"news": arrow("ag_news"), "entertainment": arrow("imdb")}
    science_path = CACHE / "arxiv_abstracts.json"
    if science_path.exists():
        result["science"] = json.loads(science_path.read_text())
    else:
        url = "https://s3.amazonaws.com/datasets.huggingface.co/scientific_papers/1.1.1/arxiv-dataset.zip"
        print("Fetching arXiv abstracts from the dataset archive", flush=True)
        with fsspec.open(url, block_size=2**20, cache_type="readahead").open() as f:
            with zipfile.ZipFile(f) as z:
                name = next(n for n in z.namelist() if n.endswith("/train.txt"))
                collected = []
                with z.open(name) as stream:
                    for line in stream:
                        item = json.loads(line)
                        abstract = (
                            "\n".join(item["abstract_text"])
                            .replace("<S>", "")
                            .replace("</S>", "")
                        )
                        collected.append(abstract)
                        if len(collected) % 2000 == 0:
                            print("arXiv abstracts", len(collected), flush=True)
                        if len(collected) == POOL:
                            break
        result["science"] = collected
        science_path.write_text(json.dumps(collected))
    assert all(len(t) == POOL for t in result.values())
    target.write_text(json.dumps(result))
    return result


def embed(data):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    revision = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", revision=revision, device=device
    )
    metadata = {
        "model": "all-MiniLM-L6-v2",
        "revision": revision,
        "device": device,
        "max_seq_length": model.max_seq_length,
        "pool_per_category": POOL,
        "trials": 10,
        "users_per_category": 40,
        "documents_per_user": 400,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    (CACHE / "metadata.json").write_text(json.dumps(metadata, indent=2))
    result = []
    for category in ["science", "news", "entertainment"]:
        path = CACHE / f"{category}_embeddings.npy"
        if not path.exists():
            chunks = []
            for start in range(0, POOL, 1000):
                chunkpath = CACHE / f"{category}_{start}.npy"
                if chunkpath.exists():
                    chunk = np.load(chunkpath)
                else:
                    begin = time.perf_counter()
                    chunk = model.encode(
                        data[category][start : start + 1000],
                        batch_size=64,
                        show_progress_bar=False,
                        normalize_embeddings=False,
                    )
                    np.save(chunkpath, chunk)
                    print(
                        "embedded",
                        category,
                        start + len(chunk),
                        "seconds",
                        time.perf_counter() - begin,
                        flush=True,
                    )
                chunks.append(chunk)
            np.save(path, np.vstack(chunks))
        result.append(np.load(path))
    return result


def run(embeddings):
    rows = []
    labels = np.repeat(np.arange(3), 40)
    for trial in range(10):
        rng = np.random.default_rng(20260913 + trial)
        means = []
        spectra = []
        for pool in embeddings:
            selected = rng.choice(len(pool), size=16000, replace=False).reshape(40, 400)
            for indices in selected:
                x = pool[indices].astype(np.float64)
                mean = x.mean(axis=0)
                centered = x - mean
                eigenvalues = np.linalg.eigvalsh(centered.T @ centered / (len(x) - 1))[
                    ::-1
                ]
                means.append(mean)
                spectra.append(np.maximum(eigenvalues, 0))
        means = np.asarray(means)
        spectra = np.asarray(spectra)
        normalized = []
        for feature in [means, spectra]:
            centered = feature - feature.mean(axis=0)
            scale = np.sqrt(np.sum(centered * centered) / len(feature))
            normalized.append(centered / scale)
        combined = np.column_stack(normalized)
        for name, features in [
            ("Mean only", means),
            ("Spectrum only", spectra),
            ("Combined", combined),
        ]:
            clusters = KMeans(
                n_clusters=3, init="k-means++", n_init=20, random_state=20260913 + trial
            ).fit_predict(features)
            rows.append(
                dict(
                    trial=trial,
                    representation=name,
                    ari=adjusted_rand_score(labels, clusters),
                    nmi=normalized_mutual_info_score(labels, clusters),
                )
            )
        np.savez_compressed(
            CACHE / f"trial_{trial}_features.npz",
            means=means,
            spectra=spectra,
            labels=labels,
        )
        pd.DataFrame(rows).to_csv(
            HERE / "cache" / "clustering_comparisons.csv", index=False
        )
        print("clustering trial", trial, rows[-3:], flush=True)
    print(
        pd.DataFrame(rows)
        .groupby("representation")[["ari", "nmi"]]
        .agg(["mean", "std", "min", "max"])
        .to_string(),
        flush=True,
    )


if __name__ == "__main__":
    data = texts()
    embeddings = embed(data)
    run(embeddings)
