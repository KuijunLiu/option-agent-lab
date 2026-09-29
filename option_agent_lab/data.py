"""Reproducible synthetic data. No downloaded market prices are used."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from .pricing import bs_call

NAMES = ("m", "T", "sigma")
BOUNDS = np.array([[0.5, 1.5], [1 / 365, 1.0], [0.05, 0.8]])
STRIKE, RATE = 100.0, 0.02
THRESHOLD = 0.001  # C/K units, i.e. 0.10 price units when K=100
REGION_EDGES = (
    np.array([.5, .8, .95, 1., 1.05, 1.2, 1.5]),
    np.array([1/365, 7/365, 30/365, .25, .5, 1.]),
    np.array([.05, .15, .3, .5, .8]),
)


def sample_points(n, seed, method="random", bounds=BOUNDS):
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    n = int(n)
    bounds = np.asarray(bounds, dtype=float)
    if method == "sobol":
        if n & (n - 1):
            raise ValueError("Sobol batch size must be a power of two")
        unit = qmc.Sobol(3, scramble=True, seed=seed).random_base2(n.bit_length()-1)
    elif method == "random":
        unit = np.random.default_rng(seed).random((n, 3))
    else:
        raise ValueError("Unknown sampler")
    return qmc.scale(unit, bounds[:, 0], bounds[:, 1])


def labels(x):
    x = np.asarray(x)
    return bs_call(x[:, 0] * STRIKE, STRIKE, x[:, 1], x[:, 2], RATE) / STRIKE


def generate_data(destination, seed=20260929, train=32768, validation=4096, test=8192):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    meta = {"kind": "synthetic_black_scholes", "features": list(NAMES),
            "feature_meanings": ["spot / strike", "years to expiry", "annualized volatility"],
            "target": "European call price / strike", "strike": STRIKE, "rate": RATE,
            "dividend_yield": 0, "bounds": BOUNDS.tolist(), "splits": {},
            "stress_description": "Independent diagnostic draw: m uniform[.9,1.1], T log-uniform in full domain; NOT used by search or training."}
    for i, (name, size) in enumerate([( "train", train), ("validation", validation), ("test", test), ("stress", test)]):
        split_seed = seed + i
        x = sample_points(size, split_seed)
        if name == "stress":
            rng = np.random.default_rng(split_seed)
            x[:, 0] = rng.uniform(.9, 1.1, size)
            x[:, 1] = np.exp(rng.uniform(np.log(BOUNDS[1, 0]), 0, size))
            x[:, 2] = rng.uniform(BOUNDS[2, 0], BOUNDS[2, 1], size)
        y = labels(x)
        # Publish only a complete, readable archive (interrupted writes should
        # not leave a half-written dataset at the final training path).
        temporary = destination / f".{name}.tmp.npz"
        np.savez_compressed(temporary, X=x, y=y)
        with np.load(temporary) as check:
            if not (np.array_equal(check["X"], x) and np.array_equal(check["y"], y)):
                raise RuntimeError(f"Data write verification failed for {name}")
        temporary.replace(destination / f"{name}.npz")
        # Both beginner-readable CSV and compact NumPy formats are delivered.
        with gzip.open(destination / f"{name}.csv.gz", "wt") as f:
            np.savetxt(f, np.column_stack([x, y]), delimiter=",", header="m,T,sigma,call_over_strike", comments="", fmt="%.12g")
        meta["splits"][name] = {"n": size, "seed": split_seed, "sha256_arrays": hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()}
    (destination / "metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    return meta
