"""Small CPU neural pricer; checkpoint selection uses validation data only."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

BOUNDS = np.array([[0.5, 1.5], [1 / 365, 1.0], [0.05, 0.8]])


class Pricer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def _normalize(x: np.ndarray, bounds: np.ndarray = BOUNDS) -> torch.Tensor:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite X of shape (n, 3): [S/K, T_years, sigma].")
    scaled = 2 * (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1
    return torch.as_tensor(scaled, dtype=torch.float32)


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    error = np.asarray(pred, dtype=np.float64) - target
    return {
        "mae_normalized": float(np.mean(np.abs(error))),
        "rmse_normalized": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_normalized": float(np.max(np.abs(error))),
    }


def train_model(data_dir: Path, run_dir: Path, epochs: int = 200,
                seed: int = 7, threads: int = 2) -> dict:
    """Fit a 3→64→64→1 SiLU MLP to C/K and save its best validation checkpoint.

    Inputs use fixed domain bounds, with no learned preprocessing or output
    clipping. Test labels are read only after model selection is complete.
    """
    if epochs < 1 or threads < 1:
        raise ValueError("epochs and threads must be positive.")
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    train = np.load(data_dir / "train.npz")
    validation = np.load(data_dir / "validation.npz")
    x = _normalize(train["X"])
    y = torch.as_tensor(train["y"], dtype=torch.float32)
    vx = _normalize(validation["X"])
    vy = torch.as_tensor(validation["y"], dtype=torch.float32)
    net = Pricer()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=12, min_lr=1e-5,
    )
    history, best_loss, best_epoch, stale = [], float("inf"), 0, 0
    best_state = None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        net.train()
        order = torch.randperm(len(x))
        loss_sum = 0.0
        for indices in order.split(1024):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((net(x[indices]) - y[indices]) ** 2)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        net.eval()
        with torch.inference_mode():
            validation_loss = torch.mean((net(vx) - vy) ** 2).item()
        history.append({"epoch": epoch, "train_mse": loss_sum / len(x),
                        "validation_mse": validation_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        scheduler.step(validation_loss)
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if stale >= 35:
            break
    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    metadata = {
        "architecture": "3-64-64-1 MLP; SiLU hidden activations; linear output",
        "input_features": ["S/K", "T_years", "sigma"], "target": "C/K",
        "bounds": BOUNDS.tolist(), "risk_free_rate": 0.02, "strike": 100.0,
        "input_scaling": "fixed bounds mapped to [-1, 1]", "output_clipping": False,
        "training_seed": seed, "best_epoch": best_epoch,
    }
    torch.save({"state_dict": best_state, "metadata": metadata}, run_dir / "model.pt")
    with (run_dir / "training_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    net.load_state_dict(best_state)
    net.eval()
    test = np.load(data_dir / "test.npz")
    result = {"seed": seed, "nrows": {"train": len(x), "validation": len(vx),
                                       "test": len(test["X"])},
              "best_epoch": best_epoch, "epochs_run": len(history),
              "train_seconds": elapsed, "metadata": metadata}
    with torch.inference_mode():
        for name, data in [("validation", validation), ("test", test)]:
            pred = net(_normalize(data["X"])).numpy()
            result.update({f"{name}_{key}": value
                           for key, value in _metrics(pred, data["y"]).items()})
    (run_dir / "training_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_predictor(checkpoint: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Load once; return an inference-only function mapping physical inputs to C/K."""
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    net = Pricer()
    net.load_state_dict(saved["state_dict"])
    net.eval()
    bounds = np.asarray(saved["metadata"]["bounds"], dtype=np.float64)

    def predict(x: np.ndarray) -> np.ndarray:
        normalized = _normalize(x, bounds)
        if len(normalized) == 0:
            return np.empty(0, dtype=np.float64)
        with torch.inference_mode():
            return np.concatenate([net(batch).numpy() for batch in normalized.split(65536)])

    return predict
