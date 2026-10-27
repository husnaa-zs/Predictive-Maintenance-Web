"""A small PyTorch autoencoder for anomaly detection.

Idea: train it ONLY on normal data to compress and rebuild the sensor features.
It gets good at rebuilding normal behaviour, so when a machine starts
behaving oddly the rebuild error ("reconstruction error") goes up.
"""
from __future__ import annotations

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("PyTorch is not installed. Run: pip install torch") from e


def make_model(n_features: int, latent: int = 6):
    torch = _require_torch()
    nn = torch.nn

    class AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, latent),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent, 16), nn.ReLU(),
                nn.Linear(16, 32), nn.ReLU(),
                nn.Linear(32, n_features),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return AutoEncoder()


def train_autoencoder(x_normal: np.ndarray, epochs: int = 25, batch_size: int = 256,
                      lr: float = 1e-3, seed: int = 42):
    """Train on already-scaled normal data. Returns the trained model."""
    torch = _require_torch()
    torch.manual_seed(seed)
    model = make_model(x_normal.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    data = torch.tensor(x_normal, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  autoencoder epoch {epoch + 1:>2}/{epochs}  loss={total / len(data):.5f}")
    model.eval()
    return model


def reconstruction_error(model, x: np.ndarray) -> np.ndarray:
    torch = _require_torch()
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32)
        return ((model(t) - t) ** 2).mean(dim=1).numpy()


def save(model, scaler_mean, scaler_scale, threshold: float, path) -> None:
    torch = _require_torch()
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": len(scaler_mean),
        "scaler_mean": list(map(float, scaler_mean)),
        "scaler_scale": list(map(float, scaler_scale)),
        "threshold": float(threshold),
    }, path)


def load(path):
    torch = _require_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(ckpt["n_features"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
