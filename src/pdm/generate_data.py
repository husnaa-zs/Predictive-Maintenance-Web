"""Create realistic-looking synthetic sensor data for several machines.

Each machine runs normally, then slowly degrades before a failure:
vibration, temperature and current creep up, rpm becomes unstable.
That gives us a signal to learn from, with clear labels.

Run:  python -m pdm.generate_data
"""
import argparse

import numpy as np
import pandas as pd

from .config import HORIZON_HOURS, RAW_CSV, RANDOM_SEED


def simulate_machine(machine_id: int, hours: int, rng: np.random.Generator,
                     start: pd.Timestamp) -> pd.DataFrame:
    t = np.arange(hours)
    daily = np.sin(2 * np.pi * t / 24)  # daily load cycle

    temperature = 70 + 3 * daily + rng.normal(0, 0.8, hours)
    vibration = 0.30 + 0.02 * daily + rng.normal(0, 0.03, hours)
    pressure = 100 + 2 * daily + rng.normal(0, 1.0, hours)
    rpm = 1500 + 20 * daily + rng.normal(0, 8, hours)
    current = 12 + 0.8 * daily + rng.normal(0, 0.3, hours)

    failure = np.zeros(hours, dtype=int)

    # Randomly place 1-3 failures, at least 5 days apart, none in the first 3 days.
    n_failures = int(rng.integers(1, 4))
    candidates = list(range(72, hours - 1, 1))
    fail_times: list[int] = []
    for _ in range(n_failures):
        if not candidates:
            break
        ft = int(rng.choice(candidates))
        fail_times.append(ft)
        candidates = [c for c in candidates if abs(c - ft) > 120]

    for ft in fail_times:
        degr_len = int(rng.integers(36, 72))  # hours of visible degradation
        lo = max(0, ft - degr_len)
        ramp = np.linspace(0, 1, ft - lo + 1) ** 2  # accelerating wear
        idx = slice(lo, ft + 1)
        vibration[idx] += 0.55 * ramp + rng.normal(0, 0.04, ft - lo + 1) * ramp
        temperature[idx] += 9 * ramp
        current[idx] += 2.5 * ramp
        rpm[idx] += rng.normal(0, 25, ft - lo + 1) * ramp
        pressure[idx] -= 4 * ramp
        failure[ft] = 1

    # Machine is stopped / being repaired for 6 hours after a failure -> drop those rows.
    keep = np.ones(hours, dtype=bool)
    for ft in fail_times:
        keep[ft + 1: ft + 7] = False

    df = pd.DataFrame({
        "timestamp": start + pd.to_timedelta(t, unit="h"),
        "machine_id": f"M{machine_id:03d}",
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "rpm": rpm,
        "current": current,
        "failure": failure,
    })
    return df[keep].reset_index(drop=True)


def add_label(df: pd.DataFrame, horizon: int = HORIZON_HOURS) -> pd.DataFrame:
    """Add `fails_within_horizon`: 1 if a failure happens in the next `horizon` hours."""
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    labels = np.zeros(len(df), dtype=int)
    for _, g in df.groupby("machine_id"):
        fail_ts = g.loc[g["failure"] == 1, "timestamp"].to_numpy()
        ts = g["timestamp"].to_numpy()
        lab = np.zeros(len(g), dtype=int)
        for f in fail_ts:
            lab |= ((ts <= f) & (ts > f - np.timedelta64(horizon, "h"))).astype(int)
        labels[g.index.to_numpy()] = lab
    df["fails_within_horizon"] = labels
    return df


def generate(n_machines: int = 20, days: int = 120, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01 00:00:00")
    frames = [simulate_machine(i + 1, days * 24, rng, start) for i in range(n_machines)]
    return add_label(pd.concat(frames, ignore_index=True))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic sensor data")
    p.add_argument("--machines", type=int, default=20)
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = p.parse_args()

    df = generate(args.machines, args.days, args.seed)
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    print(f"Wrote {len(df):,} rows for {df['machine_id'].nunique()} machines to {RAW_CSV}")
    print(f"Failures: {int(df['failure'].sum())} | positive-label rate: "
          f"{df['fails_within_horizon'].mean():.1%}")


if __name__ == "__main__":
    main()
