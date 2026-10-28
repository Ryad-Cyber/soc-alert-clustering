"""Generate a fictional SOC alert dataset for the clustering demo.

All values below are synthetic. IPs are drawn from private ranges
(10.0.0.0/8, 192.168.0.0/16) and do not correspond to any real host.
The generator is seeded, so running it twice produces the same CSV.

Usage:
    python src/generate_data.py [--rows 300] [--out data/sample_alerts.csv] [--seed 42]
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# Six fictional alert "profiles". Each profile has its own typical port,
# protocol, severity range and byte/event-count distribution, so that the
# dataset contains a handful of genuinely different event shapes for
# k-means to find -- plus a "background noise" profile so the data is not
# artificially clean.
PROFILES = [
    {
        "alert_type": "Failed Login",
        "protocol": "TCP",
        "ports": [22, 3389],
        "severity_choices": ["Medium", "High"],
        "bytes_range": (200, 2_000),
        "event_count_range": (5, 40),  # repeated attempts in a short window
    },
    {
        "alert_type": "Port Scan",
        "protocol": "TCP",
        "ports": list(range(1, 1025)),
        "severity_choices": ["Low", "Medium"],
        "bytes_range": (40, 500),
        "event_count_range": (10, 80),  # many distinct ports touched
    },
    {
        "alert_type": "Malware Signature Match",
        "protocol": "TCP",
        "ports": [4444, 8080, 8443],
        "severity_choices": ["High"],
        "bytes_range": (5_000, 50_000),
        "event_count_range": (1, 5),
    },
    {
        "alert_type": "DNS Tunneling Suspected",
        "protocol": "UDP",
        "ports": [53],
        "severity_choices": ["Medium", "High"],
        "bytes_range": (3_000, 20_000),
        "event_count_range": (20, 100),
    },
    {
        "alert_type": "Unusual Outbound Traffic",
        "protocol": "TCP",
        "ports": [443, 8081, 9001, 51820],
        "severity_choices": ["Low", "Medium"],
        "bytes_range": (10_000, 80_000),
        "event_count_range": (1, 10),
    },
    {
        "alert_type": "Informational / Benign",
        "protocol": "TCP",
        "ports": [80, 443],
        "severity_choices": ["Low"],
        "bytes_range": (100, 3_000),
        "event_count_range": (1, 6),
    },
]

# How many rows to draw from each profile, as a share of --rows.
PROFILE_WEIGHTS = [0.18, 0.18, 0.12, 0.12, 0.15, 0.25]


def _random_private_ip(rng: np.random.Generator) -> str:
    if rng.random() < 0.5:
        return f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
    return f"192.168.{rng.integers(0, 255)}.{rng.integers(1, 255)}"


def generate(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = np.round(np.array(PROFILE_WEIGHTS) * rows).astype(int)
    counts[-1] += rows - counts.sum()  # absorb rounding remainder

    # A handful of recurring "noisy" source IPs, so some alerts plausibly
    # share an origin -- this is still synthetic data, not a real feed.
    recurring_sources = [_random_private_ip(rng) for _ in range(8)]

    start = dt.datetime(2026, 10, 1, 0, 0, 0)
    records = []
    for profile, n in zip(PROFILES, counts):
        for _ in range(n):
            src_ip = (
                rng.choice(recurring_sources)
                if rng.random() < 0.6
                else _random_private_ip(rng)
            )
            minute_offset = int(rng.integers(0, 60 * 24 * 7))  # within one week
            record = {
                "timestamp": (start + dt.timedelta(minutes=minute_offset)).isoformat(),
                "src_ip": src_ip,
                "dst_ip": _random_private_ip(rng),
                "dst_port": int(rng.choice(profile["ports"])),
                "protocol": profile["protocol"],
                "alert_type": profile["alert_type"],
                "severity": rng.choice(profile["severity_choices"]),
                "bytes_transferred": int(rng.integers(*profile["bytes_range"])),
                "event_count": int(rng.integers(*profile["event_count_range"])),
            }
            records.append(record)

    df = pd.DataFrame.from_records(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle rows
    df.insert(0, "alert_id", [f"A{idx:04d}" for idx in range(1, len(df) + 1)])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=300, help="number of alerts to generate")
    parser.add_argument("--out", type=Path, default=Path("data/sample_alerts.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} fictional alerts to {args.out}")


if __name__ == "__main__":
    main()
