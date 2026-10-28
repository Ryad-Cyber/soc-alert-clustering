"""Group simulated SOC alerts into clusters of similar events with k-means.

This script is a learning exercise in applying pandas + scikit-learn to
security-style tabular data. It clusters alerts that *look* similar on a
handful of numeric/categorical fields (port, protocol, byte volume, event
count, severity). See the README's "Limitations" section before reading
too much into the output: clustering finds statistical similarity, not
confirmed incidents.

Usage:
    python src/cluster_alerts.py
    python src/cluster_alerts.py --input data/sample_alerts.csv --k 5
    python src/cluster_alerts.py --k-max 10   # let silhouette score pick k
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; we only save the figure to disk
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEVERITY_ORDER = {"Low": 0, "Medium": 1, "High": 2}


REQUIRED_COLUMNS = {
    "alert_id", "timestamp", "src_ip", "dst_ip", "dst_port",
    "protocol", "alert_type", "severity", "bytes_transferred", "event_count",
}


def load_alerts(path: Path) -> pd.DataFrame:
    """Load the CSV and do light, explicit cleaning with pandas."""
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing expected column(s): {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["dst_port", "bytes_transferred", "event_count", "severity"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} row(s) with missing required values.")

    df["severity_rank"] = df["severity"].map(SEVERITY_ORDER)
    unknown_severity = df["severity_rank"].isna()
    if unknown_severity.any():
        print(f"Dropped {unknown_severity.sum()} row(s) with an unrecognised severity label.")
        df = df[~unknown_severity]

    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the cleaned alert table into a numeric feature matrix.

    Numeric fields are used as-is; `protocol` and `alert_type` are
    one-hot encoded. `severity` is used through its ordinal rank
    (Low=0, Medium=1, High=2) rather than one-hot, since it has a
    natural order.
    """
    numeric = df[["dst_port", "bytes_transferred", "event_count", "severity_rank"]]
    categorical = pd.get_dummies(df[["protocol", "alert_type"]], prefix=["proto", "type"])
    return pd.concat([numeric, categorical], axis=1)


def choose_k(X_scaled, k_min: int, k_max: int, seed: int) -> tuple[int, dict[int, float]]:
    """Pick k by silhouette score over the given range (higher is better).

    This is a simple, interpretable way to let the data suggest a cluster
    count instead of hard-coding one -- it is not the only valid method,
    and the "best" k by silhouette score is not guaranteed to be the most
    analytically useful one for a human analyst.
    """
    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        if k >= len(X_scaled):
            break
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster: size and the fields that describe it."""
    def dominant(series: pd.Series) -> str:
        return series.mode().iloc[0]

    summary = (
        df.groupby("cluster")
        .agg(
            alert_count=("alert_id", "count"),
            dominant_alert_type=("alert_type", dominant),
            dominant_protocol=("protocol", dominant),
            mean_severity_rank=("severity_rank", "mean"),
            mean_bytes_transferred=("bytes_transferred", "mean"),
            mean_event_count=("event_count", "mean"),
            mean_dst_port=("dst_port", "mean"),
            distinct_src_ip=("src_ip", "nunique"),
        )
        .round(1)
        .sort_values("alert_count", ascending=False)
    )
    return summary


def plot_clusters(X_scaled, labels, out_path: Path) -> None:
    """2D PCA projection of the scaled features, coloured by cluster.

    PCA is used only to make a >2-dimensional feature space viewable in a
    2D scatter plot; the clustering itself runs on the full feature set,
    not on these two components.
    """
    coords = PCA(n_components=2, random_state=0).fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=18, alpha=0.8)
    ax.set_title("Simulated SOC alerts -- k-means clusters (PCA projection)")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
    ax.add_artist(legend)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sample_alerts.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--k", type=int, default=None, help="fixed number of clusters")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8, help="used only when --k is not set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_alerts(args.input)
    print(f"Loaded {len(df)} alerts from {args.input}")

    features = build_features(df)
    X_scaled = StandardScaler().fit_transform(features)

    if args.k is not None:
        k = args.k
    else:
        k, scores = choose_k(X_scaled, args.k_min, args.k_max, args.seed)
        print("Silhouette score by k:", {kk: round(s, 3) for kk, s in scores.items()})
        print(f"Selected k={k} (highest silhouette score)")

    kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    summary = summarize_clusters(df)
    print("\nCluster summary:")
    print(summary.to_string())

    args.outdir.mkdir(parents=True, exist_ok=True)
    clustered_path = args.outdir / "clustered_alerts.csv"
    summary_path = args.outdir / "cluster_summary.csv"
    plot_path = args.outdir / "cluster_plot.png"

    df.to_csv(clustered_path, index=False)
    summary.to_csv(summary_path)
    plot_clusters(X_scaled, df["cluster"].to_numpy(), plot_path)

    print(f"\nWrote: {clustered_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {plot_path}")
    print(
        "\nReminder: these are statistical groupings of simulated data, not a "
        "confirmed incident. See the README's Limitations section."
    )


if __name__ == "__main__":
    main()
