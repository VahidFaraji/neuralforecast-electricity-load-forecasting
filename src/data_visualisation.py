from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Utility
# =========================================================

def savefig(fig: plt.Figure, out_path: Path, dpi: int = 300) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Seasonal Overall Profile
# =========================================================

def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def plot_seasonal_hourly_profile_overall(
    df_long: pd.DataFrame,
    out_path: Path,
    ds_col: str,
    y_col: str,
    uid_col: str,
    agg_across_clients: str = "mean",
    q_low: float = 0.10,
    q_high: float = 0.90,
):
    df_long[ds_col] = pd.to_datetime(df_long[ds_col])
    df_long[y_col] = pd.to_numeric(df_long[y_col], errors="coerce").astype("float32") #coerce: Replace invalid values with NaN

    # Aggregate across clients
    if agg_across_clients == "median":
        sys = df_long.groupby(ds_col)[y_col].median()
    else:
        sys = df_long.groupby(ds_col)[y_col].mean()

    sys = sys.sort_index()
    idx = sys.index

    tmp = pd.DataFrame(
        {
            "y": sys.values,
            "hour": idx.hour,
            "month": idx.month,
        },
        index=idx,
    )

    tmp["season"] = tmp["month"].map(_month_to_season)

    # Seasonal hourly statistics
    g = tmp.groupby(["season", "hour"])["y"]
    stats = g.agg(
        mean="mean",
        ql=lambda x: x.quantile(q_low),
        qh=lambda x: x.quantile(q_high),
    ).reset_index()

    season_order = ["Winter", "Spring", "Summer", "Fall"]
    stats["season"] = pd.Categorical(stats["season"],
                                     categories=season_order,
                                     ordered=True)
    stats = stats.sort_values(["season", "hour"])

    fig, ax = plt.subplots(figsize=(9.5, 4.2))

    for season in season_order:
        sub = stats[stats["season"] == season]
        if sub.empty:
            continue

        x = sub["hour"].values + 1
        m = sub["mean"].values
        lo = sub["ql"].values
        hi = sub["qh"].values

        line, = ax.plot(x, m, marker="o", linewidth=2.2, label=season)
        c = line.get_color()

        ax.fill_between(x, lo, hi, alpha=0.12)
        ax.plot(x, lo, linestyle="--", color=c, linewidth=1.4)
        ax.plot(x, hi, linestyle="--", color=c, linewidth=1.4)

    ax.set_xlim(1, 24)
    ax.set_xticks([1,4,7,10,13,16,19,22,24])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Load (kW)")
    ax.set_title("ECL Seasonal Daily Profile (Overall Load)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=4, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))

    savefig(fig, out_path)


# =========================================================
# Client Statistics & Representative Selection
# =========================================================

def compute_client_statistics(df, uid_col, y_col):
    g = df.groupby(uid_col)[y_col]
    stats = g.agg(mean="mean", std="std", min="min", max="max").reset_index()
    stats["cv"] = stats["std"] / stats["mean"]
    stats["amplitude"] = stats["max"] - stats["min"]
    return stats


def select_representative_clients(stats):
    # High amplitude (industrial-like)
    industrial_id = stats.sort_values("amplitude",
                                      ascending=False).iloc[0]["unique_id"]

    # Irregular (highest CV excluding industrial)
    irregular_id = (
        stats[stats["unique_id"] != industrial_id]
        .sort_values("cv", ascending=False)
        .iloc[0]["unique_id"]
    )

    # Typical (median CV)
    stats_sorted = stats.sort_values("cv")
    residential_id = stats_sorted.iloc[len(stats_sorted)//2]["unique_id"]

    return industrial_id, residential_id, irregular_id


# =========================================================
# 3x2 Client Diversity Plot
# =========================================================

def plot_client_diversity(
    df,
    industrial_id,
    residential_id,
    irregular_id,
    ds_col,
    y_col,
    uid_col,
    out_path,
):
    df[ds_col] = pd.to_datetime(df[ds_col])
    df = df.sort_values(ds_col)

    ids = [
        (industrial_id, "High-amplitude (industrial-like)"),
        (residential_id, "Typical (residential-like)"),
        (irregular_id, "Irregular / high-variability"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for row, (uid, title) in enumerate(ids):

        sub = df[df[uid_col] == uid].sort_values(ds_col)
        end = sub[ds_col].max()

        long_window = sub[sub[ds_col] >= end - pd.Timedelta(days=30)]
        zoom_window = sub[sub[ds_col] >= end - pd.Timedelta(days=7)]

        axes[row, 0].plot(long_window[ds_col], long_window[y_col])
        axes[row, 0].set_title(f"{title} — 30 days")
        axes[row, 0].set_ylabel("Load (kW)")

        axes[row, 0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axes[row, 0].xaxis.set_major_locator(mdates.AutoDateLocator())

        # --- Right (7 days)
        axes[row, 1].plot(zoom_window[ds_col], zoom_window[y_col])
        axes[row, 1].set_title("Zoom — 7 days")

        axes[row, 1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axes[row, 1].xaxis.set_major_locator(mdates.AutoDateLocator())

    for ax in axes.flatten():
        ax.grid(alpha=0.3)

    savefig(fig, out_path)

# =========================================================
# Distribution of Mean Load per Client
# =========================================================
def plot_mean_load_distribution(
    df: pd.DataFrame,
    uid_col: str,
    y_col: str,
    out_path: Path,
):

    # --- Compute mean per client ---
    mean_per_client = (
        df.groupby(uid_col)[y_col]
        .mean()
        .astype("float32")
    )

    # --- Get Top 10 ---
    top10 = mean_per_client.sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    counts, bins, patches = ax.hist(
        mean_per_client,
        bins=30,
        edgecolor="black",
        alpha=0.7
    )

    # --- Add count labels on bars ---
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                count,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    # --- Create tidy info box text ---
    info_lines = ["Top 10 Mean Loads (kW):"]
    header = f"{'unique_id':<10} {'Mean Load':>12}"
    separator = "-" * 24

    info_lines = [header, separator]

    for uid, value in top10.items():
        info_lines.append(f"{str(uid):<10} {value:>12,.0f}")

    info_text = "\n".join(info_lines)

    # --- Add info box (left side) ---
    ax.text(
        0.7, 0.95,                
        info_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        family="monospace",        # Ensures clean column alignment
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
            edgecolor="gray"
        )
    )


    ax.set_title("Distribution of Mean Load per Client")
    ax.set_xlabel("Mean Load (kW)")
    ax.set_ylabel("Number of Clients")
    ax.grid(alpha=0.3)

    savefig(fig, out_path)

# =========================================================
# Compute ACF(24) and ACF(168) Distribution Across Clients
# =========================================================
def compute_acf_at_lags(x: np.ndarray, lags=(24, 168)):
    """
    Compute ACF at specific lags using FFT method.
    """
    x = x.astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    if len(x) < max(lags) + 1:
        return [np.nan] * len(lags)

    x = x - x.mean()
    n = len(x)

    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(fx * np.conj(fx), n=nfft)[:n]
    acf /= acf[0] if acf[0] != 0 else 1.0

    return [acf[lag] if lag < len(acf) else np.nan for lag in lags]

# =========================================================
# Plot ACF(24) and ACF(168) Distribution Across Clients
# =========================================================
def plot_acf_distribution(
    df: pd.DataFrame,
    uid_col: str,
    y_col: str,
    out_path: Path,
):
    """
    Plot distribution of ACF(24) and ACF(168) across clients.
    """

    acf_24_values = []
    acf_168_values = []

    # Group by client
    for uid, group in df.groupby(uid_col):
        series = group[y_col].values.astype("float32")
        acf_24, acf_168 = compute_acf_at_lags(series, lags=(24, 168))

        if not np.isnan(acf_24):
            acf_24_values.append(acf_24)
        if not np.isnan(acf_168):
            acf_168_values.append(acf_168)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ACF(24)
    axes[0].hist(acf_24_values, bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_title("Distribution of ACF(24)")
    axes[0].set_xlabel("ACF at 24 hours")
    axes[0].set_ylabel("Number of Clients")
    axes[0].grid(alpha=0.3)

    # ACF(168)
    axes[1].hist(acf_168_values, bins=30, edgecolor="black", alpha=0.7)
    axes[1].set_title("Distribution of ACF(168)")
    axes[1].set_xlabel("ACF at 168 hours")
    axes[1].grid(alpha=0.3)

    savefig(fig, out_path)



# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=r"D:\Time series\NHiTS\data\ECL\ecl_long.csv",
        )

    parser.add_argument(
        "--out",
        default=r"D:\Time series\NHiTS\thesis_figures\ECL",
    )

    # Fixed defaults for ECL schema
    parser.add_argument("--ds_col", default="ds")
    parser.add_argument("--y_col", default="y")
    parser.add_argument("--uid_col", default="unique_id")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df[args.ds_col] = pd.to_datetime(df[args.ds_col])

    # --- 1) Seasonal overall profile ---
    plot_seasonal_hourly_profile_overall(
        df_long=df,
        out_path=out_dir / "ecl_seasonal_hourly_profile_overall.png",
        ds_col=args.ds_col,
        y_col=args.y_col,
        uid_col=args.uid_col,
    )

    # --- 2) Client diversity ---
    stats = compute_client_statistics(df, args.uid_col, args.y_col)

    industrial_id, residential_id, irregular_id = \
        select_representative_clients(stats)

    print("\nSelected representative clients:")
    print("Industrial-like:", industrial_id)
    print("Residential-like:", residential_id)
    print("Irregular:", irregular_id)

    plot_client_diversity(
        df,
        industrial_id,
        residential_id,
        irregular_id,
        args.ds_col,
        args.y_col,
        args.uid_col,
        out_dir / "ecl_client_diversity.png",
    )

    # ---3) Mean load distribution ---
    plot_mean_load_distribution(
        df,
        args.uid_col,
        args.y_col,
        out_dir / "ecl_mean_load_distribution.png",
    )

    # --- 4) ACF distribution ---
    plot_acf_distribution(
        df,
        args.uid_col,
        args.y_col,
        out_dir / "ecl_acf_distribution.png",
    )

    top40 = (
        df.groupby("unique_id")["y"]
        .mean()
        .sort_values(ascending=False)
        .head(40)
    )

    print(top40)

    print("\n[OK] Generated:")
    print(" - ecl_seasonal_hourly_profile_overall.png")
    print(" - ecl_client_diversity.png")
    print(" - ecl_mean_load_distribution.png")
    print(" - ecl_acf_distribution.png")


if __name__ == "__main__":
    main()
