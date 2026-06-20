# reporting_nf.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics_export import (
    apply_train_zscore,
    global_metrics_from_arrays,
    load_predictions_cv,
    load_scaler_stats,
    make_view_cutoffmean,
    make_view_stitched,
    per_series_metrics_fast,
    write_json,
)

# =========================================================
# ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# Configs
# =========================================================

PLOT_LABELS = {
    "ECL": {"y_label": "Load (kWh)"},
    "PJM": {"y_label": "Load"},
    "IRAN": {"y_label": "Load"},
}



# =========================================================
# Helper
# =========================================================
# Just for being appliable to different datasets

def _is_single_series(meta: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> bool:
    if "is_single_series" in meta:
        try:
            return bool(int(meta["is_single_series"]))
        except Exception:
            pass
    if df is not None and "unique_id" in df.columns:
        return int(df["unique_id"].nunique()) <= 1
    return False

# =========================================================
# META
# =========================================================

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_meta(payload: Dict[str, Any], *, run_root: Path) -> Dict[str, Any]:
    """
    Supports both schemas:
      - NEW: {"meta": {...}, "raw": {...}, "norm": {...}}
      - OLD: meta keys at top-level
    Falls back to parsing from run_root name if still missing.
    """
    if isinstance(payload.get("meta"), dict):
        meta = dict(payload["meta"])
    else:
        meta = dict(payload)

    # last-resort fallbacks from folder name: ..._H24_..._EX0_..._S10_..._W0_..._SEED43
    rn = str(meta.get("run_name") or run_root.name)
    meta.setdefault("run_name", rn)

    try:
        parts = rn.upper().split("_")
        for p in parts:
            if p.startswith("H") and p[1:].isdigit():
                meta.setdefault("H", int(p[1:]))
            elif p.startswith("LF") and p[2:].isdigit():
                meta.setdefault("L_factor", int(p[2:]))
            elif p.startswith("EX") and p[2:].isdigit():
                meta.setdefault("use_exog", int(p[2:]))
            elif p.startswith("S") and p[1:].isdigit():
                meta.setdefault("step_size", int(p[1:]))
            elif p.startswith("W") and p[1:].isdigit():
                meta.setdefault("n_windows", int(p[1:]))
            elif p.startswith("SEED") and p[4:].isdigit():
                meta.setdefault("seed", int(p[4:]))
    except Exception:
        pass

    # derived fields
    if "input_size" not in meta:
        H = meta.get("H")
        lf = meta.get("L_factor")
        if isinstance(H, int) and isinstance(lf, int):
            meta["input_size"] = int(H) * int(lf)

    return meta

# =========================================================
# TIME WINDOW
# =========================================================

def apply_last_days(df: pd.DataFrame, last_days: Optional[int], *, time_col: str = "ds") -> pd.DataFrame:
    if last_days is None:
        return df
    if time_col not in df.columns:
        return df
    tmax = df[time_col].max()
    if pd.isna(tmax):
        return df
    t0 = pd.Timestamp(tmax) - pd.Timedelta(days=int(last_days))
    return df[df[time_col] >= t0]


# =========================================================
# PLOT HELPERS
# =========================================================

def _safe_quantile(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _clip_keep_mask(x: np.ndarray, qlo: float, qhi: float) -> Tuple[np.ndarray, Tuple[float, float], float]:
    lo = _safe_quantile(x, qlo)
    hi = _safe_quantile(x, qhi)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        mask = np.isfinite(x)
        return mask, (lo, hi), float(np.mean(mask)) if mask.size else 0.0
    mask = np.isfinite(x) & (x >= lo) & (x <= hi)
    return mask, (lo, hi), float(np.mean(mask))


def _date_axis(ax: plt.Axes) -> None:
    loc = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _run_tag(meta: Dict[str, Any], run_root: Path) -> str:
    rn = str(meta.get("run_name") or "").strip()
    return rn if rn else run_root.name


def _fname_prefix(meta: Dict[str, Any], run_root: Path) -> str:
    rn = _run_tag(meta, run_root)
    model = str(meta.get("model", "")).upper()
    if model and not rn.upper().startswith(model):
        return f"{model}__{rn}"
    return rn


def _format_ds_span(df: pd.DataFrame, *, col: str = "ds") -> str:
    if col not in df.columns or len(df) == 0:
        return "ds=?"
    a = pd.Timestamp(df[col].min()).date().isoformat()
    b = pd.Timestamp(df[col].max()).date().isoformat()
    return f"{col}=[{a}..{b}]"


def _title_tag(
    meta: Dict[str, Any],
    *,
    gm: Optional[Dict[str, float]] = None,
    view: Optional[str] = None,
    n_points: Optional[int] = None,
    ds_span: Optional[str] = None,
) -> str:
    dataset = str(meta.get("dataset", "")).upper() or "DATASET"
    model = str(meta.get("model", meta.get("model_name", ""))).upper() or "MODEL"

    seed = meta.get("seed", "?")
    use_exog = meta.get("use_exog", "?")

    H = meta.get("H", "?")
    Lf = meta.get("L_factor", "?")
    step = meta.get("step_size", "?")
    nw = meta.get("n_windows", "?")

    # ----------------- Line 1 (priority 1) -----------------
    line1_parts = [
        f"{model} {dataset}",
        f"H={H} Lf={Lf}",
        f"step={step} Exog={use_exog}",
        
    ]
    if view:
        line1_parts.append(f"view={view}")
    line1 = " | ".join(line1_parts)
    '''
    # ----------------- Line 2 (priority 2, only if useful) -----------------
    line2_parts: List[str] = []
    if ds_span:
        line2_parts.append(ds_span)
    if gm:
        line2_parts.append(f"MAE={gm['MAE']:.3f} RMSE={gm['RMSE']:.3f} sMAPE={gm['sMAPE']:.4f}")

    if line2_parts:
        return line1 + "\n" + " | ".join(line2_parts)
    '''''
    return line1


def _bincount_mean_by_key(keys: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean(values) per key using factorize+bincount (fast, no groupby-apply).
    Returns:
      uniq_keys (as numpy array)
      mean_values float64
    """
    codes, uniq = pd.factorize(keys, sort=True)
    n = int(len(uniq))
    if n == 0:
        return np.asarray(uniq), np.asarray([], dtype=float)

    cnt = np.bincount(codes, minlength=n).astype(np.float64)
    s = np.bincount(codes, weights=np.asarray(values, dtype=np.float64), minlength=n).astype(np.float64)
    mean = s / np.maximum(1.0, cnt)

    # pandas may return Index or ndarray; normalize to ndarray
    if hasattr(uniq, "to_numpy"):
        uniq_arr = uniq.to_numpy()
    else:
        uniq_arr = np.asarray(uniq)

    return uniq_arr, mean


# =========================================================
# PLOTS (EXISTING)
# =========================================================

def plot_error_overview_2x2(
    df_view: pd.DataFrame,
    per_uid: pd.DataFrame,
    out_path: Path,
    meta: Dict[str, Any],
    *,
    view: str,
    clip: Tuple[float, float] = (0.01, 0.99),
) -> None:
    y = df_view["y"].to_numpy(dtype=float)
    yhat = df_view["yhat"].to_numpy(dtype=float)
    err = yhat - y

    fig, axes = plt.subplots(2, 2, figsize=(18, 8))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # (1) signed error histogram clipped (symmetric x, % y, median only)
    mask, (lo, hi), kept = _clip_keep_mask(err, clip[0], clip[1])
    err_clip = err[mask]

    if err_clip.size:
        w = np.full(err_clip.shape, 100.0 / float(err_clip.size), dtype=float)  # percent
        ax1.hist(err_clip, bins=80, weights=w)
        med = float(np.median(err_clip))
        ax1.axvline(0.0, color="red", linestyle="--")
        ax1.axvline(med, color="red", linestyle="-", linewidth=1.2)
        # symmetric limits around zero
        lim = float(max(abs(lo), abs(hi)))
        if np.isfinite(lim) and lim > 0:
            ax1.set_xlim(-lim, +lim)
        ax1.text(
            0.02, 0.98,
            f"clip=[{lo:.2f}, {hi:.2f}] | kept={kept*100:.1f}%\n"
            f"median={med:.2f}",
            transform=ax1.transAxes, va="top", ha="left",
        )
    else:
        ax1.hist([], bins=80)

    ax1.set_title("Signed error histogram (yhat - y), clipped")
    ax1.set_xlabel("error")
    ax1.set_ylabel("% of points (within clip)")
    ax1.grid(True, alpha=0.25)

    # (2) nMAE histogram clipped (FD bins, % y, median + p90, count > 0.3)
    nmae = per_uid["nmae"].to_numpy(dtype=float)
    mask2, (lo2, hi2), kept2 = _clip_keep_mask(nmae, clip[0], clip[1])
    nmae_clip = nmae[mask2]

    if nmae_clip.size:
        # adaptive binning (Freedman–Diaconis)
        edges = np.histogram_bin_edges(nmae_clip, bins="fd")
        w2 = np.full(nmae_clip.shape, 100.0 / float(nmae_clip.size), dtype=float)  # percent
        ax2.hist(nmae_clip, bins=edges, weights=w2)

        med2 = float(np.median(nmae_clip))
        p90 = float(np.quantile(nmae_clip, 0.90))
        ax2.axvline(med2, color="red", linestyle="-", linewidth=1.2, label="median")
        ax2.axvline(p90, color="red", linestyle="--", linewidth=1.2, label="p90")

        n_gt = int(np.sum(nmae > 0.30))
        pct_gt = 100.0 * float(n_gt) / max(1, int(np.isfinite(nmae).sum()))

        ax2.text(
            0.4, 0.98,
            f"clip=[{lo2:.4f}, {hi2:.4f}] | kept={kept2*100:.1f}%\n"
            f"median={med2:.3f} | p90={p90:.3f}\n"
            f">0.30: {n_gt} series ({pct_gt:.1f}%)",
            transform=ax2.transAxes, va="top", ha="left",
        )
        ax2.legend(loc="upper right")
    else:
        ax2.hist([], bins=60)

    ax2.set_title("Per-series nMAE histogram (MAE / mean(|y|)), clipped")
    ax2.set_xlabel("nMAE")
    ax2.set_ylabel("% of series (within clip)")
    ax2.grid(True, alpha=0.25)

    # (3) Top-10 by normalized max |error| (vertical, x=unique_id, normalized by mean load)
    if "mean_abs_y" in per_uid.columns:
        denom = per_uid["mean_abs_y"].to_numpy(dtype=float)
        denom = np.where(np.isfinite(denom) & (denom > 0), denom, np.nan)
        norm_max = per_uid["max_abs_err"].to_numpy(dtype=float) / denom
        tmp = per_uid[["unique_id"]].copy()
        tmp["norm_max_abs_err"] = norm_max
        top = tmp.sort_values("norm_max_abs_err", ascending=False).head(10)
        ax3.bar(top["unique_id"].astype(str).tolist(), top["norm_max_abs_err"].to_numpy(dtype=float))
        ax3.set_title("Top-10 series by max |error| / mean(|y|)")
        ax3.set_ylabel("max |error| / mean(|y|)")
    else:
        # fallback (should not happen if per_series_metrics_fast provides mean_abs_y)
        top = per_uid.sort_values("max_abs_err", ascending=False).head(10)
        ax3.bar(top["unique_id"].astype(str).tolist(), top["max_abs_err"].to_numpy(dtype=float))
        ax3.set_title("Top-10 series by max |error|")
        ax3.set_ylabel("max |error|")

    ax3.set_xlabel("unique_id")
    ax3.tick_params(axis="x", labelrotation=90)
    ax3.grid(True, axis="y", alpha=0.25)

    # (4) Pareto/Lorenz by sum_abs_err + Gini + annotate top 5/10% + shaded area
    # (4) Pareto/Lorenz by sum_abs_err + annotate top 5/10%
    p = per_uid.sort_values("sum_abs_err", ascending=False).reset_index(drop=True)
    w = p["sum_abs_err"].to_numpy(dtype=float)
    w = w[np.isfinite(w)]
    if w.size:
        c = np.cumsum(w)
        tot = float(c[-1]) if c.size else 1.0
        yy = 100.0 * (c / max(1e-12, tot))
        xx = 100.0 * (np.arange(1, len(w) + 1) / max(1, len(w)))

        ax4.plot(xx, yy, label="Cumulative |error|")
        ax4.plot([0, 100], [0, 100], linestyle="--", label="Equality line")
        ax4.fill_between(xx, yy, xx, alpha=0.15)

        n = int(len(w))
        for share in (0.05, 0.10):
            k = max(1, int(np.ceil(share * n)))
            xk = 100.0 * k / n
            yk = 100.0 * float(np.sum(w[:k]) / max(1e-12, tot))
            ax4.axvline(xk, linestyle=":", linewidth=1.0)
            ax4.axhline(yk, linestyle=":", linewidth=1.0)
            ax4.text(xk + 1.0, min(99.0, yk + 1.0), f"top {int(share*100)}% → {yk:.1f}%")

    ax4.set_title("Pareto/Lorenz: cumulative |error| vs % series")
    ax4.set_xlabel("% of series (worst → best)")
    ax4.set_ylabel("Cumulative % of total |error|")
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="lower right")


    gm = global_metrics_from_arrays(y, yhat)
    fig.suptitle(
        _title_tag(
            meta,
            gm=gm,
            view=view,
            n_points=len(df_view),
            ds_span=_format_ds_span(df_view, col="ds"),
        ),
        y=0.995,
        fontsize=12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_selected_grid_zscore_train(
    df_view: pd.DataFrame,
    per_uid: pd.DataFrame,
    out_path: Path,
    meta: Dict[str, Any],
    scaler_stats: pd.DataFrame,
    *,
    view: str,
    k: int = 12,
    last_days: Optional[int] = 30,
    worst_by: str = "sum_abs_err",
) -> Optional[Path]:
    k_eff = min(int(k), int(per_uid["unique_id"].nunique()))
    if k_eff <= 0:
        return None

    sel = per_uid.sort_values(worst_by, ascending=False).head(k_eff)["unique_id"].tolist()

    df = df_view[df_view["unique_id"].isin(sel)].copy()
    df = df.sort_values(["unique_id", "ds"], kind="mergesort")
    df = apply_last_days(df, last_days, time_col="ds")

    if df.empty:
        return None

    y_z, yhat_z = apply_train_zscore(df[["unique_id", "y", "yhat"]], scaler_stats)
    df["y_z"] = y_z
    df["yhat_z"] = yhat_z

    ncols = min(4, k_eff)
    nrows = int(np.ceil(k_eff / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for i, uid in enumerate(sel):
        ax = axes[i]
        sub = df[df["unique_id"] == uid]
        ax.plot(sub["ds"], sub["y_z"], label="True (z)", linewidth=1.4)
        ax.plot(sub["ds"], sub["yhat_z"], label="Forecast (z)", linestyle="--", linewidth=1.2)
        _date_axis(ax)
        ax.set_title(f"uid={uid}")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper center", ncol=2)

    for j in range(k_eff, len(axes)):
        axes[j].axis("off")

    suffix = f" | Selected worst-{k_eff} series (train z-score)"
    if last_days is not None:
        suffix += f" | last_days={last_days}"

    fig.suptitle(
        _title_tag(meta, view=view) + "\n" + _format_ds_span(df, col="ds") + suffix,
        y=0.99,
        fontsize=12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path

# =========================================================
# mean_over_series
# =========================================================

def plot_mean_over_series(
    df_view: pd.DataFrame,
    out_path: Path,
    meta: Dict[str, Any],
    *,
    view: str,
    last_days: Optional[int] = 30,
) -> None:
    df = apply_last_days(df_view.sort_values("ds", kind="mergesort"), last_days, time_col="ds")

    is_single = _is_single_series(meta, df)
    if is_single:
        ts = df.groupby("ds", sort=True)[["y", "yhat"]].mean()
        title_prefix = "Series forecast"
        y_true = ts["y"].to_numpy()
        y_pred = ts["yhat"].to_numpy()
        x = ts.index
    else:
        mean_y = df.groupby("ds", sort=True)["y"].mean()
        mean_yhat = df.groupby("ds", sort=True)["yhat"].mean()
        title_prefix = "Mean over series"
        y_true = mean_y.to_numpy()
        y_pred = mean_yhat.to_numpy()
        x = mean_y.index

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, y_true, label="True", linewidth=1.6)
    ax.plot(x, y_pred, label="Forecast", linestyle="--", linewidth=1.4)
    _date_axis(ax)

    ax.set_title(f"{title_prefix} | {_title_tag(meta, view=view)}")
    ax.set_ylabel(str(meta.get("y_label", "Load")))
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
# ========================================
# plot_rolling_stability
# ========================================

def plot_rolling_stability(
    df_cv: pd.DataFrame,
    out_path: Path,
    meta: Dict[str, Any],
    *,
    view: str,
    last_days: Optional[int],
) -> Optional[Path]:
    if "cutoff" not in df_cv.columns:
        return None

    df = df_cv[["cutoff", "y", "yhat"]].copy()
    df["cutoff"] = pd.to_datetime(df["cutoff"], errors="raise")
    df = apply_last_days(df, last_days, time_col="cutoff")

    y = df["y"].to_numpy(dtype=float)
    yhat = df["yhat"].to_numpy(dtype=float)
    abs_err = np.abs(yhat - y)

    cutoffs, mae_by = _bincount_mean_by_key(df["cutoff"].to_numpy(), abs_err)
    if mae_by.size == 0:
        return None

    cutoffs = pd.to_datetime(pd.Series(cutoffs))
    order = np.argsort(cutoffs.to_numpy())
    cutoffs = cutoffs.iloc[order].to_numpy()
    mae_by = mae_by[order]

    mu = float(np.mean(mae_by))
    sd = float(np.std(mae_by, ddof=0))
    cv = float(sd / mu) if mu > 0 else float("nan")
    max_mae = float(np.max(mae_by))
    max_idx = int(np.argmax(mae_by))
    max_cutoff = pd.to_datetime(cutoffs[max_idx])

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    ax1, ax2 = axes[0], axes[1]

    ax1.plot(cutoffs, mae_by, linewidth=1.6, label="MAE per cutoff (pooled)")
    ax1.axhline(mu, linestyle="--", linewidth=1.2, label="mean")
    ax1.fill_between(cutoffs, mu - sd, mu + sd, alpha=0.15, label="mean ± std")
    ax1.axvline(max_cutoff, linestyle=":", linewidth=1.0)

    ax1.text(
        0.02,
        0.95,
        f"max={max_mae:.2f} @ {max_cutoff.date()}\nCV={cv:.3f}",
        transform=ax1.transAxes,
        va="top",
        ha="left",
    )

    ax1.set_title("Rolling stability: pooled MAE per cutoff")
    ax1.set_ylabel("MAE")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="center left", bbox_to_anchor=(0.4, 0.85))
    _date_axis(ax1)

    mask, (lo, hi), kept = _clip_keep_mask(mae_by, 0.01, 0.99)
    vals = mae_by[mask]
    if vals.size:
        w = np.full(vals.shape, 100.0 / float(vals.size), dtype=float)
        p90 = float(np.quantile(vals, 0.90))
        ax2.hist(vals, bins=50, weights=w)
        ax2.axvline(mu, color="red", linestyle="-", linewidth=1.2, label="mean")
        ax2.axvline(p90, color="red", linestyle="--", linewidth=1.2, label="p90")
        ax2.legend(loc="best")
    else:
        p90 = float("nan")
        ax2.hist([], bins=50)

    ax2.set_title("Distribution of cutoff MAE (clipped 1–99%)")
    ax2.set_xlabel("MAE per cutoff")
    ax2.set_ylabel("% of cutoffs")
    ax2.grid(True, alpha=0.25)
    ax2.text(
        0.37,
        0.95,
        f"cutoffs={int(mae_by.size)} | clip=[{lo:.2f},{hi:.2f}] | kept={kept*100:.1f}%\n"
        f"mean={mu:.3f} | std={sd:.3f} | p90={p90:.3f}",
        transform=ax2.transAxes,
        va="top",
        ha="left",
    )

    suffix = " | Rolling-window diagnostics"
    if last_days is not None:
        suffix += f" | last_days={last_days}"

    fig.suptitle(
        _title_tag(meta, view=view) + "\n" + _format_ds_span(df, col="cutoff") + suffix,
        y=0.995,
        fontsize=12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path







# ========================================
# scale_sensitivity_scatter
# ========================================

def plot_scale_sensitivity_scatter(
    per_uid: pd.DataFrame,
    out_path: Path,
    meta: Dict[str, Any],
    *,
    view: str,
) -> Optional[Path]:
    if _is_single_series(meta) or int(per_uid["unique_id"].nunique()) <= 1:
        return None

    x = per_uid["mean_abs_y"].to_numpy(dtype=float)
    y = per_uid["mae"].to_numpy(dtype=float)
    ids = per_uid["unique_id"].astype(str).to_numpy()

    m = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    x = x[m]
    y = y[m]
    ids = ids[m]

    if x.size == 0:
        return None

    xlog = np.log1p(x)
    ylog = np.log1p(y)

    slope = float("nan")
    intercept = float("nan")
    r2 = float("nan")

    if xlog.size >= 2:
        slope, intercept = np.polyfit(xlog, ylog, deg=1)
        yfit = slope * xlog + intercept
        ss_res = float(np.sum((ylog - yfit) ** 2))
        ss_tot = float(np.sum((ylog - float(np.mean(ylog))) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xlog, ylog, s=10, alpha=0.35, label="series")

    k = min(3, int(xlog.size))
    if k > 0:
        idx_sorted = np.argsort(xlog)
        idx_small = idx_sorted[:k]
        idx_large = idx_sorted[-k:]

        texts = []
        for i in idx_small:
            texts.append(ax.text(xlog[i], ylog[i], ids[i], fontsize=8, ha="left", va="bottom"))
        for i in idx_large:
            texts.append(ax.text(xlog[i], ylog[i], ids[i], fontsize=8, ha="left", va="bottom"))

        try:
            from adjustText import adjust_text
            adjust_text(
                texts,
                ax=ax,
                expand_text=(1.05, 1.15),
                expand_points=(1.10, 1.25),
                force_text=(0.10, 0.25),
                force_points=(0.05, 0.15),
                only_move={"text": "xy"},
            )
        except Exception:
            pass

    if np.isfinite(slope) and np.isfinite(intercept):
        xx = np.array([float(np.min(xlog)), float(np.max(xlog))], dtype=float)
        yy = slope * xx + intercept
        ax.plot(xx, yy, linewidth=1.6, linestyle="--", color="red", label="OLS fit")
        ax.text(
            0.02,
            0.98,
            f"slope={slope:.3f}\nR²={r2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )

    fig.suptitle(
        f"{_title_tag(meta, view=view)}\nScale sensitivity: per-series MAE vs mean(|y|) (log1p–log1p)",
        y=0.95,
        fontsize=12,
        linespacing=1.4,
    )

    ax.set_xlabel("log1p(mean(|y|))")
    ax.set_ylabel("log1p(MAE)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
# =========================================================
# RUN RESOLVE
# =========================================================

def run_root_from_parts(dataset: str, run_date: str, H: int, run_name: str) -> Path:
    return PROJECT_ROOT / "experiments" / dataset.upper() / run_date / f"H{int(H)}" / run_name


# =========================================================
# CLI
# =========================================================

@dataclass(frozen=True)
class ReportArgs:
    run_root: Path
    view: str
    out_dir: Path
    last_days: Optional[int]


def parse_args() -> ReportArgs:
    p = argparse.ArgumentParser()

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run_root", type=str, default=None)
    g.add_argument("--run_parts", type=str, nargs=4, metavar=("DATASET", "RUN_DATE", "H", "RUN_NAME"))

    p.add_argument("--view", type=str, choices=["stitched", "cutoffmean"], default="stitched")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--last_days", type=int, default=30)

    a = p.parse_args()

    if a.run_root:
        rr = Path(a.run_root)
    else:
        dataset, run_date, H, run_name = a.run_parts
        rr = run_root_from_parts(dataset=dataset, run_date=run_date, H=int(H), run_name=run_name)

    out_dir = Path(a.out_dir) if a.out_dir else (rr / "reports")
    last_days = int(a.last_days) if a.last_days is not None else None

    return ReportArgs(run_root=rr, view=str(a.view), out_dir=out_dir, last_days=last_days)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    args = parse_args()

    run_root = args.run_root
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    pred_path = run_root / "predictions_cv.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions_cv.parquet: {pred_path}")

    payload = load_json(run_root / "metrics.json")
    meta = resolve_meta(payload, run_root=run_root)

    df_cv = load_predictions_cv(pred_path)
    df_view = make_view_cutoffmean(df_cv) if args.view == "cutoffmean" else make_view_stitched(df_cv)

    per_uid = per_series_metrics_fast(df_view)

    scaler = None
    scaler_path = run_root / "scaler_stats.parquet"
    if scaler_path.exists():
        scaler = load_scaler_stats(scaler_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # prefix کوتاه
    prefix = str(meta.get("model", "M")).upper()

    created_outputs = {}

    # 1. error overview
    p_err = args.out_dir / f"{prefix}_err.png"
    plot_error_overview_2x2(df_view, per_uid, p_err, meta, view=args.view, clip=(0.01, 0.99))
    created_outputs["err"] = str(p_err)

    # 2. grid
    if scaler is not None:
        p_grid = args.out_dir / f"{prefix}_grid.png"
        out = plot_selected_grid_zscore_train(
            df_view, per_uid, p_grid, meta, scaler,
            view=args.view, k=12, last_days=args.last_days, worst_by="sum_abs_err"
        )
        created_outputs["grid"] = str(out) if out else None
    else:
        created_outputs["grid"] = None

    # 3. mean
    p_mean = args.out_dir / f"{prefix}_mean.png"
    plot_mean_over_series(df_view, p_mean, meta, view=args.view, last_days=args.last_days)
    created_outputs["mean"] = str(p_mean)

    # 4. scatter
    p_scatter = args.out_dir / f"{prefix}_scale.png"
    out = plot_scale_sensitivity_scatter(per_uid, p_scatter, meta, view=args.view)
    created_outputs["scale"] = str(out) if out else None

    # 5. rolling
    p_roll = args.out_dir / f"{prefix}_roll.png"
    try:
        out = plot_rolling_stability(df_cv, p_roll, meta, view=args.view, last_days=args.last_days)
        created_outputs["roll"] = str(out) if out else None
    except Exception as e:
        print(f"[WARN] rolling plot failed: {e}")
        created_outputs["roll"] = None

    # summary کوتاه
    summary = {
        "run": str(run_root),
        "outputs": created_outputs,
    }
    write_json(args.out_dir / f"{prefix}_summary.json", summary)

    print(f"[REPORT] saved -> {args.out_dir}")


if __name__ == "__main__":
    main()