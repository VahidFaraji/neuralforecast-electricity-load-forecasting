from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BASE_COLS = ("unique_id", "ds", "y", "cutoff")


# =========================================================
# IO
# =========================================================

def load_predictions_cv(path: Path, *, yhat_col: str = "yhat") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    df = pd.read_parquet(path)

    req = {"unique_id", "ds", "y"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"predictions missing columns: {sorted(miss)}")

    if yhat_col not in df.columns:
        cand = [c for c in df.columns if c not in BASE_COLS]
        if not cand:
            raise ValueError(f"predictions missing '{yhat_col}' and no candidate yhat column found")
        df = df.rename(columns={cand[0]: yhat_col})

    keep = ["unique_id", "ds", "y", yhat_col] + (["cutoff"] if "cutoff" in df.columns else [])
    df = df[keep]

    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"])
    if "cutoff" in df.columns:
        df["cutoff"] = pd.to_datetime(df["cutoff"])

    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def load_scaler_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing scaler stats: {path}")

    df = pd.read_parquet(path)

    req = {"unique_id", "y_mean", "y_std"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"scaler_stats missing columns: {sorted(miss)}")

    out = df[["unique_id", "y_mean", "y_std"]]
    out["unique_id"] = out["unique_id"].astype(str)
    out["y_mean"] = out["y_mean"].astype(np.float64, copy=False)
    out["y_std"] = out["y_std"].astype(np.float64, copy=False)
    return out


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =========================================================
# VIEWS
# =========================================================

def make_view_stitched(df_cv: pd.DataFrame) -> pd.DataFrame:
    if "cutoff" not in df_cv.columns:
        return df_cv[["unique_id", "ds", "y", "yhat"]]

    df = df_cv.sort_values(["unique_id", "ds", "cutoff"], kind="mergesort")
    df = df.drop_duplicates(subset=["unique_id", "ds"], keep="last")
    return df[["unique_id", "ds", "y", "yhat"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def make_view_cutoffmean(df_cv: pd.DataFrame) -> pd.DataFrame:
    if "cutoff" not in df_cv.columns:
        return df_cv[["unique_id", "ds", "y", "yhat"]]

    out = (
        df_cv.groupby(["unique_id", "ds"], as_index=False, sort=False)
        .agg(y=("y", "first"), yhat=("yhat", "mean"))
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    return out


# =========================================================
# CORE METRICS
# =========================================================

def _smape_from_arrays(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-12) -> float:
    y64 = y.astype(np.float64, copy=False)
    yhat64 = yhat.astype(np.float64, copy=False)
    denom = np.abs(y64) + np.abs(yhat64) + eps
    return float(np.mean(2.0 * np.abs(yhat64 - y64) / denom))


def stitched_pooled_metrics(df_cv: pd.DataFrame) -> Dict[str, float]:
    df = make_view_stitched(df_cv)
    y = df["y"].to_numpy(dtype=np.float32, copy=False)
    yhat = df["yhat"].to_numpy(dtype=np.float32, copy=False)

    err = (yhat - y).astype(np.float64, copy=False)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    mse = float(np.mean(err * err)) 
    smape = _smape_from_arrays(y, yhat)

    return {"MAE_pooled": mae, "RMSE_pooled": rmse, "MSE_pooled": mse, "sMAPE_pooled": smape}


def _factorize_cutoff(df_cv: pd.DataFrame) -> Tuple[np.ndarray, int]:
    cutoff = df_cv["cutoff"]
    codes, uniques = pd.factorize(cutoff, sort=False)
    return codes.astype(np.int64, copy=False), int(uniques.size)


def rolling_mae_series_stats_streaming(df_cv: pd.DataFrame) -> Dict[str, Optional[float]]:
    if "cutoff" not in df_cv.columns:
        return {"MAE_cutoff_series_mean": None, "MAE_cutoff_series_std": None, "n_cutoffs": None}

    codes, n = _factorize_cutoff(df_cv)
    if n == 0:
        return {"MAE_cutoff_series_mean": None, "MAE_cutoff_series_std": None, "n_cutoffs": 0}

    y = df_cv["y"].to_numpy(dtype=np.float32, copy=False)
    yhat = df_cv["yhat"].to_numpy(dtype=np.float32, copy=False)

    abs_err = np.abs(yhat - y).astype(np.float64, copy=False)
    cnt = np.bincount(codes, minlength=n).astype(np.float64, copy=False)
    sabs = np.bincount(codes, weights=abs_err, minlength=n).astype(np.float64, copy=False)

    mae_by = sabs / np.maximum(1.0, cnt)
    return {
        "MAE_cutoff_series_mean": float(np.mean(mae_by)),
        "MAE_cutoff_series_std": float(np.std(mae_by, ddof=0)),
        "n_cutoffs": int(n),
    }


def cutoffmean_metrics_streaming(df_cv: pd.DataFrame) -> Dict[str, Optional[float]]:
    if "cutoff" not in df_cv.columns:
        m = stitched_pooled_metrics(df_cv)
        return {"MAE_cutoff_mean": m["MAE_pooled"], "RMSE_cutoff_mean": m["RMSE_pooled"], "sMAPE_cutoff_mean": m["sMAPE_pooled"]}

    codes, n = _factorize_cutoff(df_cv)
    if n == 0:
        return {"MAE_cutoff_mean": None, "RMSE_cutoff_mean": None, "sMAPE_cutoff_mean": None}

    y = df_cv["y"].to_numpy(dtype=np.float32, copy=False)
    yhat = df_cv["yhat"].to_numpy(dtype=np.float32, copy=False)

    err = (yhat - y).astype(np.float64, copy=False)
    abs_err = np.abs(err)
    sq_err = err * err

    denom = (np.abs(y.astype(np.float64, copy=False)) + np.abs(yhat.astype(np.float64, copy=False)) + 1e-12)
    sm_obs = 2.0 * abs_err / denom

    cnt = np.bincount(codes, minlength=n).astype(np.float64, copy=False)
    s_abs = np.bincount(codes, weights=abs_err, minlength=n).astype(np.float64, copy=False)
    s_sq = np.bincount(codes, weights=sq_err, minlength=n).astype(np.float64, copy=False)
    s_sm = np.bincount(codes, weights=sm_obs, minlength=n).astype(np.float64, copy=False)

    mae_by = s_abs / np.maximum(1.0, cnt)
    rmse_by = np.sqrt(s_sq / np.maximum(1.0, cnt))
    smape_by = s_sm / np.maximum(1.0, cnt)

    return {
        "MAE_cutoff_mean": float(np.mean(mae_by)),
        "RMSE_cutoff_mean": float(np.mean(rmse_by)),
        "sMAPE_cutoff_mean": float(np.mean(smape_by)),
    }

# =========================================================
# COMPATIBILITY EXPORTS (reporting_nf.py expects these names)
# =========================================================

def global_metrics_from_arrays(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    """
    Global pooled metrics.
    Must stay stable because reporting_nf.py imports this symbol.
    """
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)

    err = yhat - y
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))

    denom = np.abs(y) + np.abs(yhat) + 1e-12
    smape = float(np.mean(2.0 * np.abs(err) / denom))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "sMAPE": smape,
        "MAE_pooled": mae,
    }




# =========================================================
# DIAGNOSTICS (optional, used by reporting)
# =========================================================

def per_series_metrics_fast(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    uid = df["unique_id"].to_numpy()
    y = df["y"].to_numpy(dtype=np.float64, copy=False)
    yhat = df["yhat"].to_numpy(dtype=np.float64, copy=False)

    codes, uniques = pd.factorize(uid, sort=False)
    n = int(uniques.size)

    abs_err = np.abs(yhat - y)
    abs_y = np.abs(y)

    cnt = np.bincount(codes, minlength=n).astype(np.float64, copy=False)
    sum_abs_err = np.bincount(codes, weights=abs_err, minlength=n).astype(np.float64, copy=False)
    sum_abs_y = np.bincount(codes, weights=abs_y, minlength=n).astype(np.float64, copy=False)

    max_abs_err = np.zeros(n, dtype=np.float64)
    np.maximum.at(max_abs_err, codes, abs_err)

    denom = np.maximum(1.0, cnt)
    mae = sum_abs_err / denom
    mean_abs_y = sum_abs_y / denom
    nmae = mae / (mean_abs_y + eps)

    out = pd.DataFrame(
        {
            "unique_id": uniques.astype(str),
            "sum_abs_err": sum_abs_err,
            "mae": mae,
            "max_abs_err": max_abs_err,
            "mean_abs_y": mean_abs_y,
            "nmae": nmae,
        }
    )
    return out.sort_values("mae", ascending=False).reset_index(drop=True)


def dist_stats(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": float("nan"), "mean": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
    }


def pareto_shares(weights: np.ndarray, shares: Sequence[float] = (0.01, 0.05, 0.10)) -> Dict[str, float]:
    w = weights[np.isfinite(weights)]
    if w.size == 0:
        return {f"top_{int(s * 100)}pct_share": float("nan") for s in shares}

    w = np.sort(w)[::-1]
    tot = float(np.sum(w))
    if tot <= 0:
        return {f"top_{int(s * 100)}pct_share": 0.0 for s in shares}

    out: Dict[str, float] = {}
    n = w.size
    for s in shares:
        k = max(1, int(np.ceil(float(s) * n)))
        out[f"top_{int(s * 100)}pct_share"] = float(np.sum(w[:k]) / tot)
    return out


def diagnostics_from_per_series(per_uid: pd.DataFrame) -> Dict[str, float]:
    nmae = per_uid["nmae"].to_numpy(dtype=np.float64, copy=False)
    max_abs_err = per_uid["max_abs_err"].to_numpy(dtype=np.float64, copy=False)
    sum_abs_err = per_uid["sum_abs_err"].to_numpy(dtype=np.float64, copy=False)

    out: Dict[str, float] = {}

    ds = dist_stats(nmae)
    out["nmae_median"] = ds["median"]
    out["nmae_mean"] = ds["mean"]
    out["nmae_p90"] = ds["p90"]
    out["nmae_p95"] = ds["p95"]

    out["max_abs_err_max"] = float(np.nanmax(max_abs_err)) if max_abs_err.size else float("nan")
    out["max_abs_err_p95"] = float(np.nanquantile(max_abs_err, 0.95)) if max_abs_err.size else float("nan")

    ps = pareto_shares(sum_abs_err, shares=(0.01, 0.05, 0.10))
    out["pareto_top_1pct_share"] = ps["top_1pct_share"]
    out["pareto_top_5pct_share"] = ps["top_5pct_share"]
    out["pareto_top_10pct_share"] = ps["top_10pct_share"]

    return out


# =========================================================
# NORMALIZATION (train-only, Option B)
# =========================================================

def apply_train_zscore(df: pd.DataFrame, scaler_stats: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    
    stats = scaler_stats.set_index("unique_id")[["y_mean", "y_std"]]

    uid = df["unique_id"].astype(str)
    mu = uid.map(stats["y_mean"]).to_numpy(dtype=np.float64)
    sd = uid.map(stats["y_std"]).to_numpy(dtype=np.float64)

    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)

    y = df["y"].to_numpy(dtype=np.float64)
    yhat = df["yhat"].to_numpy(dtype=np.float64)

    return (y - mu) / sd, (yhat - mu) / sd

def _scaler_index(scaler_stats: pd.DataFrame) -> pd.DataFrame:
    st = scaler_stats[["unique_id", "y_mean", "y_std"]].copy()
    st["unique_id"] = st["unique_id"].astype(str)
    st["y_mean"] = st["y_mean"].astype(np.float64, copy=False)
    st["y_std"] = st["y_std"].astype(np.float64, copy=False)
    return st.set_index("unique_id")[["y_mean", "y_std"]]


def _scaler_has_full_coverage(df: pd.DataFrame, scaler_stats: pd.DataFrame) -> bool:
    stats_ids = set(scaler_stats["unique_id"].astype(str).unique().tolist())
    df_ids = set(df["unique_id"].astype(str).unique().tolist())
    return df_ids.issubset(stats_ids)


def _map_mu_sd(df: pd.DataFrame, stats_idx: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    uid = df["unique_id"].astype(str)
    mu = uid.map(stats_idx["y_mean"]).to_numpy(dtype=np.float64, copy=False)
    sd = uid.map(stats_idx["y_std"]).to_numpy(dtype=np.float64, copy=False)

    if not (np.isfinite(mu).all() and np.isfinite(sd).all()):
        return None, None
    if not np.all(sd > 0):
        return None, None
    return mu, sd


def _zscore_arrays(y: np.ndarray, yhat: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y64 = y.astype(np.float64, copy=False)
    yhat64 = yhat.astype(np.float64, copy=False)
    return (y64 - mu) / sd, (yhat64 - mu) / sd


def stitched_pooled_metrics_norm(df_cv: pd.DataFrame, scaler_stats: pd.DataFrame) -> Optional[Dict[str, float]]:
    df = make_view_stitched(df_cv)
    stats_idx = _scaler_index(scaler_stats)
    mu, sd = _map_mu_sd(df, stats_idx)
    if mu is None:
        return None

    y = df["y"].to_numpy(dtype=np.float32, copy=False)
    yhat = df["yhat"].to_numpy(dtype=np.float32, copy=False)
    y_z, yhat_z = _zscore_arrays(y, yhat, mu, sd)

    err = (yhat_z - y_z).astype(np.float64, copy=False)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    smape = _smape_from_arrays(y_z, yhat_z)

    return {"MAE_pooled": mae, "RMSE_pooled": rmse, "sMAPE_pooled": smape}


def cutoffmean_metrics_norm_streaming(df_cv: pd.DataFrame, scaler_stats: pd.DataFrame) -> Optional[Dict[str, Optional[float]]]:
    stats_idx = _scaler_index(scaler_stats)
    mu, sd = _map_mu_sd(df_cv, stats_idx)
    if mu is None:
        return None

    y = df_cv["y"].to_numpy(dtype=np.float32, copy=False)
    yhat = df_cv["yhat"].to_numpy(dtype=np.float32, copy=False)
    y_z, yhat_z = _zscore_arrays(y, yhat, mu, sd)

    if "cutoff" not in df_cv.columns:
        err = (yhat_z - y_z).astype(np.float64, copy=False)
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        smape = _smape_from_arrays(y_z, yhat_z)
        return {"MAE_cutoff_mean": mae, "RMSE_cutoff_mean": rmse, "sMAPE_cutoff_mean": smape}

    codes, n = _factorize_cutoff(df_cv)
    if n == 0:
        return {"MAE_cutoff_mean": None, "RMSE_cutoff_mean": None, "sMAPE_cutoff_mean": None}

    err = (yhat_z - y_z).astype(np.float64, copy=False)
    abs_err = np.abs(err)
    sq_err = err * err

    denom = (np.abs(y_z) + np.abs(yhat_z) + 1e-12)
    sm_obs = 2.0 * abs_err / denom

    cnt = np.bincount(codes, minlength=n).astype(np.float64, copy=False)
    s_abs = np.bincount(codes, weights=abs_err, minlength=n).astype(np.float64, copy=False)
    s_sq = np.bincount(codes, weights=sq_err, minlength=n).astype(np.float64, copy=False)
    s_sm = np.bincount(codes, weights=sm_obs, minlength=n).astype(np.float64, copy=False)

    mae_by = s_abs / np.maximum(1.0, cnt)
    rmse_by = np.sqrt(s_sq / np.maximum(1.0, cnt))
    smape_by = s_sm / np.maximum(1.0, cnt)

    return {
        "MAE_cutoff_mean": float(np.mean(mae_by)),
        "RMSE_cutoff_mean": float(np.mean(rmse_by)),
        "sMAPE_cutoff_mean": float(np.mean(smape_by)),
    }


def rolling_mae_series_stats_norm_streaming(df_cv: pd.DataFrame, scaler_stats: pd.DataFrame) -> Optional[Dict[str, Optional[float]]]:
    if "cutoff" not in df_cv.columns:
        return {"MAE_cutoff_series_mean": None, "MAE_cutoff_series_std": None, "n_cutoffs": None}

    stats_idx = _scaler_index(scaler_stats)
    mu, sd = _map_mu_sd(df_cv, stats_idx)
    if mu is None:
        return None

    codes, n = _factorize_cutoff(df_cv)
    if n == 0:
        return {"MAE_cutoff_series_mean": None, "MAE_cutoff_series_std": None, "n_cutoffs": 0}

    y = df_cv["y"].to_numpy(dtype=np.float32, copy=False)
    yhat = df_cv["yhat"].to_numpy(dtype=np.float32, copy=False)
    y_z, yhat_z = _zscore_arrays(y, yhat, mu, sd)

    abs_err = np.abs(yhat_z - y_z).astype(np.float64, copy=False)
    cnt = np.bincount(codes, minlength=n).astype(np.float64, copy=False)
    sabs = np.bincount(codes, weights=abs_err, minlength=n).astype(np.float64, copy=False)

    mae_by = sabs / np.maximum(1.0, cnt)
    return {
        "MAE_cutoff_series_mean": float(np.mean(mae_by)),
        "MAE_cutoff_series_std": float(np.std(mae_by, ddof=0)),
        "n_cutoffs": int(n),
    }


# =========================================================
# PAYLOAD (Contract 10)
# =========================================================

def build_metrics_payload(
    *,
    meta: Dict[str, Any],
    df_cv: pd.DataFrame,
    scaler_stats: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    raw_stitched = stitched_pooled_metrics(df_cv)
    per_uid_raw = per_series_metrics_fast(make_view_stitched(df_cv))
    raw_diag = diagnostics_from_per_series(per_uid_raw)
    raw_stitched.update(raw_diag)

    raw_cutoffmean = cutoffmean_metrics_streaming(df_cv)
    raw_rolling = rolling_mae_series_stats_streaming(df_cv)

    out: Dict[str, Any] = {
        "meta": dict(meta),
        "metrics": {
            "raw": {
                "stitched": raw_stitched,
                "cutoffmean": raw_cutoffmean,
                "rolling": {
                    "MAE_cutoff_series_mean": raw_rolling["MAE_cutoff_series_mean"],
                    "MAE_cutoff_series_std": raw_rolling["MAE_cutoff_series_std"],
                    "n_cutoffs": raw_rolling["n_cutoffs"],
                },
            },
            "norm": {
                "available": False,
                "stitched": None,
                "cutoffmean": None,
                "rolling": None,
            },
        },
    }

    if scaler_stats is None:
        return out

    if not _scaler_has_full_coverage(df_cv, scaler_stats):
        return out

    stitched_n = stitched_pooled_metrics_norm(df_cv, scaler_stats)
    if stitched_n is None:
        return out

    df_norm = make_view_stitched(df_cv).copy()
    y_z, yhat_z = apply_train_zscore(df_norm[["unique_id", "y", "yhat"]], scaler_stats)
    df_norm["y"] = y_z
    df_norm["yhat"] = yhat_z

    per_uid_norm = per_series_metrics_fast(df_norm)
    norm_diag = diagnostics_from_per_series(per_uid_norm)
    stitched_n.update(norm_diag)

    cutoffmean_n = cutoffmean_metrics_norm_streaming(df_cv, scaler_stats)
    rolling_n = rolling_mae_series_stats_norm_streaming(df_cv, scaler_stats)
    if cutoffmean_n is None or rolling_n is None:
        return out

    out["metrics"]["norm"] = {
        "available": True,
        "stitched": stitched_n,
        "cutoffmean": cutoffmean_n,
        "rolling": rolling_n,
    }
    return out
# =========================================================
# CSV UPSERT
# =========================================================

# metrics_export.py (MODIFICATION)
def upsert_metrics_runs_csv(csv_path: Path, row: Dict[str, Any], key_cols: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        return

    df = pd.read_csv(csv_path)
    row_df = pd.DataFrame([row])

    for k in key_cols:
        if k not in df.columns:
            df[k] = np.nan

    for c in row_df.columns:
        if c not in df.columns:
            df[c] = np.nan

    def _key_tuple(frame: pd.DataFrame) -> pd.Series:
        return frame[key_cols].astype(str).agg("§".join, axis=1)

    df_key = _key_tuple(df)
    row_key = _key_tuple(row_df).iloc[0]

    hit = df_key == row_key
    if hit.any():
        idx = int(np.flatnonzero(hit.to_numpy())[0])
        for c in row_df.columns:
            val = row_df.at[0, c]
            if c in df.columns and df[c].dtype != object:
                if isinstance(val, str):
                    df[c] = df[c].astype(object)
            df.at[idx, c] = val
    else:
        for c in row_df.columns:
            val = row_df.at[0, c]
            if c in df.columns and df[c].dtype != object and isinstance(val, str):
                df[c] = df[c].astype(object)
        row_aligned = row_df.reindex(columns=df.columns, fill_value=np.nan)
        df = pd.concat([df, row_aligned], axis=0, ignore_index=True)

    df.to_csv(csv_path, index=False, encoding="utf-8")
