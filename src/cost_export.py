from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================================================
# CONSTANTS
# =========================================================

TIMING_RUN_COLS: List[str] = [
    "summary_type",
    "dataset",
    "model",
    "model_family",
    "use_exog",
    "exog_variant",
    "H",
    "L_factor",
    "input_size",
    "batch_size",
    "n_channels",
    "bytes_per_element",
    "has_futr_exog",
    "has_hist_exog",
    "has_stat_exog",
    "insample_y_numel",
    "futr_exog_numel",
    "hist_exog_numel",
    "stat_exog_numel",
    "M_batch",
    "seed",
    "train_seconds",
    "inf_seconds",
    "run_date",
    "run_name",
    "mode",
    "eval_mode",
    "step_size",
    "step_size_spec",
    "n_windows",
    "train_size",
    "val_size",
    "test_size",
    "split_mode",
    "split_source",
]

TIMING_SUMMARY_GROUP_COLS: List[str] = [
    "dataset",
    "model",
    "model_family",
    "use_exog",
    "exog_variant",
    "H",
    "L_factor",
    "input_size",
    "batch_size",
    "n_channels",
    "bytes_per_element",
    "has_futr_exog",
    "has_hist_exog",
    "has_stat_exog",
    "insample_y_numel",
    "futr_exog_numel",
    "hist_exog_numel",
    "stat_exog_numel",
    "mode",
    "eval_mode",
    "step_size",
    "step_size_spec",
    "n_windows",
    "train_size",
    "val_size",
    "test_size",
    "split_mode",
    "split_source",
]

TIMING_VALUE_COLS: List[str] = [
    "train_seconds",
    "inf_seconds",
    "M_batch",
]

COST_BENEFIT_COLS: List[str] = [
    "dataset",
    "model",
    "model_family",
    "H",
    "L_factor",
    "input_size",
    "mode",
    "eval_mode",
    "step_size",
    "step_size_spec",
    "n_windows",
    "exog_variant",
    "MAE_norm_noX",
    "MAE_norm_X",
    "train_seconds_noX",
    "train_seconds_X",
    "inf_seconds_noX",
    "inf_seconds_X",
    "M_batch_noX",
    "M_batch_X",
    "delta_MAE_norm",
    "delta_T_train",
    "delta_T_inf",
    "delta_M_batch",
]


# =========================================================
# BASIC HELPERS
# =========================================================

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return int(x)
    except Exception:
        return default


def _safe_div(num: float, den: float) -> Optional[float]:
    num_f = _safe_float(num, None)
    den_f = _safe_float(den, None)
    if num_f is None or den_f is None or abs(den_f) < 1e-12:
        return None
    return float(num_f / den_f)


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    return df


def _reindex_with_tail(df: pd.DataFrame, ordered_cols: Sequence[str]) -> pd.DataFrame:
    head = [c for c in ordered_cols if c in df.columns]
    tail = [c for c in df.columns if c not in head]
    return df.reindex(columns=head + tail)


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _blank_row_like(columns: Sequence[str]) -> Dict[str, Any]:
    return {c: "" for c in columns}


def _summary_std(series: pd.Series) -> Optional[float]:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return None
    return float(x.std(ddof=0))


def _summary_mean(series: pd.Series) -> Optional[float]:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return None
    return float(x.mean())


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# BUILD RUN-LEVEL TIMING ROW
# =========================================================
def build_timing_row(
    *,
    meta: Dict[str, Any],
    train_seconds: float,
    inf_seconds: float,
    batch_size: int,
    input_size: int,
    n_channels: int,
    step_size_spec: Optional[int | str] = None,
    bytes_per_element: int = 4,
    insample_y_numel: int = 0,
    futr_exog_numel: int = 0,
    hist_exog_numel: int = 0,
    stat_exog_numel: int = 0,
    has_futr_exog: int = 0,
    has_hist_exog: int = 0,
    has_stat_exog: int = 0,
) -> Dict[str, Any]:
    """
    Build one standardized RUN row for timings_runs.csv.

    M_batch follows the paper-aligned forward-input payload proxy:
        M_batch =
            bytes(insample_y)
          + bytes(futr_exog)
          + bytes(hist_exog)
          + bytes(stat_exog)

        where bytes(Z) = numel(Z) * bytes_per_element
    """
    dataset = str(meta.get("dataset", "")).upper()
    model = str(meta.get("model", "")).upper()
    model_family = str(meta.get("model_family", meta.get("family", ""))).upper()
    use_exog = _safe_int(meta.get("use_exog"), 0)
    H = _safe_int(meta.get("H"))
    seed = _safe_int(meta.get("seed"))
    L_factor = _safe_int(meta.get("L_factor"))

    batch_size_i = int(batch_size)
    input_size_i = int(input_size)
    n_channels_i = int(n_channels)

    bytes_per_element_i = int(bytes_per_element)

    insample_y_numel_i = int(insample_y_numel)
    futr_exog_numel_i = int(futr_exog_numel)
    hist_exog_numel_i = int(hist_exog_numel)
    stat_exog_numel_i = int(stat_exog_numel)

    has_futr_exog_i = int(has_futr_exog)
    has_hist_exog_i = int(has_hist_exog)
    has_stat_exog_i = int(has_stat_exog)

    # Paper-aligned forward-input memory proxy
    M_batch = float(
        bytes_per_element_i
        * (
            insample_y_numel_i
            + futr_exog_numel_i
            + hist_exog_numel_i
            + stat_exog_numel_i
        )
    )

    row: Dict[str, Any] = {
        "summary_type": "RUN",
        "dataset": dataset,
        "model": model,
        "model_family": model_family,
        "use_exog": use_exog,
        "exog_variant": meta.get("exog_variant"),
        "H": H,
        "L_factor": L_factor,
        "input_size": input_size_i,
        "batch_size": batch_size_i,
        "n_channels": n_channels_i,
        "bytes_per_element": bytes_per_element_i,
        "has_futr_exog": has_futr_exog_i,
        "has_hist_exog": has_hist_exog_i,
        "has_stat_exog": has_stat_exog_i,
        "insample_y_numel": insample_y_numel_i,
        "futr_exog_numel": futr_exog_numel_i,
        "hist_exog_numel": hist_exog_numel_i,
        "stat_exog_numel": stat_exog_numel_i,
        "M_batch": M_batch,
        "seed": seed,
        "train_seconds": float(train_seconds),
        "inf_seconds": float(inf_seconds),
        "run_date": meta.get("run_date"),
        "run_name": meta.get("run_name"),
        "mode": meta.get("mode"),
        "eval_mode": meta.get("eval_mode"),
        "step_size": _safe_int(meta.get("step_size")),
        "step_size_spec": meta.get("step_size_spec", step_size_spec),
        "n_windows": _safe_int(meta.get("n_windows")),
        "train_size": _safe_int(meta.get("train_size")),
        "val_size": _safe_int(meta.get("val_size")),
        "test_size": _safe_int(meta.get("test_size")),
        "split_mode": meta.get("split_mode"),
        "split_source": meta.get("split_source"),
    }
    return row


# =========================================================
# RUN-LEVEL CSV UPSERT
# =========================================================

def upsert_timings_runs_csv(
    csv_path: Path,
    row: Dict[str, Any],
    key_cols: Sequence[str],
) -> None:
    """
    Upsert RUN rows only.
    Summary rows are rebuilt later from RUN rows.
    """
    _ensure_parent(csv_path)

    row = dict(row)
    row["summary_type"] = "RUN"

    if not csv_path.exists():
        df = pd.DataFrame([row])
        df = _reindex_with_tail(df, TIMING_RUN_COLS)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return

    df = pd.read_csv(csv_path)

    if "summary_type" not in df.columns:
        df["summary_type"] = "RUN"

    # keep only RUN rows in this file layer; summaries are rebuilt explicitly
    df = df[df["summary_type"].fillna("RUN") == "RUN"].copy()

    for c in row.keys():
        if c not in df.columns:
            df[c] = np.nan

    row_df = pd.DataFrame([row])
    row_df = _reindex_with_tail(row_df, df.columns)

    def _key(frame: pd.DataFrame) -> pd.Series:
        return frame[list(key_cols)].astype(str).agg("§".join, axis=1)

    if len(df) == 0:
        df = row_df.copy()
    else:
        k_df = _key(df)
        k_row = _key(row_df).iloc[0]
        hit = k_df == k_row

        if hit.any():
            idx = int(np.flatnonzero(hit.to_numpy())[0])
            for c in row_df.columns:
                val = row_df.at[0, c]
                if c in df.columns and df[c].dtype != object and isinstance(val, str):
                    df[c] = df[c].astype(object)
                df.at[idx, c] = val
        else:
            for c in row_df.columns:
                val = row_df.at[0, c]
                if c in df.columns and df[c].dtype != object and isinstance(val, str):
                    df[c] = df[c].astype(object)
            row_df = row_df.reindex(columns=df.columns, fill_value=np.nan)
            df = pd.concat([df, row_df], axis=0, ignore_index=True)

    df = _reindex_with_tail(df, TIMING_RUN_COLS)
    df.to_csv(csv_path, index=False, encoding="utf-8")


# =========================================================
# SUMMARY CSV (RUN / AVG / STD / blank divider)
# =========================================================

def rebuild_timings_runs_with_summary(
    csv_path: Path,
    *,
    group_cols: Optional[Sequence[str]] = None,
    value_cols: Optional[Sequence[str]] = None,
) -> None:
    """
    Rebuilds the file in block form:

      RUN
      RUN
      ...
      AVG
      STD
      <blank row>

    Summary is always computed only from RUN rows.
    """
    df = _read_csv_safe(csv_path)
    if df.empty:
        return

    if "summary_type" not in df.columns:
        df["summary_type"] = "RUN"

    df_run = df[df["summary_type"].fillna("RUN") == "RUN"].copy()
    if df_run.empty:
        return

    group_cols_use = [c for c in (group_cols or TIMING_SUMMARY_GROUP_COLS) if c in df_run.columns]
    value_cols_use = [c for c in (value_cols or TIMING_VALUE_COLS) if c in df_run.columns]

    # scientific ordering: deterministic by config, then seed
    sort_head = [c for c in ["dataset", "model", "use_exog", "H", "input_size", "seed"] if c in df_run.columns]
    if sort_head:
        df_run = df_run.sort_values(sort_head, kind="mergesort").reset_index(drop=True)

    blocks: List[pd.DataFrame] = []

    for _, g in df_run.groupby(group_cols_use, sort=False, dropna=False):
        g = g.sort_values([c for c in ["seed"] if c in g.columns], kind="mergesort").copy()
        blocks.append(g)

        avg_row = g.iloc[0].copy()
        avg_row["summary_type"] = "AVG"
        if "seed" in avg_row.index:
            avg_row["seed"] = ""
        if "run_name" in avg_row.index:
            avg_row["run_name"] = ""
        for c in value_cols_use:
            avg_row[c] = _summary_mean(g[c])

        std_row = g.iloc[0].copy()
        std_row["summary_type"] = "STD"
        if "seed" in std_row.index:
            std_row["seed"] = ""
        if "run_name" in std_row.index:
            std_row["run_name"] = ""
        for c in value_cols_use:
            std_row[c] = _summary_std(g[c])

        blocks.append(pd.DataFrame([avg_row]))
        blocks.append(pd.DataFrame([std_row]))
        blocks.append(pd.DataFrame([_blank_row_like(df_run.columns)]))

    df_out = pd.concat(blocks, axis=0, ignore_index=True)
    df_out = _reindex_with_tail(df_out, TIMING_RUN_COLS)
    df_out.to_csv(csv_path, index=False, encoding="utf-8")


# =========================================================
# ARTICLE-READY SUMMARY FILE (AVG only)
# =========================================================

def build_timings_summary_csv(
    runs_csv_path: Path,
    out_csv_path: Path,
    *,
    group_cols: Optional[Sequence[str]] = None,
    value_cols: Optional[Sequence[str]] = None,
) -> None:
    """
    Produces one clean AVG row per config, plus *_std columns.
    This file is better suited for article tables than the block-style RUN/AVG/STD file.
    """
    df = _read_csv_safe(runs_csv_path)
    if df.empty:
        return

    if "summary_type" not in df.columns:
        df["summary_type"] = "RUN"

    df_run = df[df["summary_type"].fillna("RUN") == "RUN"].copy()
    if df_run.empty:
        return

    group_cols_use = [c for c in (group_cols or TIMING_SUMMARY_GROUP_COLS) if c in df_run.columns]
    value_cols_use = [c for c in (value_cols or TIMING_VALUE_COLS) if c in df_run.columns]

    rows: List[Dict[str, Any]] = []
    for _, g in df_run.groupby(group_cols_use, sort=False, dropna=False):
        row: Dict[str, Any] = {c: g.iloc[0][c] for c in group_cols_use}
        for c in value_cols_use:
            row[c] = _summary_mean(g[c])
            row[f"{c}_std"] = _summary_std(g[c])
        row["n_seeds"] = int(len(g))
        rows.append(row)

    out = pd.DataFrame(rows)
    preferred = [
        "dataset", "model", "model_family", "use_exog", "exog_variant", "H", "L_factor",
        "input_size", "batch_size", "n_channels", "mode", "eval_mode", "step_size",
        "step_size_spec", "n_windows", "train_size", "val_size", "test_size",
        "split_mode", "split_source",
        "train_seconds", "train_seconds_std",
        "inf_seconds", "inf_seconds_std",
        "M_batch", "M_batch_std",
        "n_seeds",
    ]
    out = _reindex_with_tail(out, preferred)
    _ensure_parent(out_csv_path)
    out.to_csv(out_csv_path, index=False, encoding="utf-8")


# =========================================================
# COST-BENEFIT TABLE FOR ARTICLE
# =========================================================

def build_cost_benefit_csv(
    *,
    metrics_runs_csv: Path,
    timings_runs_csv: Path,
    out_csv_path: Path,
    metric_col: str = "MAE_pooled_norm",
) -> None:
    """
    Builds article-ready cost-benefit rows by pairing:
      use_exog = 0  vs  use_exog = 1

    Inputs:
      - metrics_runs_csv: RUN/AVG/STD style metrics file
      - timings_runs_csv: RUN/AVG/STD style timings file

    Uses AVG rows when available; otherwise computes means from RUN rows.
    """
    df_m = _read_csv_safe(metrics_runs_csv)
    df_t = _read_csv_safe(timings_runs_csv)
    if df_m.empty or df_t.empty:
        return

    if "summary_type" not in df_m.columns:
        df_m["summary_type"] = "RUN"
    if "summary_type" not in df_t.columns:
        df_t["summary_type"] = "RUN"

    # Prefer AVG rows; otherwise aggregate RUN rows to config-level means
    if (df_m["summary_type"] == "AVG").any():
        df_m_use = df_m[df_m["summary_type"] == "AVG"].copy()
    else:
        df_m_run = df_m[df_m["summary_type"].fillna("RUN") == "RUN"].copy()
        metric_group_cols = [c for c in [
            "dataset", "model", "use_exog", "exog_variant",
            "H", "L_factor", "input_size", "mode", "eval_mode",
            "step_size", "step_size_spec", "n_windows",
        ] if c in df_m_run.columns]
        df_m_use = df_m_run.groupby(metric_group_cols, dropna=False, as_index=False)[metric_col].mean()

    if (df_t["summary_type"] == "AVG").any():
        df_t_use = df_t[df_t["summary_type"] == "AVG"].copy()
    else:
        df_t_run = df_t[df_t["summary_type"].fillna("RUN") == "RUN"].copy()
        timing_group_cols = [c for c in TIMING_SUMMARY_GROUP_COLS if c in df_t_run.columns]
        value_cols_use = [c for c in ["train_seconds", "inf_seconds", "M_batch"] if c in df_t_run.columns]
        df_t_use = df_t_run.groupby(timing_group_cols, dropna=False, as_index=False)[value_cols_use].mean()

    join_keys = [
        "dataset", "model", "model_family", "use_exog", "exog_variant",
        "H", "L_factor", "input_size", "mode", "eval_mode",
        "step_size", "step_size_spec", "n_windows",
    ]
    join_keys = [c for c in join_keys if c in df_m_use.columns and c in df_t_use.columns]

    keep_metrics = join_keys + [metric_col]
    keep_timings = join_keys + ["train_seconds", "inf_seconds", "M_batch"]

    df_m_use = df_m_use[keep_metrics].copy()
    df_t_use = df_t_use[keep_timings].copy()

    df_mt = pd.merge(df_m_use, df_t_use, on=join_keys, how="inner")

    # Pair noX and X
    pair_keys = [
        "dataset", "model", "model_family", "H", "L_factor", "input_size",
        "mode", "eval_mode", "step_size", "step_size_spec", "n_windows",
    ]
    pair_keys = [c for c in pair_keys if c in df_mt.columns]

    left_noX = df_mt[df_mt["use_exog"] == 0].copy()
    left_X = df_mt[df_mt["use_exog"] == 1].copy()

    if "exog_variant" not in left_noX.columns:
        left_noX["exog_variant"] = None
    if "exog_variant" not in left_X.columns:
        left_X["exog_variant"] = None

    merged = pd.merge(
        left_noX,
        left_X,
        on=pair_keys,
        how="inner",
        suffixes=("_noX", "_X"),
    )

    rows: List[Dict[str, Any]] = []
    for _, r in merged.iterrows():
        mae_noX = _safe_float(r.get(f"{metric_col}_noX"))
        mae_X = _safe_float(r.get(f"{metric_col}_X"))
        tr_noX = _safe_float(r.get("train_seconds_noX"))
        tr_X = _safe_float(r.get("train_seconds_X"))
        inf_noX = _safe_float(r.get("inf_seconds_noX"))
        inf_X = _safe_float(r.get("inf_seconds_X"))
        mem_noX = _safe_float(r.get("M_batch_noX"))
        mem_X = _safe_float(r.get("M_batch_X"))

        row = {
            "dataset": r.get("dataset"),
            "model": r.get("model"),
            "model_family": r.get("model_family"),
            "H": r.get("H"),
            "L_factor": r.get("L_factor"),
            "input_size": r.get("input_size"),
            "mode": r.get("mode"),
            "eval_mode": r.get("eval_mode"),
            "step_size": r.get("step_size"),
            "step_size_spec": r.get("step_size_spec"),
            "n_windows": r.get("n_windows"),
            "exog_variant": r.get("exog_variant_X"),
            "MAE_norm_noX": mae_noX,
            "MAE_norm_X": mae_X,
            "train_seconds_noX": tr_noX,
            "train_seconds_X": tr_X,
            "inf_seconds_noX": inf_noX,
            "inf_seconds_X": inf_X,
            "M_batch_noX": mem_noX,
            "M_batch_X": mem_X,
            "delta_MAE_norm": (mae_noX - mae_X) if mae_noX is not None and mae_X is not None else None,
            "delta_T_train": _safe_div((tr_X - tr_noX) if tr_X is not None and tr_noX is not None else None, tr_noX or 0.0),
            "delta_T_inf": _safe_div((inf_X - inf_noX) if inf_X is not None and inf_noX is not None else None, inf_noX or 0.0),
            "delta_M_batch": _safe_div((mem_X - mem_noX) if mem_X is not None and mem_noX is not None else None, mem_noX or 0.0),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out = _reindex_with_tail(out, COST_BENEFIT_COLS)
    _ensure_parent(out_csv_path)
    out.to_csv(out_csv_path, index=False, encoding="utf-8")


# =========================================================
# BUILD FROM timing_meta.json / metrics.json
# =========================================================


def build_timing_row_from_timing_meta(
    timing_meta_path: Path,
    *,
    step_size_spec: Optional[int | str] = None,
) -> Dict[str, Any]:
    payload = load_json(timing_meta_path)
    return build_timing_row(
        meta=payload,
        train_seconds=float(payload.get("train_seconds", 0.0)),
        inf_seconds=float(payload.get("inf_seconds", 0.0)),
        batch_size=int(payload.get("batch_size", 0)),
        input_size=int(payload.get("input_size", 0)),
        n_channels=int(payload.get("n_channels", 0)),
        step_size_spec=payload.get("step_size_spec", step_size_spec),
        bytes_per_element=int(payload.get("bytes_per_element", 4)),
        insample_y_numel=int(payload.get("insample_y_numel", 0)),
        futr_exog_numel=int(payload.get("futr_exog_numel", 0)),
        hist_exog_numel=int(payload.get("hist_exog_numel", 0)),
        stat_exog_numel=int(payload.get("stat_exog_numel", 0)),
        has_futr_exog=int(payload.get("has_futr_exog", 0)),
        has_hist_exog=int(payload.get("has_hist_exog", 0)),
        has_stat_exog=int(payload.get("has_stat_exog", 0)),
    )


def build_timing_row_from_metrics_json(
    metrics_json_path: Path,
    *,
    train_seconds: float,
    inf_seconds: float,
    batch_size: int,
    input_size: int,
    n_channels: int,
    step_size_spec: Optional[int | str] = None,
) -> Dict[str, Any]:
    payload = load_json(metrics_json_path)
    meta = payload.get("meta", payload)
    return build_timing_row(
        meta=meta,
        train_seconds=train_seconds,
        inf_seconds=inf_seconds,
        batch_size=batch_size,
        input_size=input_size,
        n_channels=n_channels,
        step_size_spec=step_size_spec,
    )