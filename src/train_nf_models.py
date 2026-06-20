from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


import metrics_export as mx
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import NBEATS, NBEATSx, NHITS

# =========================================================
# Warning Handeling
# =========================================================
import warnings

warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated.*",
    category=UserWarning,
)

# =========================================================
# CONFIG
# =========================================================

DEFAULT_VAL_SIZE: int = 24 * 183   # 6 months (hourly)
DEFAULT_TEST_SIZE: int = 24 * 365  # 1 year  (hourly)

# Iran
#DEFAULT_VAL_SIZE: int = 183   # 6 months
#DEFAULT_TEST_SIZE: int =365  # 1 year

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ECL":  {"freq": "h", "default_H": 96, "default_L_factor": 5},
    "IRAN": {"freq": "D", "default_H": 7,  "default_L_factor": 4},
    "PJM":  {"freq": "h", "default_H": 24, "default_L_factor": 5},
}

MODE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "debug": {
        "max_steps": 20,
        "batch_size": 16,
        "windows_batch_size": 8,
        "mlp_units": [[64, 64], [64, 64], [64, 64]],
    },
    "lite": {
        "max_steps": 600,
        "batch_size": 24,
        "windows_batch_size": 12,
        "mlp_units": [[128, 128], [128, 128], [128, 128]],
    },
    "full": {
        "max_steps": 1000,
        "batch_size": 4,
        "windows_batch_size": 4,
        "mlp_units": [[512, 512], [512, 512], [512, 512]],
    },
}


# =========================================================
# ARGS
# =========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, choices=list(DATASET_CONFIGS.keys()), required=True)
    p.add_argument("--mode", type=str, choices=list(MODE_CONFIGS.keys()), default="debug")
    p.add_argument("--model", type=str, choices=["NHITS", "NBEATS", "NBEATSx"], default="NHITS")
    p.add_argument("--use_exog", type=int, choices=[0, 1], default=0)
    p.add_argument("--exog_variant",type=str,choices=["standard", "min6"],
    default="standard",help="Which global exogenous file to load when --use_exog=1.")

    p.add_argument("--H", type=int, default=None)
    p.add_argument("--L_factor", type=int, default=None)

    p.add_argument("--eval_mode", type=str, choices=["fixed", "rolling"], default="rolling")
    p.add_argument("--step_size", type=int, default=None)
    p.add_argument("--n_windows", type=int, default=-1, help="-1 means: use val_size/test_size mode (n_windows=0).")

    p.add_argument("--val_size", type=int, default=None)
    p.add_argument("--test_size", type=int, default=None)

    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--out_root", type=str, default=None)

    p.add_argument("--run_date", type=str, default=None, help="YYYY-MM-DD. If omitted, uses today.")

    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--n_seeds", type=int, default=1)

    p.add_argument("--save_predictions", action="store_true", help="Save predictions parquet.")
    p.add_argument("--save_ckpt", action="store_true", help="Save model checkpoint.")
    p.add_argument("--save_scaler", action="store_true", help="Save per-series scaler stats (train-only).")

    return p.parse_args()


# =========================================================
# IO
# =========================================================

def dataset_data_root(dataset: str) -> Path:
    return PROJECT_ROOT / "datasets" / dataset.upper()


def dataset_out_root(dataset: str, run_date: str) -> Path:
    return PROJECT_ROOT / "experiments" / dataset.upper() / run_date


def y_long_path(data_root: Path, dataset: str) -> Path:
    return data_root / f"{dataset.upper()}__y__long__standard.parquet"


def x_global_path(data_root: Path, dataset: str, variant: str) -> Path:
    suffix = "standard" if variant == "standard" else "min6"
    return data_root / f"{dataset.upper()}__x__global__{suffix}.parquet"

def load_y_long(*, data_root: Path, dataset: str) -> pd.DataFrame:
    path = y_long_path(data_root, dataset)
    if not path.exists():
        raise FileNotFoundError(f"Missing y_long parquet: {path}")

    df = pd.read_parquet(path)
    req = {"unique_id", "ds", "y"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"y_long missing columns: {sorted(miss)}")

    df = df[["unique_id", "ds", "y"]]
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def load_x_global(*, data_root: Path, dataset: str, variant: str) -> pd.DataFrame:
    path = x_global_path(data_root, dataset, variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing x_global ({variant}) parquet: {path}")

    x = pd.read_parquet(path)

    if "ds" not in x.columns:
        raise ValueError("x_global must include 'ds' column")

    x["ds"] = pd.to_datetime(x["ds"])
    return x.sort_values("ds").reset_index(drop=True)

def attach_global_exog(y_long: pd.DataFrame, x_global: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    exog_cols = [c for c in x_global.columns if c != "ds"]
    if not exog_cols:
        raise ValueError("x_global has no exogenous columns (only ds).")

    out = y_long.merge(x_global, on="ds", how="left", copy=False)
    if out[exog_cols].isna().any().any():
        bad = int(out[exog_cols].isna().any(axis=1).sum())
        raise RuntimeError(f"exog merge produced NaN in {bad} rows; x_global must cover all ds in y_long.")
    return out, exog_cols


# =========================================================
# SPLIT
# =========================================================

def split_by_time(
    df: pd.DataFrame,
    *,
    val_size: int,
    test_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp, int]:
    ds_sorted = pd.Index(sorted(df["ds"].unique()))
    n_ts = int(len(ds_sorted))
    if val_size + test_size >= n_ts:
        raise ValueError(f"val_size + test_size must be < total timestamps ({n_ts}).")

    i_train_end = n_ts - (val_size + test_size)
    i_val_end = n_ts - test_size

    ds_train_end = pd.Timestamp(ds_sorted[i_train_end - 1])
    ds_val_end = pd.Timestamp(ds_sorted[i_val_end - 1])

    train_df = df[df["ds"] <= ds_train_end].copy()
    val_df = df[(df["ds"] > ds_train_end) & (df["ds"] <= ds_val_end)].copy()
    test_df = df[df["ds"] > ds_val_end].copy()

    return train_df, val_df, test_df, ds_train_end, ds_val_end, n_ts


# =========================================================
# SCALER (TRAIN ONLY)
# =========================================================

def compute_per_series_scaler_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    g = train_df.groupby("unique_id", sort=False, observed=False)["y"]
    stats = g.agg(["mean", "std"]).reset_index()
    stats.rename(columns={"mean": "y_mean", "std": "y_std"}, inplace=True)
    stats["y_std"] = stats["y_std"].astype("float32")
    stats["y_mean"] = stats["y_mean"].astype("float32")
    stats.loc[stats["y_std"] <= 0, "y_std"] = 1.0
    return stats


def write_scaler_artifacts(run_root: Path, stats_df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    stats_df.to_parquet(run_root / "scaler_stats.parquet", index=False)
    with open(run_root / "scaler_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# =========================================================
# MODEL
# =========================================================

def build_model(
    *,
    model_name: str,
    H: int,
    input_size: int,
    mlp_units: List[List[int]],
    max_steps: int,
    batch_size: int,
    windows_batch_size: int,
    seed: int,
    exog_cols: Optional[List[str]],
):

    use_gpu = torch.cuda.is_available()

    common = dict(
        h=H,
        input_size=input_size,
        loss=MAE(),
        max_steps=max_steps,
        learning_rate=1e-3,
        num_lr_decays=3,
        early_stop_patience_steps=-1,
        batch_size=batch_size,
        windows_batch_size=windows_batch_size,
        scaler_type="standard",
        random_seed=seed,
        dataloader_kwargs={"num_workers": 4, "pin_memory": True},
        num_sanity_val_steps=0,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision="16-mixed" if use_gpu else 32,
    )

    m = model_name.upper()

    if exog_cols and m in {"NHITS", "NBEATSX"}:
        common["futr_exog_list"] = list(exog_cols)

    if m == "NHITS":
        return NHITS(
            **common,
            mlp_units=mlp_units,
            stack_types=["identity", "identity", "identity"],
            n_blocks=[1, 1, 1],
            n_pool_kernel_size=[2, 2, 1],
            n_freq_downsample=[4, 2, 1],
            pooling_mode="MaxPool1d",
        )

    if m == "NBEATS":
        return NBEATS(
            **common,
            mlp_units=mlp_units,
            stack_types=["identity", "identity"],
            n_blocks=[1, 1],
            n_harmonics=0,
            n_polynomials=0,
        )

    if m == "NBEATSX":
        return NBEATSx(
            **common,
            mlp_units=mlp_units,
            stack_types=["identity", "identity"],
            n_blocks=[1, 1],
            n_harmonics=0,
            n_polynomials=0,
        )

    raise ValueError(f"Unsupported model: {model_name}")


# =========================================================
# EVAL
# =========================================================

def _select_pred_col(df: pd.DataFrame) -> str:
    base_cols = {"unique_id", "ds", "y", "cutoff"}
    pred_cols = [c for c in df.columns if c not in base_cols]
    if len(pred_cols) == 0:
        raise RuntimeError("No prediction column found.")
    if len(pred_cols) > 1:
        raise RuntimeError(f"Multiple prediction columns found: {pred_cols}")
    return pred_cols[0]


def test_eval(
    nf: NeuralForecast,
    *,
    eval_mode: str,
    fit_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
    full_df: Optional[pd.DataFrame],
    val_size: Optional[int],
    test_size: Optional[int],
    H: int,
    n_windows: int,
    step_size: int,
    fixed_cutoff: Optional[pd.Timestamp],
    verbose: bool = False,
) -> Tuple[pd.DataFrame, str]:

    if eval_mode == "fixed":
        if fit_df is None or test_df is None:
            raise ValueError("fixed eval requires fit_df and test_df")
        if test_size is None:
            raise ValueError("fixed eval requires test_size")
        if fixed_cutoff is None:
            raise ValueError("fixed eval requires fixed_cutoff")

        pred_df = nf.predict(df=fit_df)
        gt_df = test_df.groupby("unique_id", as_index=False).head(H)[["unique_id", "ds", "y"]]
        cv_df = gt_df.merge(pred_df, on=["unique_id", "ds"], how="left")

        pred_col = _select_pred_col(cv_df)
        if cv_df[pred_col].isna().any():
            miss = int(cv_df[pred_col].isna().sum())
            raise RuntimeError(f"predict() produced {miss} NaN forecasts; check ds/freq alignment and exog coverage.")

        cv_df["cutoff"] = fixed_cutoff
        return cv_df, pred_col

    if eval_mode == "rolling":
        if full_df is None:
            raise ValueError("rolling eval requires full_df")

        kwargs: Dict[str, Any] = {"df": full_df, "step_size": int(step_size), "refit": False}

        if n_windows > 0:
            # user explicitly controls windows -> do NOT touch val/test sizing here
            kwargs["n_windows"] = int(n_windows)
        else:
            # val/test sizing mode
            if val_size is None or test_size is None:
                raise ValueError("rolling val/test mode requires val_size and test_size")

            kwargs["n_windows"] = None
            kwargs["val_size"] = int(val_size)
            kwargs["test_size"] = int(test_size)


            H_local = int(H)  
            step_local = int(kwargs["step_size"])
            test_local = int(kwargs["test_size"])

            max_shift = test_local - H_local
            if max_shift <= 0:
                raise ValueError(f"Invalid rolling: test_size({test_local}) <= H({H_local})")

            n_windows_valid = (max_shift // step_local) + 1
            test_eff = H_local + step_local * (n_windows_valid - 1)

            if test_eff != test_local:
                dropped = test_local - test_eff
                print(
                    f"[ADJUST] rolling: keep H={H_local}, step={step_local}; "
                    f"test_size {test_local}->{test_eff} (drop_last={dropped}, n_windows={n_windows_valid})"
                )
                kwargs["test_size"] = int(test_eff)
                # kwargs["n_windows"] stays None (NeuralForecast will infer from val/test/step)

        if "verbose" in nf.cross_validation.__code__.co_varnames:
            kwargs["verbose"] = verbose

        cv_df = nf.cross_validation(**kwargs)
        pred_col = _select_pred_col(cv_df)
        return cv_df, pred_col

    raise ValueError(f"Unknown eval_mode: {eval_mode}")


# =========================================================
# RUN NAME
# =========================================================

def make_run_name(
    *,
    model: str,
    dataset: str,
    mode: str,
    H: int,
    L_factor: int,
    use_exog: int,
    exog_variant: str,
    eval_mode: str,
    step_size: int,
    n_windows: int,
    seed: int,
) -> str:
    return (
        f"{model.upper()}_{dataset.upper()}_{mode.upper()}"
        f"_H{H}_LF{L_factor}_EX{use_exog}"
        f"_XV{exog_variant.upper()}"
        f"_{eval_mode.upper()}_S{step_size}_W{n_windows}"
        f"_SEED{seed}"
    )


# =========================================================
# TIMING / COST META
# =========================================================

def _model_family(model_name: str) -> str:
    return "MLP" if str(model_name).upper() in {"NHITS", "NBEATS", "NBEATSX"} else "OTHER"


def _split_source(eval_mode: str, n_windows: int) -> str:
    if str(eval_mode).lower() == "rolling" and int(n_windows) > 0:
        return "n_windows"
    return "val_size_test_size"


def _fit_df_for_timing(*, train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([train_df, val_df], axis=0, ignore_index=True)


def _input_payload_meta(
    *,
    batch_size: int,
    input_size: int,
    H: int,
    exog_cols: Optional[List[str]],
    bytes_per_element: int = 4,
) -> Dict[str, Any]:
    futr_features = int(len(exog_cols) if exog_cols else 0)
    hist_features = 0
    stat_features = 0

    insample_y_numel = int(batch_size) * int(input_size)
    futr_exog_numel = int(batch_size) * int(input_size + H) * futr_features
    hist_exog_numel = int(batch_size) * int(input_size) * hist_features
    stat_exog_numel = int(batch_size) * stat_features

    total_numel = insample_y_numel + futr_exog_numel + hist_exog_numel + stat_exog_numel
    m_batch = int(total_numel * int(bytes_per_element))

    return {
        "batch_size": int(batch_size),
        "n_channels": int(1 + futr_features + hist_features + stat_features),
        "bytes_per_element": int(bytes_per_element),
        "has_futr_exog": int(futr_features > 0),
        "has_hist_exog": int(hist_features > 0),
        "has_stat_exog": int(stat_features > 0),
        "futr_features": futr_features,
        "hist_features": hist_features,
        "stat_features": stat_features,
        "insample_y_numel": insample_y_numel,
        "futr_exog_numel": futr_exog_numel,
        "hist_exog_numel": hist_exog_numel,
        "stat_exog_numel": stat_exog_numel,
        "M_batch": m_batch,
    }


def _write_timing_meta(run_root: Path, payload: Dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    with open(run_root / "timing_meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()

    ds_conf = DATASET_CONFIGS[args.dataset]
    mode_conf = MODE_CONFIGS[args.mode]
    run_date = str(args.run_date) if args.run_date else datetime.now().strftime("%Y-%m-%d")

    data_root = Path(args.data_root) if args.data_root else dataset_data_root(args.dataset)
    out_root = Path(args.out_root) if args.out_root else dataset_out_root(args.dataset, run_date)

    freq: str = str(ds_conf["freq"])

    H = int(args.H if args.H is not None else ds_conf["default_H"])
    L_factor = int(args.L_factor if args.L_factor is not None else ds_conf["default_L_factor"])
    input_size = int(L_factor * H)

    val_size = int(args.val_size if args.val_size is not None else DEFAULT_VAL_SIZE)
    test_size = int(args.test_size if args.test_size is not None else DEFAULT_TEST_SIZE)

    eval_mode = str(args.eval_mode)
    if eval_mode == "fixed":
        step_size = 0
        n_windows = 1
    else:
        step_size = int(args.step_size if args.step_size is not None else H)
        n_windows = int(args.n_windows) if int(args.n_windows) != -1 else 0

    full_df = load_y_long(data_root=data_root, dataset=args.dataset)

    exog_cols: Optional[List[str]] = None
    exog_variant = str(args.exog_variant)

    if int(args.use_exog) == 1:
        xg = load_x_global(data_root=data_root, dataset=args.dataset, variant=exog_variant)
        full_df, exog_cols = attach_global_exog(full_df, xg)
    else:
        exog_variant = "none" 

    min_ds = pd.Timestamp(full_df["ds"].min()).isoformat()
    max_ds = pd.Timestamp(full_df["ds"].max()).isoformat()
    rows_full = int(len(full_df))

    train_df, val_df, test_df, ds_train_end, ds_val_end, n_ts = split_by_time(
        full_df, val_size=val_size, test_size=test_size
    )

    scaler_df = compute_per_series_scaler_stats(train_df)

    for i in range(int(args.n_seeds)):
        seed = int(args.seed_base) + int(i)

        run_name = make_run_name(
            model=args.model,
            dataset=args.dataset,
            mode=args.mode,
            H=H,
            L_factor=L_factor,
            use_exog=int(args.use_exog),
            exog_variant=exog_variant,
            eval_mode=eval_mode,
            step_size=step_size,
            n_windows=n_windows,
            seed=seed,
        )

        run_root = out_root / f"H{H}" / run_name
        run_root.mkdir(parents=True, exist_ok=True)

        if args.save_scaler:
            scaler_meta = {
                "run_date": run_date,
                "dataset": args.dataset.upper(),
                "freq": freq,
                "scaler_type": "standard",
                "stats_scope": "train_only_per_series",
                "min_ds": min_ds,
                "max_ds": max_ds,
                "n_ts_full": rows_full,
                "n_series": int(train_df["unique_id"].nunique()),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
                "val_size": int(val_size),
                "test_size": int(test_size),
                "ds_train_end": ds_train_end.isoformat(),
                "ds_val_end": ds_val_end.isoformat(),
                "n_ts_total": int(n_ts),
                "use_exog": int(args.use_exog),
                "exog_variant": exog_variant,
                "exog_cols": exog_cols if exog_cols else [],
            }
            write_scaler_artifacts(run_root, stats_df=scaler_df, meta=scaler_meta)

        fit_df = _fit_df_for_timing(train_df=train_df, val_df=val_df)

        payload_meta = _input_payload_meta(
            batch_size=mode_conf["batch_size"],
            input_size=input_size,
            H=H,
            exog_cols=exog_cols,
            bytes_per_element=np.dtype(np.float32).itemsize,
        )

        train_seconds = 0.0
        cv_total_seconds: Optional[float] = None
        inf_seconds = 0.0

        if eval_mode == "rolling":
            timing_model = build_model(
                model_name=args.model,
                H=H,
                input_size=input_size,
                mlp_units=mode_conf["mlp_units"],
                max_steps=mode_conf["max_steps"],
                batch_size=mode_conf["batch_size"],
                windows_batch_size=mode_conf["windows_batch_size"],
                seed=seed,
                exog_cols=exog_cols,
            )
            timing_nf = NeuralForecast(models=[timing_model], freq=freq)
            t0 = time.perf_counter()
            timing_nf.fit(df=fit_df, val_size=val_size)
            train_seconds = float(time.perf_counter() - t0)
            del timing_nf
            del timing_model

            model = build_model(
                model_name=args.model,
                H=H,
                input_size=input_size,
                mlp_units=mode_conf["mlp_units"],
                max_steps=mode_conf["max_steps"],
                batch_size=mode_conf["batch_size"],
                windows_batch_size=mode_conf["windows_batch_size"],
                seed=seed,
                exog_cols=exog_cols,
            )
            nf = NeuralForecast(models=[model], freq=freq)

            t0 = time.perf_counter()
            cv_df, pred_col = test_eval(
                nf=nf,
                eval_mode=eval_mode,
                fit_df=fit_df,
                test_df=test_df,
                full_df=full_df,
                val_size=val_size,
                test_size=test_size,
                H=H,
                n_windows=n_windows,
                step_size=step_size,
                fixed_cutoff=ds_train_end,
                verbose=False,
            )
            cv_total_seconds = float(time.perf_counter() - t0)
            inf_seconds = max(0.0, cv_total_seconds - train_seconds)
        else:
            model = build_model(
                model_name=args.model,
                H=H,
                input_size=input_size,
                mlp_units=mode_conf["mlp_units"],
                max_steps=mode_conf["max_steps"],
                batch_size=mode_conf["batch_size"],
                windows_batch_size=mode_conf["windows_batch_size"],
                seed=seed,
                exog_cols=exog_cols,
            )

            nf = NeuralForecast(models=[model], freq=freq)
            t0 = time.perf_counter()
            nf.fit(df=fit_df, val_size=val_size)
            train_seconds = float(time.perf_counter() - t0)

            t0 = time.perf_counter()
            cv_df, pred_col = test_eval(
                nf=nf,
                eval_mode=eval_mode,
                fit_df=fit_df,
                test_df=test_df,
                full_df=full_df,
                val_size=val_size,
                test_size=test_size,
                H=H,
                n_windows=n_windows,
                step_size=step_size,
                fixed_cutoff=ds_train_end,
                verbose=False,
            )
            inf_seconds = float(time.perf_counter() - t0)

        if args.save_ckpt and eval_mode == "fixed":
            ckpt_path = run_root / "model"
            nf.save(str(ckpt_path), overwrite=True)

        if pred_col != "yhat":
            cv_df.rename(columns={pred_col: "yhat"}, inplace=True)

        keep_cols = ["unique_id", "ds", "y", "yhat"] + (["cutoff"] if "cutoff" in cv_df.columns else [])
        cv_df = cv_df[keep_cols]

        cv_df["y"] = cv_df["y"].astype("float32", copy=False)
        cv_df["yhat"] = cv_df["yhat"].astype("float32", copy=False)
        cv_df["ds"] = pd.to_datetime(cv_df["ds"])
        if "cutoff" in cv_df.columns:
            cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])

        meta: Dict[str, Any] = {
            "dataset": args.dataset,
            "run_date": run_date,
            "run_name": run_name,
            "model": args.model.upper(),
            "model_family": _model_family(args.model),
            "mode": args.mode,
            "eval_mode": eval_mode,
            "use_exog": int(args.use_exog),
            "exog_variant": exog_variant,
            "seed": int(seed),
            "H": int(H),
            "L_factor": int(L_factor),
            "input_size": int(input_size),
            "step_size": int(step_size),
            "n_windows": int(n_windows),
            "freq": freq,
            "min_ds": min_ds,
            "max_ds": max_ds,
            "n_rows": rows_full,
            "n_uids": int(train_df["unique_id"].nunique()),
            "ds_train_end": ds_train_end.isoformat(),
            "ds_val_end": ds_val_end.isoformat(),
            "val_size": int(val_size),
            "test_size": int(test_size),
        }

        timing_meta: Dict[str, Any] = {
            "dataset": args.dataset.upper(),
            "run_date": run_date,
            "run_name": run_name,
            "model": args.model.upper(),
            "model_family": _model_family(args.model),
            "mode": args.mode,
            "eval_mode": eval_mode,
            "use_exog": int(args.use_exog),
            "exog_variant": exog_variant,
            "seed": int(seed),
            "H": int(H),
            "L_factor": int(L_factor),
            "input_size": int(input_size),
            "step_size": int(step_size),
            "n_windows": int(n_windows),
            "step_size_spec": int(step_size) if eval_mode == "fixed" else int(step_size),
            "train_seconds": float(train_seconds),
            "inf_seconds": float(inf_seconds),
            "cv_total_seconds": float(cv_total_seconds) if cv_total_seconds is not None else None,
            "inference_mode": "predict_eval" if eval_mode == "fixed" else "rolling_cv_minus_fit",
            "train_size": int(train_df["ds"].nunique()),
            "val_size": int(val_size),
            "test_size": int(test_size),
            "split_mode": "chronological",
            "split_source": _split_source(eval_mode, n_windows),
        }
        timing_meta.update(payload_meta)
        _write_timing_meta(run_root, timing_meta)

        scaler_for_metrics = scaler_df if bool(args.save_scaler) else None
        payload = mx.build_metrics_payload(meta=meta, df_cv=cv_df, scaler_stats=scaler_for_metrics)
        mx.write_json(run_root / "metrics.json", payload)

        if args.save_predictions:
            cv_df.to_parquet(run_root / "predictions_cv.parquet", index=False)

    print("[DONE]")


if __name__ == "__main__":
    main()
