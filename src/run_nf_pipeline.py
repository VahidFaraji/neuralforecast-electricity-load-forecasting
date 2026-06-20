# python run_nf_pipeline.py --dataset IRAN --mode full --n_seeds 8 --seed_base 1 --use_step_map 0 --cleanup_losers 1 --save_ckpt 0 --save_scaler 1 --save_predictions_all 1



from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from metrics_export import upsert_metrics_runs_csv
from cost_export import (
    build_cost_benefit_csv,
    build_timing_row_from_timing_meta,
    build_timings_summary_csv,
    rebuild_timings_runs_with_summary,
    upsert_timings_runs_csv,
)

PY = sys.executable


# =========================================================
# CONFIG (edit defaults here)
# =========================================================

DEFAULT_DATASET = "PJM"
DEFAULT_MODE = "full"

MODELS = ["NBEATSx"]      #"NBEATS", "NBEATSx", "NHITS"
USE_EXOG_LIST = [1]                         #[0 , 1]
EXOG_VARIANTS = ["min6"]                     #["standard", "min6"]

H_LIST_FIXED: List[int] = []
H_LIST_ROLL: List[int] =  [24]  #[24, 48, 96, 192, 336, 720]

L_FACTOR_LIST = [3]

VAL_SIZE = 24 * 183
TEST_SIZE = 24 *365


SEED_BASE = 1
N_SEEDS = 8

STEP_SPEC_LIST: List[Any] = [24]
N_WINDOWS_LIST: List[Optional[int]] = [None]

STEP_MAP: Dict[int, int] = {
    24: 1,
    48: 2,
    96: 4,
    168: 24,
    192: 8,
    336: 24,
    720: 24,
}


# =========================================================
# ARGS
# =========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p.add_argument("--mode", type=str, default=DEFAULT_MODE)

    p.add_argument("--run_date", type=str, default=None, help="YYYY-MM-DD. If omitted, uses today's date.")
    p.add_argument("--seed_base", type=int, default=SEED_BASE)
    p.add_argument("--n_seeds", type=int, default=N_SEEDS)

    p.add_argument("--use_step_map", type=int, choices=[0, 1], default=0)

    p.add_argument("--cleanup_losers", type=int, choices=[0, 1], default=1)
    p.add_argument("--save_ckpt", type=int, choices=[0, 1], default=0)
    p.add_argument("--save_scaler", type=int, choices=[0, 1], default=1)
    p.add_argument("--save_predictions_all", type=int, choices=[0, 1], default=1)

    p.add_argument("--train_script", type=str, default=None, help="Path to train_nf_models.py")
    p.add_argument("--do_report", type=int, choices=[0, 1], default=1)
    p.add_argument("--report_script", type=str, default=None, help="Path to reporting_nf.py")
    p.add_argument("--report_views", type=str, default="stitched,cutoffmean", help="Comma-separated: stitched,cutoffmean")
    p.add_argument("--report_last_days", type=int, default=30)

    return p.parse_args()


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def exp_root(dataset: str) -> Path:
    return PROJECT_ROOT / "experiments" / dataset.upper()

def runs_csv_path(dataset: str) -> Path:
    return exp_root(dataset) / "metrics_runs.csv"

def timing_csv_path(dataset: str) -> Path:
    return exp_root(dataset) / "timings_runs.csv"

def timing_summary_csv_path(dataset: str) -> Path:
    return exp_root(dataset) / "timings_summary.csv"

def cost_benefit_csv_path(dataset: str) -> Path:
    return exp_root(dataset) / "cost_benefit.csv"


# =========================================================
# RUN NAME
# =========================================================

def make_run_name(
    *,
    model: str,
    dataset: str,
    mode: str,
    H: int,
    lf: int,
    use_exog: int,
    exog_variant: str,
    eval_mode: str,
    step_size: int,
    n_windows: int,
    seed: int,):
    return (
        f"{model.upper()}_{dataset.upper()}_{mode.upper()}"
        f"_H{H}_LF{lf}_EX{use_exog}"
        f"_XV{exog_variant.upper()}"
        f"_{eval_mode.upper()}_S{step_size}_W{n_windows}"
        f"_SEED{seed}"
        
    )


# =========================================================
# STEP SIZE
# =========================================================

def _resolve_step_spec(H: int, spec: Any) -> int:
    if isinstance(spec, int):
        return int(spec)
    if isinstance(spec, str):
        s = spec.strip().upper()
        if s == "H":
            return int(H)
        if s.startswith("H/"):
            denom = int(s.split("/", 1)[1])
            return max(1, int(H) // max(1, denom))
        return int(s)
    raise TypeError(f"Unsupported step spec: {spec} ({type(spec)})")


def choose_step_size(
    *,
    H: int,
    step_spec: Any,
    use_step_map: bool,
    test_size: int,
) -> Tuple[int, Optional[str]]:
    if use_step_map:
        if H not in STEP_MAP:
            raise KeyError(f"H={H} not in STEP_MAP. Add it or disable --use_step_map.")
        ss = int(STEP_MAP[H])
        msg = None
    else:
        ss = _resolve_step_spec(H, step_spec)
        msg = None

    max_shift = int(test_size) - int(H)
    if max_shift <= 0:
        ss2 = 1
        msg = f"step_size adjusted {ss} -> {ss2} (test_size - H <= 0)"
        return ss2, msg

    if ss > max_shift:
        ss2 = int(max_shift)
        msg = f"step_size adjusted {ss} -> {ss2} (step_size > test_size - H)"
        return ss2, msg

    return int(max(1, ss)), msg


# =========================================================
# IO HELPERS
# =========================================================

def _run(cmd: List[str]) -> None:
    print("\n>> " + " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True)


def _timed(cmd: List[str]) -> Tuple[str, str, float]:
    t0 = time.perf_counter()
    start = datetime.now().isoformat(timespec="seconds")
    _run(cmd)
    end = datetime.now().isoformat(timespec="seconds")
    return start, end, time.perf_counter() - t0


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _load_metrics_json(run_root: Path) -> Dict[str, Any]:
    p = run_root / "metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing metrics.json: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_run_date_dir(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir() and p.name[:4].isdigit()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.name, reverse=True)[0]


def locate_run_root(
    *,
    dataset: str,
    run_date: str,
    H: int,
    run_name: str,
) -> Path:
    base = exp_root(dataset)
    cand = base / run_date / f"H{H}" / run_name
    if cand.exists():
        return cand

    latest = _find_latest_run_date_dir(base)
    if latest is None:
        raise FileNotFoundError(f"Could not locate run directory under: {base}")

    cand2 = latest / f"H{H}" / run_name
    if cand2.exists():
        print(f"[WARN] run_date mismatch; using located date folder: {latest.name}")
        return cand2

    raise FileNotFoundError(f"Run folder not found for H={H}, run_name={run_name} under {base}")


def _cleanup_run(run_root: Path) -> None:
    preds = run_root / "predictions_cv.parquet"
    if preds.exists():
        preds.unlink()

    model_dir = run_root / "model"
    if model_dir.exists() and model_dir.is_dir():
        for p in sorted(model_dir.rglob("*"), key=lambda x: len(str(x)), reverse=True):
            if p.is_file():
                p.unlink()
        for p in sorted(model_dir.rglob("*"), key=lambda x: len(str(x)), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        try:
            model_dir.rmdir()
        except OSError:
            pass


# =========================================================
# METRICS EXTRACTORS (new schema + backward-compatible)
# =========================================================

def _payload_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    m = payload.get("meta")
    if isinstance(m, dict):
        return m
    return payload


def _score_mae_pooled_raw(payload: Dict[str, Any]) -> float:
    try:
        v = payload["metrics"]["raw"]["stitched"]["MAE_pooled"]
        return float(v)
    except Exception:
        pass
    raw = payload.get("metrics_raw", {}) or {}
    v = raw.get("MAE_pooled", raw.get("MAE"))
    return float(v) if v is not None else float("inf")


def _raw_blocks(payload: Dict[str, Any]) -> Dict[str, Any]:
    m = payload.get("metrics")
    if isinstance(m, dict) and isinstance(m.get("raw"), dict):
        return m["raw"]
    return {"stitched": payload.get("metrics_raw", {}) or {}, "cutoffmean": {}, "rolling": {}}


def _norm_blocks(payload: Dict[str, Any]) -> Dict[str, Any]:
    m = payload.get("metrics")
    if isinstance(m, dict) and isinstance(m.get("norm"), dict):
        return m["norm"]
    norm_old = payload.get("metrics_norm", {}) or {}
    return {"available": bool(norm_old), "stitched": norm_old, "cutoffmean": {}, "rolling": {}}


def _flatten_metrics_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta_src = _payload_meta(payload)
    meta = {k: meta_src.get(k) for k in [
        "run_date", "seed","run_name", "mode", "eval_mode", "dataset", "model",
        "use_exog","exog_variant", "H", "L_factor", "step_size", "n_windows",
        #"min_ds", "max_ds", "n_rows", "n_uids",
    ]}

    raw = _raw_blocks(payload)
    norm = _norm_blocks(payload)

    raw_st = raw.get("stitched") or {}
    raw_cm = raw.get("cutoffmean") or {}
    raw_roll = raw.get("rolling") or {}

    norm_available = bool(norm.get("available", False))
    norm_st = (norm.get("stitched") or {}) if norm_available else {}
    norm_cm = (norm.get("cutoffmean") or {}) if norm_available else {}
    norm_roll = (norm.get("rolling") or {}) if norm_available else {}

    row: Dict[str, Any] = {}
    row.update(meta)

    # ----------------- RAW / stitched (pooled) -----------------
    row["MAE_pooled_raw"] = raw_st.get("MAE_pooled")
    row["RMSE_pooled_raw"] = raw_st.get("RMSE_pooled")
    row["sMAPE_pooled_raw"] = raw_st.get("sMAPE_pooled")

    # ----------------- RAW / cutoffmean -----------------
    row["MAE_cutoff_mean_raw"] = raw_cm.get("MAE_cutoff_mean")
    row["RMSE_cutoff_mean_raw"] = raw_cm.get("RMSE_cutoff_mean")
    row["sMAPE_cutoff_mean_raw"] = raw_cm.get("sMAPE_cutoff_mean")

    # ----------------- RAW / rolling (cutoff series stats) -----------------
    row["MAE_cutoff_series_mean_raw"] = raw_roll.get("MAE_cutoff_series_mean")
    row["MAE_cutoff_series_std_raw"] = raw_roll.get("MAE_cutoff_series_std")
    row["n_cutoffs_raw"] = raw_roll.get("n_cutoffs")


    # ----------------- NORM / stitched (pooled) -----------------
    row["MAE_pooled_norm"] = norm_st.get("MAE_pooled") if norm_available else None
    row["RMSE_pooled_norm"] = norm_st.get("RMSE_pooled") if norm_available else None
    row["sMAPE_pooled_norm"] = norm_st.get("sMAPE_pooled") if norm_available else None
    # ----------------- NORM / cutoffmean -----------------
    row["MAE_cutoff_mean_norm"] = norm_cm.get("MAE_cutoff_mean") if norm_available else None
    row["RMSE_cutoff_mean_norm"] = norm_cm.get("RMSE_cutoff_mean") if norm_available else None
    row["sMAPE_cutoff_mean_norm"] = norm_cm.get("sMAPE_cutoff_mean") if norm_available else None
        # ----------------- NORM / rolling -----------------
    row["MAE_cutoff_series_mean_norm"] = norm_roll.get("MAE_cutoff_series_mean") if norm_available else None
    row["MAE_cutoff_series_std_norm"] = norm_roll.get("MAE_cutoff_series_std") if norm_available else None

    # optional diagnostics (if present in metrics_export payload)
    row["nmae_median_raw"] = raw_st.get("nmae_median")
    row["nmae_mean_raw"] = raw_st.get("nmae_mean")
    row["nmae_p90_raw"] = raw_st.get("nmae_p90")
    row["nmae_p95_raw"] = raw_st.get("nmae_p95")
    row["max_abs_err_max_raw"] = raw_st.get("max_abs_err_max")
    row["max_abs_err_p95_raw"] = raw_st.get("max_abs_err_p95")
    row["pareto_top_1pct_share_raw"] = raw_st.get("pareto_top_1pct_share")
    row["pareto_top_5pct_share_raw"] = raw_st.get("pareto_top_5pct_share")
    row["pareto_top_10pct_share_raw"] = raw_st.get("pareto_top_10pct_share")

    #NORM / stitched (pooled) -(if present in metrics_export payload)

    row["nmae_median_norm"] = norm_st.get("nmae_median") if norm_available else None
    row["nmae_mean_norm"] = norm_st.get("nmae_mean") if norm_available else None
    row["nmae_p90_norm"] = norm_st.get("nmae_p90") if norm_available else None
    row["nmae_p95_norm"] = norm_st.get("nmae_p95") if norm_available else None
    row["max_abs_err_max_norm"] = norm_st.get("max_abs_err_max") if norm_available else None
    row["max_abs_err_p95_norm"] = norm_st.get("max_abs_err_p95") if norm_available else None
    row["pareto_top_1pct_share_norm"] = norm_st.get("pareto_top_1pct_share") if norm_available else None
    row["pareto_top_5pct_share_norm"] = norm_st.get("pareto_top_5pct_share") if norm_available else None
    row["pareto_top_10pct_share_norm"] = norm_st.get("pareto_top_10pct_share") if norm_available else None


    row["n_cutoffs_norm"] = norm_roll.get("n_cutoffs") if norm_available else None
    # ----------------- NORM availability -----------------
    row["norm_available"] = int(norm_available)
    return row



def _ensure_keys(
    row: Dict[str, Any],
    *,
    dataset: str,
    run_date: str,
    cfg: "Config",
    mode: str,
    step_size: int,
    n_windows_eff: int,
    seed: int,
    run_name: str,
) -> Dict[str, Any]:
    row["dataset"] = row.get("dataset") or dataset
    row["run_date"] = row.get("run_date") or run_date
    row["run_name"] = row.get("run_name") or run_name

    row["model"] = row.get("model") or str(cfg.model).upper()
    row["mode"] = row.get("mode") or mode
    row["eval_mode"] = row.get("eval_mode") or str(cfg.eval_mode)

    row["use_exog"] = row.get("use_exog") if row.get("use_exog") is not None else int(cfg.use_exog)
    row["exog_variant"] = row.get("exog_variant") or str(cfg.exog_variant)
    row["H"] = row.get("H") if row.get("H") is not None else int(cfg.H)
    row["L_factor"] = row.get("L_factor") if row.get("L_factor") is not None else int(cfg.lf)
    row["step_size"] = row.get("step_size") if row.get("step_size") is not None else int(step_size)
    row["n_windows"] = row.get("n_windows") if row.get("n_windows") is not None else int(n_windows_eff)
    row["seed"] = row.get("seed") if row.get("seed") is not None else int(seed)

    if row.get("input_size") is None:
        row["input_size"] = int(row["H"]) * int(row["L_factor"])

    return row
# =========================================================
# PROGRESS / TIMING
# =========================================================

def _fmt_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _eta_seconds(elapsed: float, done_units: int, total_units: int) -> Optional[float]:
    if done_units <= 0 or total_units <= 0 or done_units > total_units:
        return None
    rate = elapsed / float(done_units)
    rem = total_units - done_units
    return rate * float(rem)


# =========================================================
# GRID
# =========================================================

@dataclass(frozen=True)
class Config:
    model: str
    use_exog: int
    exog_variant: str
    H: int
    lf: int
    eval_mode: str
    step_spec: Any
    n_windows: Optional[int]


def iter_configs() -> Iterable[Config]:
    for model in MODELS:
        for ex in USE_EXOG_LIST:
            for xv in (EXOG_VARIANTS if int(ex) == 1 else ["none"]):
                for lf in L_FACTOR_LIST:

                    use_exog_eff = int(ex)
                    xv_eff = str(xv)

                    if model.upper() == "NBEATS" and use_exog_eff == 1:
                        print("NBEATS does not support future exogenous variables, so continue with EX=0")
                        use_exog_eff = 0
                        xv_eff = "none"

                    # FIXED
                    for H in H_LIST_FIXED:
                        yield Config(
                            model=model,
                            use_exog=use_exog_eff,
                            exog_variant=xv_eff,
                            H=H,
                            lf=lf,
                            eval_mode="fixed",
                            step_spec=0,
                            n_windows=1,
                        )

                    # ROLLING
                    for H in H_LIST_ROLL:
                        for step_spec in STEP_SPEC_LIST:
                            for nw in N_WINDOWS_LIST:
                                yield Config(
                                    model=model,
                                    use_exog=use_exog_eff,
                                    exog_variant=xv_eff,
                                    H=H,
                                    lf=lf,
                                    eval_mode="rolling",
                                    step_spec=step_spec,
                                    n_windows=nw,
                                )
# =========================================================
# MAIN
# =========================================================

def main() -> None:
    args = parse_args()

    dataset = str(args.dataset).upper()
    mode = str(args.mode).lower()
    run_date = str(args.run_date) if args.run_date else datetime.now().strftime("%Y-%m-%d")

    train_script = Path(args.train_script) if args.train_script else (PROJECT_ROOT / "src" / "train_nf_models.py")
    if not train_script.exists():
        raise FileNotFoundError(f"Missing train script: {train_script}")

    report_script = Path(args.report_script) if args.report_script else (PROJECT_ROOT / "src" / "reporting_nf.py")
    if int(args.do_report) == 1 and not report_script.exists():
        raise FileNotFoundError(f"Missing report script: {report_script}")

    report_views = [v.strip() for v in str(args.report_views).split(",") if v.strip()]
    if not report_views:
        report_views = ["stitched"]


    key_cols = [
        "dataset", "run_date", "model", "use_exog", "exog_variant", "eval_mode", "mode",
        "H", "L_factor", "input_size", "step_size", "n_windows", "seed",
    ]
    timing_key_cols = [
        "dataset", "run_date", "model", "use_exog", "exog_variant", "eval_mode", "mode",
        "H", "L_factor", "input_size", "step_size", "n_windows", "seed", "run_name",
    ]

    cfgs = list(iter_configs())
    n_cfg_total = len(cfgs)
    n_seeds = int(args.n_seeds)
    total_units = n_cfg_total * max(1, n_seeds)

    t_pipeline0 = time.perf_counter()
    done_units = 0

    print("=================================================================")
    print(f"[PIPELINE] dataset={dataset} | mode={mode} | run_date={run_date}")
    print(f"[GRID] configs={n_cfg_total} | seeds_per_config={n_seeds} | total_units={total_units}")
    print("=================================================================")

    for cfg_idx, cfg in enumerate(cfgs, start=1):
        step_size, warn = (0, None)
        n_windows_eff = 1

        if cfg.eval_mode == "rolling":
            step_size, warn = choose_step_size(
                H=cfg.H,
                step_spec=cfg.step_spec,
                use_step_map=bool(int(args.use_step_map)),
                test_size=int(TEST_SIZE),
            )
            n_windows_eff = int(cfg.n_windows) if cfg.n_windows is not None else 0
        else:
            step_size = 0
            n_windows_eff = 1

        if warn:
            print(f"[WARNING] {warn} | dataset={dataset} H={cfg.H} eval=ROLLING step_spec={cfg.step_spec}")

        cfg_label = (
            f"model={cfg.model} ex={cfg.use_exog} "
            f"eval={cfg.eval_mode} H={cfg.H} lf={cfg.lf} "
            f"step={step_size} (spec={cfg.step_spec}) "
            f"nw={n_windows_eff}"
        )

        elapsed = time.perf_counter() - t_pipeline0
        eta = _eta_seconds(elapsed, done_units, total_units)
        print("-----------------------------------------------------------------")
        print(f"[STAGE] {cfg_idx}/{n_cfg_total} | {cfg_label}")
        print(f"[TIME]  elapsed={_fmt_hms(elapsed)} | eta={_fmt_hms(eta) if eta is not None else '??:??'}")
        print("-----------------------------------------------------------------")

        best_seed: Optional[int] = None
        best_score: float = float("inf")
        run_roots: List[Tuple[int, Path, float]] = []

        for j in range(n_seeds):
            seed = int(args.seed_base) + int(j)
            seed_k = j + 1

            run_name = make_run_name(
                model=cfg.model,
                dataset=dataset,
                mode=mode,
                H=cfg.H,
                lf=cfg.lf,
                use_exog=cfg.use_exog,
                exog_variant=cfg.exog_variant,
                eval_mode=cfg.eval_mode,
                step_size=step_size,
                n_windows=n_windows_eff,
                seed=seed,
            )

            cmd = [
                PY, str(train_script),
                "--dataset", dataset,
                "--mode", mode,
                "--model", cfg.model,
                "--use_exog", str(cfg.use_exog),
                "--H", str(cfg.H),
                "--L_factor", str(cfg.lf),
                "--eval_mode", cfg.eval_mode,
                "--seed_base", str(seed),
                "--n_seeds", "1",
            ]
            if int(cfg.use_exog) == 1:
                cmd += ["--exog_variant", str(cfg.exog_variant)]
            if cfg.eval_mode == "rolling":
                cmd += ["--step_size", str(step_size)]
                if cfg.n_windows is None:
                    cmd += ["--val_size", str(int(VAL_SIZE)), "--test_size", str(int(TEST_SIZE))]
                else:
                    cmd += ["--n_windows", str(int(cfg.n_windows))]
            else:
                cmd += ["--val_size", str(int(VAL_SIZE)), "--test_size", str(int(TEST_SIZE))]

            if int(args.save_predictions_all) == 1:
                cmd += ["--save_predictions"]
            if int(args.save_scaler) == 1:
                cmd += ["--save_scaler"]
            if int(args.save_ckpt) == 1:
                cmd += ["--save_ckpt"]

            cmd_with_date = cmd + ["--run_date", run_date]

            elapsed = time.perf_counter() - t_pipeline0
            eta = _eta_seconds(elapsed, done_units, total_units)
            print(f"[RUN]   seed {seed_k}/{n_seeds} (SEED={seed}) | stage {cfg_idx}/{n_cfg_total} | elapsed={_fmt_hms(elapsed)} | eta={_fmt_hms(eta) if eta is not None else '??:??'}")

            start, end, sec = ("", "", 0.0)
            try:
                start, end, sec = _timed(cmd_with_date)
            except subprocess.CalledProcessError:
                start, end, sec = _timed(cmd)

            run_root = locate_run_root(dataset=dataset, run_date=run_date, H=cfg.H, run_name=run_name)

            payload = _load_metrics_json(run_root)
            row = _flatten_metrics_row(payload)

            row = _ensure_keys(
                row,
                dataset=dataset,
                run_date=run_date,
                cfg=cfg,
                mode=mode,
                step_size=step_size,
                n_windows_eff=n_windows_eff,
                seed=seed,
                run_name=run_name,
            )

            row["step_size_spec"] = cfg.step_spec
            row["step_size_warn"] = warn or ""

            upsert_metrics_runs_csv(runs_csv_path(dataset), row=row, key_cols=key_cols)

            score = float(_score_mae_pooled_raw(payload))
            run_roots.append((seed, run_root, score))

            if score < best_score:
                best_score = score
                best_seed = seed

            timing_meta_path = run_root / "timing_meta.json"
            if not timing_meta_path.exists():
                raise FileNotFoundError(f"Missing timing_meta.json: {timing_meta_path}")

            timing_row = build_timing_row_from_timing_meta(
                timing_meta_path,
                step_size_spec=cfg.step_spec,
            )
            timing_row["run_date"] = timing_row.get("run_date") or run_date
            timing_row["run_name"] = timing_row.get("run_name") or run_name
            timing_row["dataset"] = timing_row.get("dataset") or dataset
            timing_row["model"] = timing_row.get("model") or str(cfg.model).upper()
            timing_row["mode"] = timing_row.get("mode") or mode
            timing_row["eval_mode"] = timing_row.get("eval_mode") or str(cfg.eval_mode)
            timing_row["use_exog"] = timing_row.get("use_exog") if timing_row.get("use_exog") is not None else int(cfg.use_exog)
            timing_row["exog_variant"] = timing_row.get("exog_variant") or str(cfg.exog_variant)
            timing_row["H"] = timing_row.get("H") if timing_row.get("H") is not None else int(cfg.H)
            timing_row["L_factor"] = timing_row.get("L_factor") if timing_row.get("L_factor") is not None else int(cfg.lf)
            timing_row["input_size"] = timing_row.get("input_size") if timing_row.get("input_size") is not None else int(cfg.H) * int(cfg.lf)
            timing_row["step_size"] = timing_row.get("step_size") if timing_row.get("step_size") is not None else int(step_size)
            timing_row["n_windows"] = timing_row.get("n_windows") if timing_row.get("n_windows") is not None else int(n_windows_eff)
            timing_row["seed"] = timing_row.get("seed") if timing_row.get("seed") is not None else int(seed)
            timing_row["step_size_spec"] = cfg.step_spec

            upsert_timings_runs_csv(
                timing_csv_path(dataset),
                row=timing_row,
                key_cols=timing_key_cols,
            )

            done_units += 1
            elapsed = time.perf_counter() - t_pipeline0
            eta = _eta_seconds(elapsed, done_units, total_units)
            print(f"[METRICS] seed={seed} MAE_pooled_raw={score:.6f} | sec={sec:.1f} | eta={_fmt_hms(eta) if eta is not None else '??:??'}")
            print(f"[PATH]   {run_root}")

        if best_seed is None:
            raise RuntimeError("best_seed not selected")

        best_root: Optional[Path] = None
        for s, rr, sc in run_roots:
            if s == best_seed:
                best_root = rr
                break
        if best_root is None:
            raise RuntimeError("best_run_root not found")

        print(f"[BEST]  {cfg_label} -> SEED={best_seed} (MAE_pooled_raw={best_score:.6f})")

        if int(args.do_report) == 1:
            if not (best_root / "scaler_stats.parquet").exists():
                print(f"[REPORT] skipped (missing scaler_stats.parquet) | {best_root}")
            else:
                for v in report_views:
                    if v not in ("stitched", "cutoffmean"):
                        raise ValueError(f"Invalid report view: {v}")
                    rcmd = [
                        PY, str(report_script),
                        "--run_root", str(best_root),
                        "--view", str(v),
                        "--last_days", str(int(args.report_last_days)),
                    ]
                    print(f"[REPORT] view={v} | seed={best_seed} | {best_root}")
                    _run(rcmd)

        if int(args.cleanup_losers) == 1:
            for seed, run_root, score in run_roots:
                if seed == best_seed:
                    continue
                _cleanup_run(run_root)
                print(f"[CLEAN] removed heavy artifacts | seed={seed} score={score:.6f} | {run_root}")

    rebuild_timings_runs_with_summary(timing_csv_path(dataset))
    build_timings_summary_csv(
        timing_csv_path(dataset),
        timing_summary_csv_path(dataset),
    )
    build_cost_benefit_csv(
        metrics_runs_csv=runs_csv_path(dataset),
        timings_runs_csv=timing_csv_path(dataset),
        out_csv_path=cost_benefit_csv_path(dataset),
        metric_col="MAE_pooled_norm",
    )

    elapsed = time.perf_counter() - t_pipeline0
    print("=================================================================")
    print(f"[DONE] total_elapsed={_fmt_hms(elapsed)} | dataset={dataset} | run_date={run_date}")
    print("=================================================================")


if __name__ == "__main__":
    main()
