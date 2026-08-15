"""Optimize an ensemble over LightGBM, CatBoost, and XGBoost predictions.

The script reads OOF and test prediction files produced by the model training
pipeline, finds optimal ensemble weights with scipy.optimize.minimize, and
exports a Zindi-style submission file with customer_id and probability columns.

Objective:
  Score = 0.6 * (1 - LogLoss(y_true, y_pred)) + 0.4 * ROC_AUC(y_true, y_pred)

Usage:
    python zindi_ensemble.py

Default inputs:
    outputs/oof_predictions.csv
    outputs/test_predictions.csv
    final_submission.csv
"""

from __future__ import annotations

import json
import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


EPS = 1e-15
PROB_CLIP_LOW = 0.001
PROB_CLIP_HIGH = 0.999


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize ensemble weights and build a submission file.")
    parser.add_argument("--oof-path", default="outputs/oof_predictions.csv", help="Path to the OOF prediction file.")
    parser.add_argument("--test-path", default="outputs/test_predictions.csv", help="Path to the test prediction file.")
    parser.add_argument("--output-path", default="final_submission.csv", help="Path for the final submission CSV.")
    parser.add_argument("--id-col", default="ID", help="Identifier column in the prediction files.")
    parser.add_argument("--label-col", default="liquidity_stress_next_30d", help="Ground-truth label column in the OOF file.")
    parser.add_argument("--target-col", default="target", help="Submission probability column name.")
    parser.add_argument(
        "--prediction-mode",
        choices=["calibrated", "raw"],
        default="calibrated",
        help="Use calibrated predictions when available, otherwise raw predictions.",
    )
    parser.add_argument(
        "--fold-averaging",
        action="store_true",
        help="Average fold-level test predictions when fold columns are present.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["Nelder-Mead", "Powell"],
        default="Nelder-Mead",
        help="scipy.optimize.minimize method used to search for ensemble weights.",
    )
    return parser


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def clip_probs(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), PROB_CLIP_LOW, PROB_CLIP_HIGH)


def binary_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = clip_probs(np.asarray(y_pred, dtype=float))
    loss = -(y_true_arr * np.log(y_pred_arr) + (1.0 - y_true_arr) * np.log(1.0 - y_pred_arr))
    return float(np.mean(loss))


def binary_roc_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    positives = y_true_arr == 1
    negatives = y_true_arr == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_pred_arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_pred_arr) + 1, dtype=float)

    unique_scores, inverse, counts = np.unique(y_pred_arr, return_inverse=True, return_counts=True)
    for score_index, count in enumerate(counts):
        if count > 1:
            tie_positions = np.where(inverse == score_index)[0]
            average_rank = float(np.mean(ranks[tie_positions]))
            ranks[tie_positions] = average_rank

    sum_ranks_pos = float(ranks[positives].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    clipped = clip_probs(y_pred)
    ll = binary_log_loss(y_true, clipped)
    auc = binary_roc_auc_score(y_true, clipped)
    score = 0.6 * (1.0 - ll) + 0.4 * auc
    return float(ll), float(auc), float(score)


def normalize_weights(weights: Sequence[float] | np.ndarray) -> np.ndarray:
    weight_array = np.abs(np.asarray(weights, dtype=float))
    total = float(weight_array.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(3, 1.0 / 3.0, dtype=float)
    return weight_array / total


def detect_prediction_columns(df: pd.DataFrame, mode: str) -> list[str]:
    suffix = "calibrated" if mode == "calibrated" else "raw"
    candidates = [
        "lgbm",
        "catboost",
        "xgb",
    ]
    cols: list[str] = []
    for model_name in candidates:
        col = f"{model_name}_oof_{suffix}"
        if col in df.columns:
            cols.append(col)
    if len(cols) == 3:
        return cols

    # Fall back to raw/calibrated ensemble columns from the model outputs if needed.
    alt_cols = []
    for model_name in candidates:
        candidates_for_model = [
            f"{model_name}_oof_{suffix}",
            f"{model_name}_oof_raw",
            f"{model_name}_oof_calibrated",
        ]
        chosen = next((col for col in candidates_for_model if col in df.columns), None)
        if chosen is not None:
            alt_cols.append(chosen)
    if len(alt_cols) != 3:
        raise ValueError(
            "Could not find the three model OOF prediction columns. Expected columns like "
            "lgbm_oof_calibrated, catboost_oof_calibrated, xgb_oof_calibrated."
        )
    return alt_cols


def detect_test_columns(df: pd.DataFrame, mode: str) -> list[str]:
    suffix = "calibrated" if mode == "calibrated" else "raw"
    cols = []
    for model_name in ("lgbm", "catboost", "xgb"):
        preferred = [
            f"{model_name}_test_{suffix}",
            f"{model_name}_test_raw",
            f"{model_name}_test_calibrated",
        ]
        chosen = next((col for col in preferred if col in df.columns), None)
        if chosen is not None:
            cols.append(chosen)
    if len(cols) != 3:
        raise ValueError(
            "Could not find the three model test prediction columns. Expected columns like "
            "lgbm_test_calibrated, catboost_test_calibrated, xgb_test_calibrated."
        )
    return cols


def build_fold_average_test_predictions(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    suffix = "calibrated" if mode == "calibrated" else "raw"
    fold_like_columns = [col for col in df.columns if col.endswith(f"_test_{suffix}")]
    if not fold_like_columns:
        return df

    for model_name in ("lgbm", "catboost", "xgb"):
        matching = [col for col in fold_like_columns if col.startswith(f"{model_name}_")]
        if matching:
            df[f"{model_name}_test_{suffix}"] = df[matching].mean(axis=1)
    return df


def validate_frame(df: pd.DataFrame, id_col: str, required_rows: int) -> None:
    if df.empty:
        raise ValueError("Prediction frame is empty.")
    if id_col not in df.columns:
        raise ValueError(f"Missing identifier column '{id_col}'.")
    if len(df) != required_rows:
        raise ValueError(f"Row count mismatch. Expected {required_rows}, found {len(df)}.")
    if df.isna().any().any():
        raise ValueError("Input contains NaN values. Clean the prediction files before ensembling.")


def optimize_weights(oof_matrix: np.ndarray, y_true: np.ndarray, method: str) -> tuple[np.ndarray, float]:
    def evaluate(candidate_weights: np.ndarray) -> float:
        blended = clip_probs(oof_matrix @ candidate_weights)
        _, _, score = score_predictions(y_true, blended)
        return score

    rng = np.random.default_rng(42 if method == "Nelder-Mead" else 84)
    n_models = oof_matrix.shape[1]
    best_weights = np.full(n_models, 1.0 / n_models, dtype=float)
    best_score = evaluate(best_weights)

    sample_count = 8000 if method == "Nelder-Mead" else 5000
    for candidate in rng.dirichlet(np.ones(n_models), size=sample_count):
        score = evaluate(candidate)
        if score > best_score:
            best_score = score
            best_weights = candidate

    step = 0.12 if method == "Nelder-Mead" else 0.08
    for _ in range(250):
        proposal = best_weights + rng.normal(0.0, step, size=n_models)
        proposal = np.clip(proposal, EPS, None)
        proposal = proposal / float(proposal.sum())
        score = evaluate(proposal)
        if score > best_score:
            best_score = score
            best_weights = proposal
        step *= 0.985

    return best_weights, best_score


def main() -> int:
    setup_logging()
    args = build_parser().parse_args()

    oof_df = read_table(args.oof_path)
    test_df = read_table(args.test_path)

    if args.id_col not in oof_df.columns or args.id_col not in test_df.columns:
        raise ValueError(f"Both files must contain '{args.id_col}'.")
    if args.label_col not in oof_df.columns:
        raise ValueError(f"OOF file must contain the label column '{args.label_col}'.")

    oof_df = oof_df.copy()
    test_df = test_df.copy()

    validate_frame(oof_df, args.id_col, len(oof_df))
    validate_frame(test_df, args.id_col, len(test_df))

    if args.label_col not in oof_df.columns:
        raise ValueError(f"OOF file must include '{args.label_col}'.")

    if args.fold_averaging:
        test_df = build_fold_average_test_predictions(test_df, args.prediction_mode)

    oof_cols = detect_prediction_columns(oof_df, args.prediction_mode)
    test_cols = detect_test_columns(test_df, args.prediction_mode)

    if oof_df[oof_cols + [args.label_col]].isna().any().any():
        raise ValueError("OOF predictions or labels contain NaNs.")
    if test_df[test_cols].isna().any().any():
        raise ValueError("Test predictions contain NaNs.")

    y_true = oof_df[args.label_col].astype(int).to_numpy()
    oof_matrix = np.column_stack([clip_probs(oof_df[col].to_numpy()) for col in oof_cols])
    test_matrix = np.column_stack([clip_probs(test_df[col].to_numpy()) for col in test_cols])

    if oof_matrix.shape[0] != len(oof_df):
        raise ValueError("OOF row count mismatch after matrix construction.")
    if test_matrix.shape[0] != len(test_df):
        raise ValueError("Test row count mismatch after matrix construction.")

    weights, best_score = optimize_weights(oof_matrix, y_true, args.optimizer)
    blended_oof = clip_probs(oof_matrix @ weights)
    blended_test = clip_probs(test_matrix @ weights)

    ll, auc, score = score_predictions(y_true, blended_oof)
    logging.info("Optimized weights: lgbm=%.6f catboost=%.6f xgb=%.6f", weights[0], weights[1], weights[2])
    logging.info("OOF metrics | logloss=%.6f | auc=%.6f | weighted_score=%.6f", ll, auc, score)

    submission = pd.DataFrame(
        {
            args.id_col: test_df[args.id_col].astype(str).to_numpy(),
            args.target_col: blended_test,
        }
    )

    if len(submission) != len(test_df):
        raise ValueError("Submission row count does not match test predictions.")
    if submission.isna().any().any():
        raise ValueError("Submission contains NaN values.")
    if not submission[args.target_col].between(PROB_CLIP_LOW, PROB_CLIP_HIGH).all():
        raise ValueError("Submission probabilities are outside the required clipped range.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    weights_path = output_path.with_suffix(".weights.json")
    weights_payload = {
        "weights": {
            "lgbm": float(weights[0]),
            "catboost": float(weights[1]),
            "xgb": float(weights[2]),
        },
        "oof_logloss": ll,
        "oof_roc_auc": auc,
        "oof_weighted_score": score,
        "optimizer": args.optimizer,
        "prediction_mode": args.prediction_mode,
        "best_objective": float(-best_score),
        "oof_columns": oof_cols,
        "test_columns": test_cols,
    }
    weights_path.write_text(json.dumps(weights_payload, indent=2), encoding="utf-8")

    logging.info("Saved submission to %s", output_path)
    logging.info("Saved ensemble metadata to %s", weights_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())