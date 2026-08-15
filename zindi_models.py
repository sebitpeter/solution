"""OOF-calibrated, three-model ensemble for Zindi mobile-money stress.

The supplied competition data contains six *monthly* snapshots (m1...m6),
not individual transaction timestamps.  Consequently, 30/90/180-day windows
are exact monthly aggregations; 7/14-day features are emitted only when a
daily/transaction-level version of the data supplies date columns.  This is
intentional: manufacturing 14-day values from a monthly aggregate leaks an
unsupported assumption into the model.

Example
-------
python zindi_models.py --train-path Train.csv --test-path Test.csv

Outputs include outputs/oof_preds.npy, outputs/test_preds.npy (raw model
prediction dictionaries for resuming), and outputs/fixed_submission.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
except ImportError as exc:  # clear failure before a long feature build
    raise ImportError("Install lightgbm, catboost and xgboost to run this pipeline.") from exc

EPS, CLIP_LOW, CLIP_HIGH = 1e-5, 0.005, 0.995
MODEL_NAMES = ("lightgbm", "catboost", "xgboost")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrated LGBM/CatBoost/XGBoost Zindi ensemble")
    p.add_argument("--train-path", default="Train.csv")
    p.add_argument("--test-path", default="Test.csv")
    p.add_argument("--sample-submission-path", default="SampleSubmission.csv")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--id-col", default="ID")
    p.add_argument("--label-col", default=None, help="Defaults to Target, then a detected binary train-only column.")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-seeds", default="42,2026,777", help="Comma-separated seeds for LGBM/XGBoost bagging; CatBoost uses the first.")
    p.add_argument("--force-retrain", action="store_true", help="Ignore valid raw-prediction checkpoints.")
    p.add_argument("--advanced-fold-features", action="store_true", help="Enable OOF target/group encodings; use only with genuinely high-cardinality categoricals.")
    return p.parse_args()


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), CLIP_LOW, CLIP_HIGH)


def weighted_score(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    p = clip(p)
    ll = float(log_loss(y, p, labels=[0, 1]))
    auc = float(roc_auc_score(y, p))
    return ll, auc, 0.60 * (1.0 - ll) + 0.40 * auc


def find_label(train: pd.DataFrame, test: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in train:
            raise ValueError(f"Requested label column {requested!r} is not in Train.csv")
        return requested
    if "Target" in train:
        return "Target"
    # The competition's actual file uses this name; other binary train-only
    # fields are a safe fallback for a renamed release.
    train_only = [c for c in train if c not in test and c != "ID"]
    for c in ["liquidity_stress_next_30d", *train_only]:
        if c in train and train[c].dropna().nunique() == 2:
            return c
    raise ValueError("Could not detect a binary target. Supply --label-col.")


class MobileMoneyFeatures:
    """Feature builder that uses only columns that actually exist.

    m6 is treated as most recent.  With snapshot data, 30d=m6, 90d=m4:m6,
    and 180d=m1:m6.  If daily dates exist, 7d/14d RFM is additionally built.
    """

    PRODUCTS = ("paybill", "merchantpay", "transfer_from_bank", "mm_send", "received", "deposit", "withdraw")
    MONTH_WINDOWS = {"30d": (6,), "90d": (4, 5, 6), "180d": (1, 2, 3, 4, 5, 6)}

    def __init__(self, frame: pd.DataFrame, id_col: str):
        self.df, self.id_col = frame.copy(), id_col

    def sumcols(self, cols: list[str]) -> pd.Series:
        present = [c for c in cols if c in self.df]
        return self.df[present].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) if present else pd.Series(0.0, index=self.df.index)

    def monthly(self, product: str, suffix: str) -> pd.DataFrame:
        return pd.DataFrame({m: self.sumcols([f"m{m}_{product}_{suffix}"]) for m in range(1, 7)})

    def monthly_rfm(self, out: pd.DataFrame) -> None:
        for window, months in self.MONTH_WINDOWS.items():
            for product in self.PRODUCTS:
                value = self.sumcols([f"m{m}_{product}_total_value" for m in months])
                freq = self.sumcols([f"m{m}_{product}_volume" for m in months])
                prefix = f"rfm_{window}_{product}"
                out[f"{prefix}_monetary"] = value
                out[f"{prefix}_frequency"] = freq
                out[f"{prefix}_avg_value"] = value / (freq + EPS)
                # Recency in observed months: 0 is activity in m6, 6 is none.
                active = np.zeros(len(self.df), dtype=bool)
                recency = np.full(len(self.df), 6.0)
                for m in range(6, 0, -1):
                    active |= self.sumcols([f"m{m}_{product}_volume"]).to_numpy() > 0
                    recency[(recency == 6) & active] = 6 - m
                out[f"{prefix}_recency_months"] = recency

    def stress_features(self, out: pd.DataFrame) -> None:
        values = {p: self.monthly(p, "total_value") for p in self.PRODUCTS}
        volumes = {p: self.monthly(p, "volume") for p in self.PRODUCTS}
        spend = sum((values[p] for p in ("paybill", "merchantpay", "mm_send")), pd.DataFrame(0.0, index=self.df.index, columns=range(1, 7)))
        cash_in, cash_out = values["deposit"] + values["received"], values["withdraw"] + values["mm_send"]
        out["stress_spend_m6"] = spend[6]
        out["stress_spend_mom_ratio"] = spend[6] / (spend[5] + EPS)
        out["stress_spend_90d_share"] = spend[[4, 5, 6]].sum(axis=1) / (spend.sum(axis=1) + EPS)
        # Exact name required by brief where daily data is available; for six
        # monthly snapshots this is deliberately a recent-month / 90d proxy.
        out["stress_liquidity_velocity_14d_proxy"] = spend[6] / (spend[[4, 5, 6]].sum(axis=1) + EPS)
        airtime = self._optional_monthly(("airtime", "topup", "air_time"), "total_value")
        if airtime is not None:
            out["stress_airtime_spend_drop_14d_proxy"] = airtime[6] / (spend[6] + EPS)
        out["stress_cash_in_out_ratio_90d"] = cash_in[[4, 5, 6]].sum(axis=1) / (cash_out[[4, 5, 6]].sum(axis=1) + EPS)
        out["stress_net_cash_flow_90d"] = cash_in[[4, 5, 6]].sum(axis=1) - cash_out[[4, 5, 6]].sum(axis=1)
        out["stress_cash_out_mom_ratio"] = cash_out[6] / (cash_out[5] + EPS)
        out["stress_cash_in_mom_ratio"] = cash_in[6] / (cash_in[5] + EPS)
        loan = values["transfer_from_bank"]
        out["stress_credit_inflow_spike"] = loan[6] / (loan[[3, 4, 5]].mean(axis=1) + EPS)
        total_volume = sum(volumes.values())
        out["stress_transaction_frequency_mom_ratio"] = total_volume[6] / (total_volume[5] + EPS)
        # Stable temporal-shape features.  These are compact summaries of the
        # six observed snapshots and generalise better than wide encodings.
        time_x = np.arange(1, 7, dtype=float)
        centred_x = time_x - time_x.mean()
        for product in self.PRODUCTS:
            series = values[product]
            volume = volumes[product]
            tag = f"shape_{product}"
            out[f"{tag}_value_last3_first3_ratio"] = series[[4, 5, 6]].sum(axis=1) / (series[[1, 2, 3]].sum(axis=1) + EPS)
            out[f"{tag}_volume_last3_first3_ratio"] = volume[[4, 5, 6]].sum(axis=1) / (volume[[1, 2, 3]].sum(axis=1) + EPS)
            out[f"{tag}_active_months"] = (volume > 0).sum(axis=1)
            out[f"{tag}_value_cv"] = series.std(axis=1) / (series.mean(axis=1).abs() + EPS)
            out[f"{tag}_value_trend"] = (series.to_numpy() @ centred_x) / float((centred_x ** 2).sum())
        out["shape_total_activity_last3_first3_ratio"] = total_volume[[4, 5, 6]].sum(axis=1) / (total_volume[[1, 2, 3]].sum(axis=1) + EPS)
        balance_cols = [f"m{i}_daily_avg_bal" for i in range(1, 7) if f"m{i}_daily_avg_bal" in self.df]
        if balance_cols:
            balance = self.df[balance_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            out["stress_balance_drawdown"] = (balance.iloc[:, -2] - balance.iloc[:, -1]) / (balance.iloc[:, -2].abs() + EPS)
            out["stress_balance_volatility"] = balance.std(axis=1)

        # Transaction-level optional columns. These are never fabricated from
        # monthly fields that do not encode time/status.
        text_cols = [c for c in self.df if self.df[c].dtype == "object"]
        status = next((c for c in text_cols if re.search(r"status|result", c, re.I)), None)
        hour = next((c for c in self.df if re.search(r"(^|_)hour$|transaction_hour", c, re.I)), None)
        if status:
            bad = self.df[status].astype(str).str.contains("fail|revers", case=False, na=False).astype(float)
            out["stress_failure_reversal_frequency"] = bad
            out["stress_failure_reversal_ratio"] = bad
        if hour:
            out["stress_night_transaction_ratio"] = pd.to_numeric(self.df[hour], errors="coerce").between(0, 5).fillna(False).astype(float)

    def _optional_monthly(self, alternatives: tuple[str, ...], suffix: str) -> pd.DataFrame | None:
        for product in alternatives:
            if any(f"m{i}_{product}_{suffix}" in self.df for i in range(1, 7)):
                return self.monthly(product, suffix)
        return None

    def build(self) -> pd.DataFrame:
        out = self.df.drop(columns=[self.id_col], errors="ignore").copy()
        self.monthly_rfm(out)
        self.stress_features(out)
        out = out.replace([np.inf, -np.inf], np.nan)
        return out.loc[:, ~out.columns.duplicated()]


class FoldSafeFeatureEngineer:
    """Supervised encodings and group features fitted *only* on a fold's train rows."""

    def __init__(self, seed: int, max_categoricals: int = 8, max_numerics: int = 6, smoothing: float = 25.0):
        self.seed, self.max_categoricals, self.max_numerics, self.smoothing = seed, max_categoricals, max_numerics, smoothing

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "FoldSafeFeatureEngineer":
        cat_candidates = x.select_dtypes(include=["object", "category"]).columns.tolist()
        self.cats = sorted(cat_candidates, key=lambda c: x[c].nunique(dropna=False), reverse=True)[:self.max_categoricals]
        numerics = x.select_dtypes(include=np.number).columns.tolist()
        # Pick interactions using the fold's training labels only.
        corr = x[numerics].apply(lambda s: pd.to_numeric(s, errors="coerce").corr(pd.Series(y, index=x.index))).abs()
        self.nums = corr.nlargest(self.max_numerics).index.tolist()
        self.global_mean = float(np.mean(y))
        work = x.copy(); work["__y__"] = y
        self.target_maps, self.frequency_maps, self.group_maps = {}, {}, {}
        for cat in self.cats:
            key = x[cat].astype("string").fillna("__MISSING__")
            stat = pd.DataFrame({"key": key, "y": y}).groupby("key", observed=True)["y"].agg(["sum", "count"])
            self.target_maps[cat] = ((stat["sum"] + self.smoothing * self.global_mean) / (stat["count"] + self.smoothing)).to_dict()
            self.frequency_maps[cat] = (key.value_counts(dropna=False) / len(x)).to_dict()
            for num in self.nums:
                tmp = pd.DataFrame({"key": key, "value": pd.to_numeric(x[num], errors="coerce")})
                self.group_maps[(cat, num)] = tmp.groupby("key", observed=True)["value"].agg(["mean", "std", "min", "max", "skew"])
        return self

    def transform(self, x: pd.DataFrame, target_maps: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
        out = x.copy()
        maps = self.target_maps if target_maps is None else target_maps
        for cat in self.cats:
            key = x[cat].astype("string").fillna("__MISSING__")
            out[f"{cat}__te"] = key.map(maps[cat]).fillna(self.global_mean).astype(float)
            out[f"{cat}__freq"] = key.map(self.frequency_maps[cat]).fillna(0.0).astype(float)
            for num in self.nums:
                agg = self.group_maps[(cat, num)]
                for stat in ("mean", "std", "min", "max", "skew"):
                    out[f"{cat}__{num}__{stat}"] = key.map(agg[stat]).fillna(0.0)
        # Top correlated numerical-variable interactions, protected from zero denominators.
        for i, left in enumerate(self.nums):
            a = pd.to_numeric(x[left], errors="coerce").fillna(0.0)
            for right in self.nums[i + 1:]:
                b = pd.to_numeric(x[right], errors="coerce").fillna(0.0)
                tag = f"inter__{left}__{right}"
                out[f"{tag}__ratio"] = a / (b.abs() + EPS)
                out[f"{tag}__diff"] = a - b
                out[f"{tag}__product"] = a * b
        return out.replace([np.inf, -np.inf], np.nan)

    def fit_transform_oof(self, x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """OOF target encodings for fold-training rows; maps for validation/test use all fold-training rows."""
        self.fit(x, y)
        oof_maps: dict[str, dict[str, float]] = {cat: {} for cat in self.cats}
        encoded = pd.DataFrame(index=x.index)
        inner = StratifiedKFold(4, shuffle=True, random_state=self.seed)
        for cat in self.cats:
            encoded[f"{cat}__te"] = np.nan
        for tr, va in inner.split(x, y):
            for cat in self.cats:
                key_tr = x.iloc[tr][cat].astype("string").fillna("__MISSING__")
                stats = pd.DataFrame({"key": key_tr, "y": y[tr]}).groupby("key", observed=True)["y"].agg(["sum", "count"])
                mapping = ((stats["sum"] + self.smoothing * np.mean(y[tr])) / (stats["count"] + self.smoothing)).to_dict()
                encoded.iloc[va, encoded.columns.get_loc(f"{cat}__te")] = x.iloc[va][cat].astype("string").fillna("__MISSING__").map(mapping).fillna(np.mean(y[tr])).to_numpy()
        out = self.transform(x)
        for cat in self.cats:
            out[f"{cat}__te"] = encoded[f"{cat}__te"].fillna(self.global_mean).to_numpy()
        return out


def make_model(name: str, seed: int) -> Any:
    common = dict(n_estimators=1500, learning_rate=0.015, subsample=0.7, colsample_bytree=0.7)
    if name == "lightgbm":
        return LGBMClassifier(**common, objective="binary", num_leaves=24, max_depth=5, min_child_samples=35,
                              reg_alpha=0.15, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1)
    if name == "catboost":
        return CatBoostClassifier(iterations=1500, learning_rate=0.015, subsample=0.7, colsample_bylevel=0.7, depth=5,
                                  l2_leaf_reg=5.0, loss_function="Logloss", random_seed=seed,
                                  verbose=False, allow_writing_files=False, thread_count=-1)
    return XGBClassifier(**common, objective="binary:logistic", eval_metric="logloss", max_depth=5,
                         min_child_weight=25, gamma=0.1, reg_alpha=0.15, reg_lambda=1.0,
                         random_state=seed, n_jobs=-1, tree_method="hist", early_stopping_rounds=100)


def fit_predict(name: str, model: Any, xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if name == "lightgbm":
        model.fit(xtr, ytr, eval_set=[(xva, yva)], callbacks=[lgb_early_stopping(100, verbose=False)])
    elif name == "catboost":
        model.fit(xtr, ytr, eval_set=(xva, yva), early_stopping_rounds=100, verbose=False)
    else:
        model.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
    return model.predict_proba(xva)[:, 1], model.predict_proba(xt)[:, 1]


def crossfit_isotonic(oof: np.ndarray, test: np.ndarray, y: np.ndarray, folds: StratifiedKFold) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted OOF calibration avoids scoring an isotonic fit on itself."""
    calibrated = np.zeros_like(oof, dtype=float)
    for tr, va in folds.split(oof.reshape(-1, 1), y):
        iso = IsotonicRegression(y_min=CLIP_LOW, y_max=CLIP_HIGH, out_of_bounds="clip")
        iso.fit(oof[tr], y[tr])
        calibrated[va] = iso.predict(oof[va])
    final_iso = IsotonicRegression(y_min=CLIP_LOW, y_max=CLIP_HIGH, out_of_bounds="clip").fit(oof, y)
    return clip(calibrated), clip(final_iso.predict(test))


def crossfit_platt(oof: np.ndarray, test: np.ndarray, y: np.ndarray, folds: StratifiedKFold) -> tuple[np.ndarray, np.ndarray]:
    """Fit logistic probability calibration without evaluating on fitted rows."""
    calibrated = np.zeros_like(oof, dtype=float)
    for tr, va in folds.split(oof.reshape(-1, 1), y):
        calibrator = LogisticRegression(C=1.0, max_iter=1000)
        calibrator.fit(oof[tr].reshape(-1, 1), y[tr])
        calibrated[va] = calibrator.predict_proba(oof[va].reshape(-1, 1))[:, 1]
    final_calibrator = LogisticRegression(C=1.0, max_iter=1000)
    final_calibrator.fit(oof.reshape(-1, 1), y)
    return clip(calibrated), clip(final_calibrator.predict_proba(test.reshape(-1, 1))[:, 1])


def select_calibration(
    oof: np.ndarray, test: np.ndarray, y: np.ndarray, folds: StratifiedKFold
) -> tuple[np.ndarray, np.ndarray, str, dict[str, float]]:
    """Choose calibration by cross-fitted leaderboard metric, avoiding in-sample selection."""
    candidates = {
        "raw": (clip(oof), clip(test)),
        "isotonic": crossfit_isotonic(oof, test, y, folds),
        "platt": crossfit_platt(oof, test, y, folds),
    }
    scores = {name: weighted_score(y, prediction[0])[2] for name, prediction in candidates.items()}
    selected = max(scores, key=scores.get)
    return *candidates[selected], selected, scores


def checkpoint_valid(path: Path, meta_expected: dict[str, Any]) -> bool:
    meta = path / "checkpoint_meta.json"
    if not (meta.exists() and (path / "oof_preds.npy").exists() and (path / "test_preds.npy").exists()):
        return False
    try:
        saved = json.loads(meta.read_text())
        return saved == meta_expected
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    train, test = pd.read_csv(args.train_path), pd.read_csv(args.test_path)
    label = find_label(train, test, args.label_col)
    if args.id_col not in train or args.id_col not in test:
        raise ValueError(f"ID column {args.id_col!r} must exist in train and test.")
    y = train[label].astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{label!r} must be binary 0/1.")

    # Build independently, then align exactly; never identify train/test rows by ID membership.
    x_train_df = MobileMoneyFeatures(train.drop(columns=[label]), args.id_col).build()
    x_test_df = MobileMoneyFeatures(test, args.id_col).build().reindex(columns=x_train_df.columns)
    logging.info("target=%s | train=%d | test=%d", label, len(y), len(test))

    model_seeds = tuple(int(v.strip()) for v in args.model_seeds.split(",") if v.strip())
    if not model_seeds:
        raise ValueError("--model-seeds must contain at least one integer.")
    meta = {"n_train": len(y), "n_test": len(test), "seed": args.seed, "splits": args.n_splits, "models": list(MODEL_NAMES), "feature_version": 4, "advanced_fold_features": args.advanced_fold_features, "model_seeds": list(model_seeds)}
    if checkpoint_valid(outdir, meta) and not args.force_retrain:
        logging.info("Using raw prediction checkpoint. Use --force-retrain to rebuild it.")
        oof_raw = np.load(outdir / "oof_preds.npy", allow_pickle=True).item()
        test_raw = np.load(outdir / "test_preds.npy", allow_pickle=True).item()
    else:
        folds = StratifiedKFold(args.n_splits, shuffle=True, random_state=args.seed)
        oof_raw = {name: np.zeros(len(y)) for name in MODEL_NAMES}
        test_parts: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_NAMES}
        for fold, (tr, va) in enumerate(folds.split(x_train_df, y), 1):
            logging.info("fold %d/%d", fold, args.n_splits)
            # Every supervised statistic, including target encoding and the
            # choice of interaction columns, is learned inside this fold.
            if args.advanced_fold_features:
                fe = FoldSafeFeatureEngineer(args.seed + fold)
                tr_df = fe.fit_transform_oof(x_train_df.iloc[tr].reset_index(drop=True), y[tr])
                va_df = fe.transform(x_train_df.iloc[va].reset_index(drop=True))
                te_df = fe.transform(x_test_df.reset_index(drop=True))
            else:
                # The supplied data has only low-cardinality categoricals;
                # retaining the lean monthly/RFM set generalises better.
                tr_df = x_train_df.iloc[tr].reset_index(drop=True)
                va_df = x_train_df.iloc[va].reset_index(drop=True)
                te_df = x_test_df.reset_index(drop=True)
            numeric = tr_df.select_dtypes(include=np.number).columns.tolist()
            categorical = [c for c in tr_df.columns if c not in numeric]
            prep = ColumnTransformer([
                ("num", SimpleImputer(strategy="median"), numeric),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
            ])
            xtr, xva, xt = prep.fit_transform(tr_df), prep.transform(va_df), prep.transform(te_df)
            for name in MODEL_NAMES:
                seeds_for_model = model_seeds if name != "catboost" else model_seeds[:1]
                fold_valid, fold_test = [], []
                for model_seed in seeds_for_model:
                    va_pred, te_pred = fit_predict(name, make_model(name, model_seed + 1009 * fold), xtr, y[tr], xva, y[va], xt)
                    fold_valid.append(va_pred)
                    fold_test.append(te_pred)
                va_pred, te_pred = np.mean(fold_valid, axis=0), np.mean(fold_test, axis=0)
                oof_raw[name][va] = va_pred
                test_parts[name].append(te_pred)
                ll, auc, score = weighted_score(y[va], va_pred)
                logging.info("  %-9s logloss=%.5f auc=%.5f score=%.5f", name, ll, auc, score)
        test_raw = {name: np.mean(parts, axis=0) for name, parts in test_parts.items()}
        np.save(outdir / "oof_preds.npy", oof_raw, allow_pickle=True)
        np.save(outdir / "test_preds.npy", test_raw, allow_pickle=True)
        (outdir / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2))

    calibration_folds = StratifiedKFold(args.n_splits, shuffle=True, random_state=args.seed + 991)
    calibrated_oof, calibrated_test, calibration_methods, calibration_scores = {}, {}, {}, {}
    for name in MODEL_NAMES:
        (
            calibrated_oof[name],
            calibrated_test[name],
            calibration_methods[name],
            calibration_scores[name],
        ) = select_calibration(np.asarray(oof_raw[name]), np.asarray(test_raw[name]), y, calibration_folds)
        logging.info(
            "%s calibration=%s | OOF scores=%s",
            name,
            calibration_methods[name],
            {method: round(value, 5) for method, value in calibration_scores[name].items()},
        )
    oof_matrix = np.column_stack([calibrated_oof[n] for n in MODEL_NAMES])
    test_matrix = np.column_stack([calibrated_test[n] for n in MODEL_NAMES])

    def objective(w: np.ndarray) -> float:
        return -weighted_score(y, oof_matrix @ w)[2]
    result = minimize(objective, np.full(3, 1 / 3), method="SLSQP", bounds=[(0, 1)] * 3,
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, options={"maxiter": 300, "ftol": 1e-10})
    weights = result.x if result.success and np.all(result.x >= 0) else np.full(3, 1 / 3)
    weights = weights / weights.sum()
    # Rank averaging is robust to differently calibrated base distributions.
    rank_oof = np.mean([rankdata(oof_matrix[:, i]) / (len(y) + 1) for i in range(3)], axis=0)
    rank_test = np.mean([rankdata(test_matrix[:, i]) / (len(test) + 1) for i in range(3)], axis=0)
    # Cross-fitted logistic stacker gives an unbiased OOF ensemble estimate.
    stack_oof = np.zeros(len(y)); stack_test_parts = []
    stack_folds = StratifiedKFold(args.n_splits, shuffle=True, random_state=args.seed + 2026)
    for tr, va in stack_folds.split(oof_matrix, y):
        meta_model = LogisticRegression(C=0.5, max_iter=2000, class_weight=None)
        meta_model.fit(oof_matrix[tr], y[tr])
        stack_oof[va] = meta_model.predict_proba(oof_matrix[va])[:, 1]
        stack_test_parts.append(meta_model.predict_proba(test_matrix)[:, 1])
    stack_test = np.mean(stack_test_parts, axis=0)
    candidates = {
        "optimized_blend": (clip(oof_matrix @ weights), clip(test_matrix @ weights)),
        "rank_average": (clip(rank_oof), clip(rank_test)),
        "logistic_stack": (clip(stack_oof), clip(stack_test)),
    }
    candidate_scores = {name: weighted_score(y, pred[0])[2] for name, pred in candidates.items()}
    chosen = max(candidate_scores, key=candidate_scores.get)
    final_oof, final_test = candidates[chosen]
    ll, auc, score = weighted_score(y, final_oof)
    logging.info("OOF %s: logloss=%.5f auc=%.5f score=%.5f | model scores=%s | weights=%s", chosen, ll, auc, score, {n: round(weighted_score(y, calibrated_oof[n])[2], 5) for n in MODEL_NAMES}, dict(zip(MODEL_NAMES, weights.round(4))))

    np.save(outdir / "final_test_preds.npy", final_test)
    pd.DataFrame({args.id_col: train[args.id_col], label: y, "Target": final_oof}).to_csv(outdir / "oof_predictions.csv", index=False)
    pd.DataFrame({args.id_col: test[args.id_col], "Target": final_test}).to_csv(outdir / "test_predictions.csv", index=False)
    template = pd.read_csv(args.sample_submission_path)
    template_id = args.id_col if args.id_col in template else template.columns[0]
    pred_by_id = pd.Series(final_test, index=test[args.id_col].astype(str))
    submission = pd.DataFrame({template_id: template[template_id], "Target": template[template_id].astype(str).map(pred_by_id)})
    if submission["Target"].isna().any():
        raise ValueError("SampleSubmission IDs do not all occur in Test.csv.")
    submission["Target"] = clip(submission["Target"].to_numpy())
    submission.to_csv(outdir / "fixed_submission.csv", index=False)
    # Diagnostic only: hard labels are invalid for this probability leaderboard.
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = float(thresholds[np.argmax([(2 * (((final_oof >= t) & (y == 1)).sum())) / (2 * (((final_oof >= t) & (y == 1)).sum()) + ((final_oof >= t) & (y == 0)).sum() + ((final_oof < t) & (y == 1)).sum() + EPS) for t in thresholds])])
    (outdir / "metrics_summary.json").write_text(json.dumps({"label": label, "logloss": ll, "roc_auc": auc, "score": score, "weights": dict(zip(MODEL_NAMES, map(float, weights))), "candidate_scores": candidate_scores, "selected_ensemble": chosen, "calibration_methods": calibration_methods, "calibration_scores": calibration_scores, "diagnostic_f1_threshold": best_threshold}, indent=2))


if __name__ == "__main__":
    main()
