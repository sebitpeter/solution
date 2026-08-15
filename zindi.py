"""Feature engineering pipeline for mobile money financial stress prediction.

The script builds customer-level features from raw transaction logs using Polars for
the expensive windowed aggregations and Pandas for the final cleaning and export.

Expected inputs:
  - transactions file with at least:
      customer_id, timestamp, transaction_type, amount, status, channel
  - train cohort file with at least:
      customer_id, cutoff_date, and optionally target
  - test cohort file with at least:
      customer_id, cutoff_date

Example:
  python zindi.py \
    --transactions-path transactions.csv \
    --train-cohort-path train_customers.csv \
    --test-cohort-path test_customers.csv \
    --train-output train_features.csv \
    --test-output test_features.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl


DEFAULT_WINDOWS = (7, 14, 30, 90, 180)
FAILED_LABELS = {"failed", "reversed"}

TYPE_SLUGS = {
    "cash in": "cash_in",
    "cash out": "cash_out",
    "airtime": "airtime",
    "p2p transfer": "p2p_transfer",
    "microloan repayment": "microloan_repayment",
    "bill pay": "bill_pay",
}


@dataclass(frozen=True)
class Columns:
    customer_id: str = "customer_id"
    timestamp: str = "timestamp"
    transaction_type: str = "transaction_type"
    amount: str = "amount"
    status: str = "status"
    channel: str = "channel"
    cutoff_date: str = "cutoff_date"
    target: str = "target"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate customer-level features for a stress prediction model.")
    parser.add_argument("--example", action="store_true", help="Print a built-in example command and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Show the resolved configuration and exit without processing files.")
    parser.add_argument("--transactions-path", required=True, help="Path to the raw transaction log (CSV or Parquet).")
    parser.add_argument("--train-cohort-path", required=True, help="Path to the train cohort file with customer_id and cutoff_date.")
    parser.add_argument("--test-cohort-path", required=True, help="Path to the test cohort file with customer_id and cutoff_date.")
    parser.add_argument("--train-output", default="train_features.csv", help="Output CSV path for train features.")
    parser.add_argument("--test-output", default="test_features.csv", help="Output CSV path for test features.")
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--transaction-type-col", default="transaction_type")
    parser.add_argument("--amount-col", default="amount")
    parser.add_argument("--status-col", default="status")
    parser.add_argument("--channel-col", default="channel")
    parser.add_argument("--cutoff-col", default="cutoff_date")
    parser.add_argument("--target-col", default="target")
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=list(DEFAULT_WINDOWS),
        help="Lookback windows in days used for rolling features.",
    )
    return parser


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def print_example_usage() -> None:
    print(
        "Example command:\n"
        "  python zindi.py --transactions-path transactions.csv --train-cohort-path train_customers.csv "
        "--test-cohort-path test_customers.csv --train-output train_features.csv --test-output test_features.csv\n\n"
        "Dry-run example:\n"
        "  python zindi.py --example\n"
    )


def _read_lazy_table(path: str) -> pl.LazyFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    return pl.scan_csv(path, infer_schema_length=2000, ignore_errors=False)


def _clean_text_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).fill_null("").str.strip_chars()


def _load_transactions(path: str, cols: Columns) -> pl.LazyFrame:
    logging.info("Loading transactions from %s", path)
    lf = _read_lazy_table(path).select(
        [
            cols.customer_id,
            cols.timestamp,
            cols.transaction_type,
            cols.amount,
            cols.status,
            cols.channel,
        ]
    )
    return (
        lf.with_columns(
            [
                pl.col(cols.timestamp).cast(pl.Utf8).str.strptime(pl.Datetime, strict=False, exact=False).alias(cols.timestamp),
                pl.col(cols.amount).cast(pl.Float64, strict=False).alias(cols.amount),
                _clean_text_expr(cols.transaction_type).str.to_lowercase().alias("_txn_type_norm"),
                _clean_text_expr(cols.status).str.to_lowercase().alias("_status_norm"),
                _clean_text_expr(cols.channel).str.to_lowercase().alias("_channel_norm"),
            ]
        )
        .sort([cols.customer_id, cols.timestamp])
    )


def _load_cohort(path: str, cols: Columns, keep_target: bool) -> pl.LazyFrame:
    logging.info("Loading cohort from %s", path)
    lf = _read_lazy_table(path).select(
        [
            cols.customer_id,
            cols.cutoff_date,
            *([cols.target] if keep_target else []),
        ]
    )
    return lf.with_columns(
        [
            pl.col(cols.cutoff_date).cast(pl.Utf8).str.strptime(pl.Datetime, strict=False, exact=False).alias(cols.cutoff_date),
            *( [pl.col(cols.target)] if keep_target else [] ),
        ]
    ).unique(subset=[cols.customer_id], keep="last")


def _type_feature_exprs(type_norm_col: str, amount_col: str, prefix: str) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for raw_type, slug in TYPE_SLUGS.items():
        exprs.append(
            pl.when(pl.col(type_norm_col) == raw_type)
            .then(pl.col(amount_col))
            .otherwise(0.0)
            .sum()
            .alias(f"{prefix}{slug}_sum")
        )
        exprs.append(
            pl.when(pl.col(type_norm_col) == raw_type)
            .then(1)
            .otherwise(0)
            .sum()
            .alias(f"{prefix}{slug}_count")
        )
    return exprs


def _aggregate_slice(
    lf: pl.LazyFrame,
    cols: Columns,
    prefix: str,
    include_time_bounds: bool,
) -> pl.LazyFrame:
    failed_expr = pl.col("_status_norm").is_in(FAILED_LABELS) | pl.col("_txn_type_norm").str.contains("reverse")
    exprs: list[pl.Expr] = [
        pl.len().alias(f"{prefix}txn_count"),
        pl.col(cols.amount).sum().alias(f"{prefix}amount_sum"),
        pl.col(cols.amount).mean().alias(f"{prefix}amount_mean"),
        pl.col(cols.amount).std().alias(f"{prefix}amount_std"),
        pl.col(cols.amount).min().alias(f"{prefix}amount_min"),
        pl.col(cols.amount).max().alias(f"{prefix}amount_max"),
        pl.col(cols.amount).median().alias(f"{prefix}amount_median"),
        pl.col(cols.timestamp).dt.date().n_unique().alias(f"{prefix}active_days"),
        pl.col("_txn_type_norm").n_unique().alias(f"{prefix}transaction_type_nunique"),
        pl.col("_channel_norm").n_unique().alias(f"{prefix}channel_nunique"),
        pl.col("_status_norm").eq("success").sum().alias(f"{prefix}success_txn_count"),
        failed_expr.sum().alias(f"{prefix}failed_or_reversed_txn_count"),
        pl.col(cols.timestamp).dt.hour().lt(5).sum().alias(f"{prefix}night_txn_count"),
    ]
    if include_time_bounds:
        exprs.extend(
            [
                pl.col(cols.timestamp).min().alias(f"{prefix}first_txn_ts"),
                pl.col(cols.timestamp).max().alias(f"{prefix}last_txn_ts"),
            ]
        )
    exprs.extend(_type_feature_exprs("_txn_type_norm", cols.amount, prefix))
    return lf.group_by(cols.customer_id).agg(exprs)


def _build_feature_frame(
    transactions: pl.LazyFrame,
    cohort: pl.LazyFrame,
    cols: Columns,
    windows: Sequence[int],
    keep_target: bool,
) -> pl.LazyFrame:
    cohort_columns = [cols.customer_id, cols.cutoff_date]
    if keep_target:
        cohort_columns.append(cols.target)
    cohort_base = cohort.select(cohort_columns)
    base_join = (
        transactions.join(cohort_base.select([cols.customer_id, cols.cutoff_date]), on=cols.customer_id, how="inner")
        .filter(pl.col(cols.timestamp) < pl.col(cols.cutoff_date))
    )

    feature_lfs: list[pl.LazyFrame] = [
        _aggregate_slice(base_join, cols, prefix="overall_", include_time_bounds=True)
    ]

    for window in windows:
        window_slice = base_join.filter(pl.col(cols.timestamp) >= (pl.col(cols.cutoff_date) - pl.duration(days=window)))
        feature_lfs.append(_aggregate_slice(window_slice, cols, prefix=f"d{window}_", include_time_bounds=False))

    features = feature_lfs[0]
    for frame in feature_lfs[1:]:
        features = features.join(frame, on=cols.customer_id, how="outer")

    return cohort_base.join(features, on=cols.customer_id, how="left")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator.divide(denominator).replace([np.inf, -np.inf], np.nan)


def _finalize_features(df: pd.DataFrame, cols: Columns, windows: Sequence[int]) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)

    if cols.cutoff_date in df.columns:
        df[cols.cutoff_date] = pd.to_datetime(df[cols.cutoff_date], errors="coerce")

    if "overall_last_txn_ts" in df.columns:
        df["overall_last_txn_ts"] = pd.to_datetime(df["overall_last_txn_ts"], errors="coerce")
        df["overall_first_txn_ts"] = pd.to_datetime(df["overall_first_txn_ts"], errors="coerce")
        df["recency_days"] = (df[cols.cutoff_date] - df["overall_last_txn_ts"]).dt.total_seconds().div(86400.0)
        df["history_span_days"] = (df["overall_last_txn_ts"] - df["overall_first_txn_ts"]).dt.total_seconds().div(86400.0)
        df.drop(columns=["overall_first_txn_ts", "overall_last_txn_ts"], inplace=True)
    else:
        df["recency_days"] = np.nan
        df["history_span_days"] = np.nan

    df["frequency"] = df.get("overall_txn_count", 0)
    df["monetary_sum"] = df.get("overall_amount_sum", 0.0)
    df["monetary_mean"] = df.get("overall_amount_mean", 0.0)
    df["monetary_std"] = df.get("overall_amount_std", 0.0)

    spend_columns = [f"overall_{slug}_sum" for slug in TYPE_SLUGS.values() if slug != "cash_in"]
    df["total_spend_sum"] = df[spend_columns].sum(axis=1, min_count=1) if spend_columns else 0.0
    df["airtime_spend_ratio"] = _safe_divide(df.get("overall_airtime_sum", pd.Series(0.0, index=df.index)), df["total_spend_sum"])
    df["cash_in_14d_to_90d_ratio"] = _safe_divide(df.get("d14_cash_in_sum", pd.Series(0.0, index=df.index)), df.get("d90_cash_in_sum", pd.Series(0.0, index=df.index)))
    df["failed_or_reversed_txn_rate"] = _safe_divide(
        df.get("overall_failed_or_reversed_txn_count", pd.Series(0.0, index=df.index)),
        df.get("overall_txn_count", pd.Series(0.0, index=df.index)),
    )
    df["night_txn_pct"] = _safe_divide(
        df.get("overall_night_txn_count", pd.Series(0.0, index=df.index)),
        df.get("overall_txn_count", pd.Series(0.0, index=df.index)),
    )
    df["txn_per_active_day"] = _safe_divide(df.get("overall_txn_count", pd.Series(0.0, index=df.index)), df.get("overall_active_days", pd.Series(0.0, index=df.index)))

    recency_fill = float(max(windows) + 1)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == "recency_days":
            df[col] = df[col].fillna(recency_fill)
        else:
            df[col] = df[col].fillna(0.0)

    if cols.target in df.columns:
        ordered_cols = [cols.customer_id, cols.target]
    else:
        ordered_cols = [cols.customer_id]

    feature_cols = [col for col in df.columns if col not in ordered_cols and col != cols.cutoff_date]
    ordered_cols.extend(sorted(feature_cols))

    return df.loc[:, ordered_cols]


def _collect_to_pandas(lf: pl.LazyFrame) -> pd.DataFrame:
    return pd.DataFrame(lf.collect().to_dicts())


def process_split(
    transactions: pl.LazyFrame,
    cohort_path: str,
    cols: Columns,
    windows: Sequence[int],
    keep_target: bool,
) -> pd.DataFrame:
    cohort = _load_cohort(cohort_path, cols, keep_target=keep_target)
    feature_frame = _build_feature_frame(transactions, cohort, cols, windows, keep_target=keep_target)
    return _collect_to_pandas(feature_frame)


def main() -> int:
    setup_logging()
    parser = build_parser()
    if len(sys.argv) == 1:
        print_example_usage()
        return 0
    args = parser.parse_args()

    if args.example:
        print_example_usage()
        return 0

    if args.dry_run:
        print("Dry run configuration:")
        for key, value in sorted(vars(args).items()):
            print(f"  {key}: {value}")
        return 0

    cols = Columns(
        customer_id=args.customer_id_col,
        timestamp=args.timestamp_col,
        transaction_type=args.transaction_type_col,
        amount=args.amount_col,
        status=args.status_col,
        channel=args.channel_col,
        cutoff_date=args.cutoff_col,
        target=args.target_col,
    )

    transactions = _load_transactions(args.transactions_path, cols)

    train_features = process_split(
        transactions=transactions,
        cohort_path=args.train_cohort_path,
        cols=cols,
        windows=args.windows,
        keep_target=True,
    )
    test_features = process_split(
        transactions=transactions,
        cohort_path=args.test_cohort_path,
        cols=cols,
        windows=args.windows,
        keep_target=False,
    )

    train_features = _finalize_features(train_features, cols, args.windows)
    test_features = _finalize_features(test_features, cols, args.windows)

    train_features.to_csv(args.train_output, index=False)
    test_features.to_csv(args.test_output, index=False)

    logging.info("Wrote %s (%d rows, %d cols)", args.train_output, len(train_features), train_features.shape[1])
    logging.info("Wrote %s (%d rows, %d cols)", args.test_output, len(test_features), test_features.shape[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())