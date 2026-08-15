"""PyTorch temporal pipeline for mobile money sequence feature extraction.

This script learns customer-level sequence embeddings from transaction logs using
either a GRU or an LSTM encoder. It supports supervised cross-fitting so the
exported train features are out-of-fold temporal embeddings, which makes them
safe to use as downstream features in stacking or blending.

Expected inputs:
  - transactions file with at least:
      customer_id, timestamp, transaction_type, amount, status, channel
  - train cohort file with at least:
      customer_id, cutoff_date, target
  - test cohort file with at least:
      customer_id, cutoff_date

Outputs:
  - train_temporal_features.csv
  - test_temporal_features.csv
  - temporal_model_metrics.json

Example:
  python zindi_temporal.py \
    --transactions-path transactions.csv \
    --train-cohort-path train_customers.csv \
    --test-cohort-path test_customers.csv \
    --output-dir temporal_outputs \
    --rnn-type gru
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = 0
UNK_TOKEN = 1
EPS = 1e-15


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


@dataclass
class CustomerEvents:
    timestamps_ns: np.ndarray
    amount: np.ndarray
    hour: np.ndarray
    dayofweek: np.ndarray
    transaction_type_ids: np.ndarray
    status_ids: np.ndarray
    channel_ids: np.ndarray


@dataclass
class SequenceSample:
    customer_id: str
    cutoff_ns: int
    target: float | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GRU/LSTM temporal encoder for sequence feature extraction.")
    parser.add_argument("--transactions-path", required=True, help="Path to the raw transaction log (CSV or Parquet).")
    parser.add_argument("--train-cohort-path", required=True, help="Path to the train cohort file with customer_id, cutoff_date, and target.")
    parser.add_argument("--test-cohort-path", required=True, help="Path to the test cohort file with customer_id and cutoff_date.")
    parser.add_argument("--output-dir", default="temporal_outputs", help="Directory where embeddings and metrics will be written.")
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--transaction-type-col", default="transaction_type")
    parser.add_argument("--amount-col", default="amount")
    parser.add_argument("--status-col", default="status")
    parser.add_argument("--channel-col", default="channel")
    parser.add_argument("--cutoff-col", default="cutoff_date")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru", help="Temporal encoder family.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size for the recurrent encoder.")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Dimension of the exported temporal embedding.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of recurrent layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout used in the encoder and classifier.")
    parser.add_argument("--max-seq-len", type=int, default=120, help="Maximum number of events retained per customer.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training and embedding extraction.")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum number of epochs per fold.")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of StratifiedKFold splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Torch device selection.")
    return parser


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_vocab(series: pd.Series) -> dict[str, int]:
    values = series.astype(str).fillna("missing").str.strip().str.lower()
    uniques = [value for value in pd.unique(values) if value not in {"", "nan", "none"}]
    vocab = {"<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN}
    for value in sorted(uniques):
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def encode_with_vocab(series: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    values = series.astype(str).fillna("missing").str.strip().str.lower()
    return np.asarray([vocab.get(value, UNK_TOKEN) for value in values], dtype=np.int64)


def clamp_probabilities(preds: np.ndarray) -> np.ndarray:
    return np.clip(preds.astype(float), 0.001, 0.999)


def custom_metric(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_pred = clamp_probabilities(y_pred)
    ll = log_loss(y_true, y_pred, labels=[0, 1])
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.5
    score = 0.6 * (1.0 - ll) + 0.4 * auc
    return ll, auc, score


def read_and_prepare_transactions(path: str, cols: Columns) -> pd.DataFrame:
    df = read_table(path).copy()
    required = [cols.customer_id, cols.timestamp, cols.transaction_type, cols.amount, cols.status, cols.channel]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Transactions file is missing required columns: {missing}")

    df[cols.timestamp] = pd.to_datetime(df[cols.timestamp], errors="coerce")
    df = df.dropna(subset=[cols.customer_id, cols.timestamp])
    df[cols.customer_id] = df[cols.customer_id].astype(str)
    df = df.sort_values([cols.customer_id, cols.timestamp]).reset_index(drop=True)
    df[cols.amount] = pd.to_numeric(df[cols.amount], errors="coerce").fillna(0.0)
    df[cols.transaction_type] = df[cols.transaction_type].astype(str).fillna("missing")
    df[cols.status] = df[cols.status].astype(str).fillna("missing")
    df[cols.channel] = df[cols.channel].astype(str).fillna("missing")
    return df


def read_cohort(path: str, cols: Columns, keep_target: bool) -> pd.DataFrame:
    df = read_table(path).copy()
    required = [cols.customer_id, cols.cutoff_date]
    if keep_target:
        required.append(cols.target)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Cohort file is missing required columns: {missing}")

    df[cols.customer_id] = df[cols.customer_id].astype(str)
    df[cols.cutoff_date] = pd.to_datetime(df[cols.cutoff_date], errors="coerce")
    df = df.dropna(subset=[cols.customer_id, cols.cutoff_date]).reset_index(drop=True)
    if keep_target:
        df[cols.target] = pd.to_numeric(df[cols.target], errors="coerce").fillna(0).astype(int)
    return df


def build_customer_store(transactions: pd.DataFrame, cols: Columns, vocabs: dict[str, dict[str, int]]) -> dict[str, CustomerEvents]:
    store: dict[str, CustomerEvents] = {}
    grouped = transactions.groupby(cols.customer_id, sort=False)
    for customer_id, frame in grouped:
        timestamps = frame[cols.timestamp].view("int64").to_numpy()
        hours = frame[cols.timestamp].dt.hour.to_numpy(dtype=np.int64)
        dayofweek = frame[cols.timestamp].dt.dayofweek.to_numpy(dtype=np.int64)
        store[customer_id] = CustomerEvents(
            timestamps_ns=timestamps,
            amount=frame[cols.amount].to_numpy(dtype=np.float32),
            hour=hours,
            dayofweek=dayofweek,
            transaction_type_ids=encode_with_vocab(frame[cols.transaction_type], vocabs[cols.transaction_type]),
            status_ids=encode_with_vocab(frame[cols.status], vocabs[cols.status]),
            channel_ids=encode_with_vocab(frame[cols.channel], vocabs[cols.channel]),
        )
    return store


class TemporalSequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[SequenceSample],
        store: dict[str, CustomerEvents],
        max_seq_len: int,
        has_target: bool,
    ) -> None:
        self.samples = samples
        self.store = store
        self.max_seq_len = max_seq_len
        self.has_target = has_target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        events = self.store.get(sample.customer_id)
        if events is None:
            return self._empty_sample(sample)

        cutoff_ns = sample.cutoff_ns
        end = np.searchsorted(events.timestamps_ns, cutoff_ns, side="left")
        if end <= 0:
            return self._empty_sample(sample)

        start = max(0, end - self.max_seq_len)
        timestamps = events.timestamps_ns[start:end]
        amount = events.amount[start:end]
        hour = events.hour[start:end]
        dayofweek = events.dayofweek[start:end]
        type_ids = events.transaction_type_ids[start:end]
        status_ids = events.status_ids[start:end]
        channel_ids = events.channel_ids[start:end]

        numeric = self._build_numeric_features(timestamps, amount, hour, dayofweek, cutoff_ns)

        item: dict[str, torch.Tensor | str] = {
            "customer_id": sample.customer_id,
            "length": torch.tensor(len(type_ids), dtype=torch.long),
            "numeric": torch.from_numpy(numeric),
            "type_ids": torch.from_numpy(type_ids),
            "status_ids": torch.from_numpy(status_ids),
            "channel_ids": torch.from_numpy(channel_ids),
        }
        if self.has_target:
            item["target"] = torch.tensor(float(sample.target), dtype=torch.float32)
        return item

    def _empty_sample(self, sample: SequenceSample) -> dict[str, torch.Tensor | str]:
        numeric = np.zeros((1, 9), dtype=np.float32)
        zeros = np.zeros(1, dtype=np.int64)
        item: dict[str, torch.Tensor | str] = {
            "customer_id": sample.customer_id,
            "length": torch.tensor(1, dtype=torch.long),
            "numeric": torch.from_numpy(numeric),
            "type_ids": torch.from_numpy(zeros),
            "status_ids": torch.from_numpy(zeros),
            "channel_ids": torch.from_numpy(zeros),
        }
        if self.has_target:
            item["target"] = torch.tensor(float(sample.target), dtype=torch.float32)
        return item

    @staticmethod
    def _build_numeric_features(
        timestamps_ns: np.ndarray,
        amount: np.ndarray,
        hour: np.ndarray,
        dayofweek: np.ndarray,
        cutoff_ns: int,
    ) -> np.ndarray:
        timestamps_sec = timestamps_ns.astype(np.float64) / 1e9
        delta_hours = np.diff(np.concatenate(([timestamps_sec[0]], timestamps_sec))) / 3600.0
        delta_hours[0] = 0.0
        recency_hours = (cutoff_ns - timestamps_ns).astype(np.float64) / 3.6e12
        amount_log = np.log1p(np.maximum(amount.astype(np.float64), 0.0))
        hour_rad = 2.0 * math.pi * (hour.astype(np.float64) / 24.0)
        dow_rad = 2.0 * math.pi * (dayofweek.astype(np.float64) / 7.0)
        is_night = ((hour >= 0) & (hour <= 5)).astype(np.float64)
        is_weekend = ((dayofweek >= 5)).astype(np.float64)
        numeric = np.column_stack(
            [
                amount_log,
                np.log1p(delta_hours),
                np.sin(hour_rad),
                np.cos(hour_rad),
                np.sin(dow_rad),
                np.cos(dow_rad),
                is_night,
                is_weekend,
                np.log1p(np.maximum(recency_hours, 0.0)),
            ]
        ).astype(np.float32)
        return numeric


def collate_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    lengths = torch.stack([item["length"] for item in batch]).long()
    numeric = pad_sequence([item["numeric"] for item in batch], batch_first=True, padding_value=0.0)
    type_ids = pad_sequence([item["type_ids"] for item in batch], batch_first=True, padding_value=PAD_TOKEN)
    status_ids = pad_sequence([item["status_ids"] for item in batch], batch_first=True, padding_value=PAD_TOKEN)
    channel_ids = pad_sequence([item["channel_ids"] for item in batch], batch_first=True, padding_value=PAD_TOKEN)
    output: dict[str, torch.Tensor | list[str]] = {
        "customer_id": [str(item["customer_id"]) for item in batch],
        "lengths": lengths,
        "numeric": numeric,
        "type_ids": type_ids,
        "status_ids": status_ids,
        "channel_ids": channel_ids,
    }
    if "target" in batch[0]:
        output["target"] = torch.stack([item["target"] for item in batch]).float()
    return output


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        type_vocab_size: int,
        status_vocab_size: int,
        channel_vocab_size: int,
        numeric_dim: int,
        hidden_size: int,
        embedding_dim: int,
        num_layers: int,
        dropout: float,
        rnn_type: str,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(type_vocab_size, 16, padding_idx=PAD_TOKEN)
        self.status_emb = nn.Embedding(status_vocab_size, 8, padding_idx=PAD_TOKEN)
        self.channel_emb = nn.Embedding(channel_vocab_size, 8, padding_idx=PAD_TOKEN)
        self.numeric_proj = nn.Sequential(
            nn.Linear(numeric_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        rnn_input_dim = 16 + 8 + 8 + 32
        rnn_kwargs = dict(
            input_size=rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        if rnn_type == "gru":
            self.rnn: nn.GRU | nn.LSTM = nn.GRU(**rnn_kwargs)
        else:
            self.rnn = nn.LSTM(**rnn_kwargs)

        rnn_out_dim = hidden_size * (2 if bidirectional else 1)
        self.embedding_head = nn.Sequential(
            nn.Linear(rnn_out_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def encode(
        self,
        numeric: torch.Tensor,
        type_ids: torch.Tensor,
        status_ids: torch.Tensor,
        channel_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                self.type_emb(type_ids),
                self.status_emb(status_ids),
                self.channel_emb(channel_ids),
                self.numeric_proj(numeric),
            ],
            dim=-1,
        )
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        mask = self._length_mask(lengths, output.size(1), output.device)

        if self.rnn_type == "lstm":
            hidden_state = hidden[0]
        else:
            hidden_state = hidden

        last_hidden = self._last_hidden(hidden_state)
        mean_pool = self._masked_mean(output, mask)
        max_pool = self._masked_max(output, mask)
        stacked = torch.cat([last_hidden, mean_pool, max_pool], dim=-1)
        return self.embedding_head(stacked)

    def forward(
        self,
        numeric: torch.Tensor,
        type_ids: torch.Tensor,
        status_ids: torch.Tensor,
        channel_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(numeric, type_ids, status_ids, channel_ids, lengths)
        logits = self.classifier(embeddings).squeeze(-1)
        return logits, embeddings

    def _last_hidden(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            forward_last = hidden_state[-2]
            backward_last = hidden_state[-1]
            return torch.cat([forward_last, backward_last], dim=-1)
        return hidden_state[-1]

    @staticmethod
    def _length_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
        return (torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)

    @staticmethod
    def _masked_mean(output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = output * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        return masked.sum(dim=1) / denom

    @staticmethod
    def _masked_max(output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = output.masked_fill(~mask, float("-inf"))
        return masked.max(dim=1).values


def build_samples(cohort: pd.DataFrame, cols: Columns) -> list[SequenceSample]:
    samples: list[SequenceSample] = []
    has_target = cols.target in cohort.columns
    for row in cohort.itertuples(index=False):
        target = float(getattr(row, cols.target)) if has_target else None
        samples.append(
            SequenceSample(
                customer_id=str(getattr(row, cols.customer_id)),
                cutoff_ns=pd.Timestamp(getattr(row, cols.cutoff_date)).value,
                target=target,
            )
        )
    return samples


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
        drop_last=False,
    )


def train_one_fold(
    model: TemporalEncoder,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[TemporalEncoder, dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    best_metrics: dict[str, float] = {"logloss": float("inf"), "roc_auc": 0.5, "weighted_score": -float("inf")}
    stagnant = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, valid_loader, device)
        logging.info(
            "Epoch %d | train_loss=%.6f | val_logloss=%.6f | val_auc=%.6f | val_score=%.6f",
            epoch,
            train_loss,
            val_metrics["logloss"],
            val_metrics["roc_auc"],
            val_metrics["weighted_score"],
        )

        if val_metrics["weighted_score"] > best_score:
            best_score = val_metrics["weighted_score"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_metrics


def run_train_epoch(model: TemporalEncoder, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    losses: list[float] = []
    criterion = nn.BCEWithLogitsLoss()
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(
            numeric=batch["numeric"].to(device),
            type_ids=batch["type_ids"].to(device),
            status_ids=batch["status_ids"].to(device),
            channel_ids=batch["channel_ids"].to(device),
            lengths=batch["lengths"].to(device),
        )
        target = batch["target"].to(device)
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model: TemporalEncoder, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    for batch in loader:
        logits, _ = model(
            numeric=batch["numeric"].to(device),
            type_ids=batch["type_ids"].to(device),
            status_ids=batch["status_ids"].to(device),
            channel_ids=batch["channel_ids"].to(device),
            lengths=batch["lengths"].to(device),
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(batch["target"].cpu().numpy())

    if not all_probs:
        return {"logloss": float("inf"), "roc_auc": 0.5, "weighted_score": -float("inf")}

    y_true = np.concatenate(all_targets)
    y_pred = clamp_probabilities(np.concatenate(all_probs))
    ll, auc, score = custom_metric(y_true, y_pred)
    return {"logloss": ll, "roc_auc": auc, "weighted_score": score}


@torch.no_grad()
def extract_embeddings(model: TemporalEncoder, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    embeddings_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    ids: list[str] = []
    for batch in loader:
        logits, embeddings = model(
            numeric=batch["numeric"].to(device),
            type_ids=batch["type_ids"].to(device),
            status_ids=batch["status_ids"].to(device),
            channel_ids=batch["channel_ids"].to(device),
            lengths=batch["lengths"].to(device),
        )
        embeddings_list.append(embeddings.cpu().numpy())
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
        ids.extend(batch["customer_id"])
    embeddings = np.vstack(embeddings_list) if embeddings_list else np.empty((0, model.embedding_head[0].out_features))
    probs = np.concatenate(probs_list) if probs_list else np.empty((0,))
    return embeddings, probs, ids


def get_feature_names(dim: int) -> list[str]:
    return [f"temporal_emb_{idx:03d}" for idx in range(dim)]


def main() -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transactions = read_and_prepare_transactions(args.transactions_path, cols)
    train_cohort = read_cohort(args.train_cohort_path, cols, keep_target=True)
    test_cohort = read_cohort(args.test_cohort_path, cols, keep_target=False)

    vocabs = {
        cols.transaction_type: build_vocab(transactions[cols.transaction_type]),
        cols.status: build_vocab(transactions[cols.status]),
        cols.channel: build_vocab(transactions[cols.channel]),
    }
    store = build_customer_store(transactions, cols, vocabs)

    train_samples = build_samples(train_cohort, cols)
    test_samples = build_samples(test_cohort, cols)

    y = train_cohort[cols.target].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Training target must contain at least two classes.")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_embeddings = np.zeros((len(train_samples), args.embedding_dim), dtype=np.float32)
    oof_probs = np.zeros(len(train_samples), dtype=np.float32)
    test_embedding_folds: list[np.ndarray] = []
    test_prob_folds: list[np.ndarray] = []
    fold_metrics: list[dict[str, float | int]] = []

    train_indices = np.arange(len(train_samples))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_indices, y), start=1):
        logging.info("Starting fold %d/%d", fold, args.n_splits)
        fold_train_samples = [train_samples[i] for i in tr_idx]
        fold_valid_samples = [train_samples[i] for i in va_idx]

        train_dataset = TemporalSequenceDataset(fold_train_samples, store, args.max_seq_len, has_target=True)
        valid_dataset = TemporalSequenceDataset(fold_valid_samples, store, args.max_seq_len, has_target=True)
        test_dataset = TemporalSequenceDataset(test_samples, store, args.max_seq_len, has_target=False)

        train_loader = build_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
        valid_loader = build_loader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = build_loader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = TemporalEncoder(
            type_vocab_size=len(vocabs[cols.transaction_type]),
            status_vocab_size=len(vocabs[cols.status]),
            channel_vocab_size=len(vocabs[cols.channel]),
            numeric_dim=9,
            hidden_size=args.hidden_size,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            rnn_type=args.rnn_type,
            bidirectional=True,
        ).to(device)

        max_class = max(int(y[tr_idx].sum()), 1)
        pos_weight = torch.tensor((len(tr_idx) - y[tr_idx].sum()) / max_class, dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        best_state = copy.deepcopy(model.state_dict())
        best_score = -float("inf")
        stagnant = 0
        best_val_probs = None

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(
                    numeric=batch["numeric"].to(device),
                    type_ids=batch["type_ids"].to(device),
                    status_ids=batch["status_ids"].to(device),
                    channel_ids=batch["channel_ids"].to(device),
                    lengths=batch["lengths"].to(device),
                )
                targets = batch["target"].to(device)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                train_losses.append(loss.item())

            val_probs_batches: list[np.ndarray] = []
            val_targets_batches: list[np.ndarray] = []
            model.eval()
            with torch.no_grad():
                for batch in valid_loader:
                    logits, _ = model(
                        numeric=batch["numeric"].to(device),
                        type_ids=batch["type_ids"].to(device),
                        status_ids=batch["status_ids"].to(device),
                        channel_ids=batch["channel_ids"].to(device),
                        lengths=batch["lengths"].to(device),
                    )
                    val_probs_batches.append(torch.sigmoid(logits).cpu().numpy())
                    val_targets_batches.append(batch["target"].cpu().numpy())

            val_probs = clamp_probabilities(np.concatenate(val_probs_batches))
            val_targets = np.concatenate(val_targets_batches)
            val_logloss, val_auc, val_score = custom_metric(val_targets, val_probs)
            avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            logging.info(
                "Fold %d | Epoch %d | train_loss=%.6f | val_logloss=%.6f | val_auc=%.6f | val_score=%.6f",
                fold,
                epoch,
                avg_train_loss,
                val_logloss,
                val_auc,
                val_score,
            )

            if val_score > best_score:
                best_score = val_score
                best_state = copy.deepcopy(model.state_dict())
                best_val_probs = val_probs.copy()
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= args.patience:
                    break

        model.load_state_dict(best_state)
        val_embeddings, val_probs_pred, val_ids = extract_embeddings(model, valid_loader, device)
        test_embeddings, test_probs_pred, _ = extract_embeddings(model, test_loader, device)

        oof_embeddings[va_idx] = val_embeddings.astype(np.float32)
        oof_probs[va_idx] = val_probs_pred.astype(np.float32)
        test_embedding_folds.append(test_embeddings.astype(np.float32))
        test_prob_folds.append(test_probs_pred.astype(np.float32))

        fold_metrics.append(
            {
                "fold": fold,
                "logloss": float(log_loss(y[va_idx], clamp_probabilities(val_probs_pred), labels=[0, 1])),
                "roc_auc": float(roc_auc_score(y[va_idx], clamp_probabilities(val_probs_pred))) if len(np.unique(y[va_idx])) > 1 else 0.5,
                "weighted_score": float(custom_metric(y[va_idx], val_probs_pred)[2]),
                "best_epoch_score": float(best_score),
            }
        )

        model_path = output_dir / f"temporal_encoder_fold_{fold}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {
                    "rnn_type": args.rnn_type,
                    "hidden_size": args.hidden_size,
                    "embedding_dim": args.embedding_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "max_seq_len": args.max_seq_len,
                },
                "vocabs": vocabs,
            },
            model_path,
        )

    avg_test_embeddings = np.mean(np.stack(test_embedding_folds, axis=0), axis=0)
    avg_test_probs = np.mean(np.stack(test_prob_folds, axis=0), axis=0)

    train_emb_df = pd.DataFrame(oof_embeddings, columns=get_feature_names(args.embedding_dim))
    train_output = pd.concat(
        [
            train_cohort[[cols.customer_id, cols.target]].reset_index(drop=True),
            train_emb_df,
            pd.DataFrame({"temporal_oof_prob": clamp_probabilities(oof_probs)}),
        ],
        axis=1,
    )

    test_emb_df = pd.DataFrame(avg_test_embeddings, columns=get_feature_names(args.embedding_dim))
    test_output = pd.concat(
        [
            test_cohort[[cols.customer_id]].reset_index(drop=True),
            test_emb_df,
            pd.DataFrame({"temporal_test_prob": clamp_probabilities(avg_test_probs)}),
        ],
        axis=1,
    )

    if len(train_output) != len(train_cohort):
        raise ValueError("Train feature row count does not match train cohort row count.")
    if len(test_output) != len(test_cohort):
        raise ValueError("Test feature row count does not match test cohort row count.")
    if train_output.isna().any().any() or test_output.isna().any().any():
        raise ValueError("NaN values detected in the exported temporal features.")

    train_metric = custom_metric(y, train_output["temporal_oof_prob"].to_numpy())
    metrics_payload = {
        "fold_metrics": fold_metrics,
        "overall_oof": {
            "logloss": train_metric[0],
            "roc_auc": train_metric[1],
            "weighted_score": train_metric[2],
        },
        "config": vars(args),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_temporal_features.csv"
    test_path = output_dir / "test_temporal_features.csv"
    metrics_path = output_dir / "temporal_model_metrics.json"

    train_output.to_csv(train_path, index=False)
    test_output.to_csv(test_path, index=False)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    logging.info("Saved train temporal features to %s", train_path)
    logging.info("Saved test temporal features to %s", test_path)
    logging.info("Saved temporal metrics to %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())