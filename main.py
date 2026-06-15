"""Unified GraphRDI trainer for miRNA/circRNA/lncRNA and benchmark splits.

Supported modes:
- random: one labeled data file is randomly split into train/val/test.
- presplit: explicit train/val/test files are used, suitable for ROBIN/PDB style benchmarks.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from configs import available_presets, get_config
from feature_extraction import configure_feature_extraction, generate_all_features, load_data
from model import DrugRNAModel, DrugRNADataSet, FeatureProcessor


class BalancedDataset(Dataset):
    """Undersample the larger class to keep a strict 1:1 positive/negative ratio."""

    def __init__(self, base_dataset: Dataset, shuffle: bool = True, seed: int = 42):
        self.base_dataset = base_dataset
        rng = np.random.default_rng(seed)
        pos, neg = [], []
        for i in range(len(base_dataset)):
            label = float(base_dataset[i]["label"].item())
            (pos if label >= 0.5 else neg).append(i)
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError(f"平衡采样失败：正样本={len(pos)}, 负样本={len(neg)}，两类样本都必须存在")
        n = min(len(pos), len(neg))
        selected = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
        if shuffle:
            rng.shuffle(selected)
        self.selected_indices = selected.tolist()
        print(f"平衡后 - 总样本: {len(self.selected_indices)}, 正样本: {n}, 负样本: {n}, 比例: 1:1")

    def __len__(self) -> int:
        return len(self.selected_indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.selected_indices[idx]]


def maybe_balance(dataset: Dataset, enabled: bool, seed: int, name: str) -> Dataset:
    if not enabled:
        return dataset
    try:
        return BalancedDataset(dataset, seed=seed)
    except ValueError as exc:
        print(f"警告：{name} 无法平衡采样，将使用原始划分。原因：{exc}")
        return dataset


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    unique = np.unique(y_true)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(unique) > 1 else 0.5,
        "aupr": average_precision_score(y_true, y_prob) if len(unique) > 1 else 0.5,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch: int, config: dict[str, Any], scaler):
    model.train()
    total_loss = 0.0
    all_true, all_pred, all_prob = [], [], []
    batch_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}", ncols=100, leave=False)
    use_amp = bool(config["use_amp"]) and device.type == "cuda"

    for batch in batch_iter:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["label"]
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch)
            loss = criterion(outputs, labels)
        if torch.isnan(loss):
            print("跳过损失为NaN的批次")
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_loss += float(loss.item())
        with torch.no_grad():
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_true.extend(labels.detach().cpu().numpy())
            all_prob.extend(probs)
            all_pred.extend((probs > 0.5).astype(int))
        batch_iter.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(1, len(train_loader))
    metrics = calculate_metrics(np.asarray(all_true), np.asarray(all_pred), np.asarray(all_prob))
    print(f"[Train] Epoch {epoch + 1:03d} | Loss {avg_loss:.4f} | Acc {metrics['accuracy']:.4f} | F1 {metrics['f1']:.4f}")
    return avg_loss, metrics


def evaluate(model, loader, criterion, device, config: dict[str, Any], split_name: str = "Val"):
    model.eval()
    total_loss = 0.0
    all_true, all_pred, all_prob = [], [], []
    use_amp = bool(config["use_amp"]) and device.type == "cuda"
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch["label"]
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(batch)
                loss = criterion(outputs, labels)
            total_loss += float(loss.item())
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_true.extend(labels.detach().cpu().numpy())
            all_prob.extend(probs)
            all_pred.extend((probs > 0.5).astype(int))
    avg_loss = total_loss / max(1, len(loader))
    metrics = calculate_metrics(np.asarray(all_true), np.asarray(all_pred), np.asarray(all_prob))
    print(f"[{split_name}] Loss {avg_loss:.4f} | Acc {metrics['accuracy']:.4f} | F1 {metrics['f1']:.4f} | AUC {metrics['roc_auc']:.4f} | AUPR {metrics['aupr']:.4f}")
    return avg_loss, metrics


def plot_results(train_losses, val_losses, test_metrics, config: dict[str, Any]) -> None:
    os.makedirs(config["result_dir"], exist_ok=True)
    run_name = config.get("run_name", config.get("rna_type", "run"))
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(config["result_dir"], f"loss_curve_{run_name}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(test_metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title("Test Confusion Matrix")
    plt.savefig(os.path.join(config["result_dir"], f"confusion_matrix_{run_name}.png"), dpi=300)
    plt.close()

    pd.DataFrame({"epoch": range(1, len(train_losses) + 1), "train_loss": train_losses, "val_loss": val_losses}).to_csv(os.path.join(config["result_dir"], f"metrics_{run_name}.csv"), index=False)
    pd.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1", "roc_auc", "aupr"],
        "value": [test_metrics["accuracy"], test_metrics["precision"], test_metrics["recall"], test_metrics["f1"], test_metrics["roc_auc"], test_metrics["aupr"]],
    }).to_csv(os.path.join(config["result_dir"], f"test_metrics_{run_name}.csv"), index=False)


def _make_loaders(train_set, val_set, test_set, config: dict[str, Any]):
    train_set = maybe_balance(train_set, config["enforce_balance"] and config["balance_train"], config["seed"], "训练集")
    val_set = maybe_balance(val_set, config["enforce_balance"] and config["balance_val"], config["seed"] + 1, "验证集")
    test_set = maybe_balance(test_set, config["enforce_balance"] and config["balance_test"], config["seed"] + 2, "测试集")
    persistent = config["num_workers"] > 0
    loader_kwargs = dict(batch_size=config["batch_size"], pin_memory=torch.cuda.is_available(), num_workers=config["num_workers"], persistent_workers=persistent)
    return (
        DataLoader(train_set, shuffle=True, **loader_kwargs),
        DataLoader(val_set, shuffle=False, **loader_kwargs),
        DataLoader(test_set, shuffle=False, **loader_kwargs),
    )


def build_dataloaders_random(df_features: pd.DataFrame, config: dict[str, Any]):
    """Build loaders from one file by random split."""
    processor = FeatureProcessor(config["max_drug_nodes"], config["max_rna_nodes"], config["drug_node_dim"], config["rna_node_dim"])
    train_sample = df_features.iloc[: max(1, int(config["train_ratio"] * len(df_features)))]
    processor.fit(train_sample)
    processed_data = processor.transform(df_features)
    dataset = DrugRNADataSet(processed_data)

    train_size = int(config["train_ratio"] * len(dataset))
    val_size = int(config["val_ratio"] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(f"数据量过少，无法按比例划分：train={train_size}, val={val_size}, test={test_size}")

    train_set_raw, val_set_raw, test_set_raw = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(config["seed"]))
    return _make_loaders(train_set_raw, val_set_raw, test_set_raw, config)


def build_dataloaders_presplit(train_features: pd.DataFrame, val_features: pd.DataFrame, test_features: pd.DataFrame, config: dict[str, Any]):
    """Build loaders from predefined train/val/test splits.

    The scaler/normalizer is fitted only on the training split, then reused for
    validation and test. This avoids validation/test information leakage.
    """
    processor = FeatureProcessor(config["max_drug_nodes"], config["max_rna_nodes"], config["drug_node_dim"], config["rna_node_dim"])
    processor.fit(train_features)
    train_processed = processor.transform(train_features)
    val_processed = processor.transform(val_features)
    test_processed = processor.transform(test_features)
    train_set_raw = DrugRNADataSet(train_processed)
    val_set_raw = DrugRNADataSet(val_processed)
    test_set_raw = DrugRNADataSet(test_processed)
    return _make_loaders(train_set_raw, val_set_raw, test_set_raw, config)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified GraphRDI trainer")
    parser.add_argument("--rna_type", default="miRNA", choices=["miRNA", "circRNA", "lncRNA"], help="RNA configuration type")
    parser.add_argument("--preset", default=None, choices=available_presets(), help="Dataset preset, e.g. robin_netperturbation or pdb")
    parser.add_argument("--split_mode", default=None, choices=["random", "presplit"], help="random: one file split; presplit: train/val/test files")
    parser.add_argument("--data_path", default=None, help="Single data file for random split mode")
    parser.add_argument("--train_data_path", default=None, help="Train file for presplit mode")
    parser.add_argument("--val_data_path", default=None, help="Validation file for presplit mode")
    parser.add_argument("--test_data_path", default=None, help="Test file for presplit mode")
    parser.add_argument("--sheet_name", default=None, help="Excel sheet name or index")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_rna_nodes", type=int, default=None)
    parser.add_argument("--rna_graph_mode", default=None, choices=["mirna", "long"], help="RNA graph construction mode")
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--result_dir", default=None)
    parser.add_argument("--save_model_dir", default=None)
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args) -> dict[str, Any]:
    for key in ["data_path", "train_data_path", "val_data_path", "test_data_path", "split_mode", "cache_dir", "result_dir", "save_model_dir", "run_name"]:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.sheet_name is not None:
        config["sheet_name"] = int(args.sheet_name) if str(args.sheet_name).isdigit() else args.sheet_name
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.max_rna_nodes is not None:
        config["max_rna_nodes"] = args.max_rna_nodes
    if args.rna_graph_mode is not None:
        config["rna_graph_mode"] = args.rna_graph_mode
    if args.sample_ratio is not None:
        config["data_sample_ratio"] = args.sample_ratio
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    return config


def _print_label_distribution(df: pd.DataFrame, name: str) -> None:
    if "interaction" not in df.columns:
        print(f"{name}: 无 interaction 标签列")
        return
    labels = df["interaction"].astype(int).to_numpy()
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    ratio = pos / neg if neg else float("inf")
    print(f"{name} 标签分布 - 正样本: {pos}, 负样本: {neg}, 比例: {ratio:.4f}")


def _load_labeled_data(path: str, config: dict[str, Any], name: str) -> pd.DataFrame | None:
    df = load_data(path, sheet_name=config["sheet_name"], sample_ratio=config["data_sample_ratio"], seed=config["seed"])
    if df is None or df.empty:
        print(f"{name} 数据加载失败或为空")
        return None
    if "interaction" not in df.columns:
        print(f"错误：{name} 必须包含 interaction/label 标签列")
        return None
    _print_label_distribution(df, name)
    return df


def main() -> None:
    args = parse_args()
    config = apply_overrides(get_config(args.rna_type, args.preset), args)
    os.makedirs(config["save_model_dir"], exist_ok=True)
    os.makedirs(config["result_dir"], exist_ok=True)
    configure_feature_extraction(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        config["use_amp"] = False
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    print("\n=== GraphRDI Multi-scenario Trainer ===")
    print(f"RNA类型: {config['rna_type']} | 数据预设: {config.get('dataset_preset')} | 划分模式: {config['split_mode']}")
    print(f"设备: {device} | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    keys = ["data_path", "train_data_path", "val_data_path", "test_data_path", "max_rna_nodes", "rna_graph_mode", "batch_size", "use_amp", "cache_dir", "run_name"]
    print(json.dumps({k: config.get(k) for k in keys}, ensure_ascii=False, indent=2))

    if config["split_mode"] == "presplit":
        missing = [k for k in ["train_data_path", "val_data_path", "test_data_path"] if not config.get(k)]
        if missing:
            print(f"presplit 模式缺少路径: {missing}")
            return
        train_df = _load_labeled_data(config["train_data_path"], config, "训练集")
        val_df = _load_labeled_data(config["val_data_path"], config, "验证集")
        test_df = _load_labeled_data(config["test_data_path"], config, "测试集")
        if train_df is None or val_df is None or test_df is None:
            return

        print("\n=== 特征提取：训练集 ===")
        train_features, rna2vec_model = generate_all_features(train_df, fit_rna2vec=True, force_train_rna2vec=True, use_rna2vec_cache=False)
        print("\n=== 特征提取：验证集 ===")
        val_features, _ = generate_all_features(val_df, rna2vec_model=rna2vec_model, fit_rna2vec=False, use_rna2vec_cache=False)
        print("\n=== 特征提取：测试集 ===")
        test_features, _ = generate_all_features(test_df, rna2vec_model=rna2vec_model, fit_rna2vec=False, use_rna2vec_cache=False)
        if train_features is None or val_features is None or test_features is None:
            print("特征提取失败或某个划分没有有效样本")
            return
        train_loader, val_loader, test_loader = build_dataloaders_presplit(train_features, val_features, test_features, config)
    else:
        raw_df = _load_labeled_data(config["data_path"], config, "总数据集")
        if raw_df is None:
            return
        df_features, _ = generate_all_features(raw_df)
        if df_features is None or df_features.empty:
            print("特征提取失败或没有有效样本")
            return
        train_loader, val_loader, test_loader = build_dataloaders_random(df_features, config)

    print(f"数据集: 训练集={len(train_loader.dataset)} | 验证集={len(val_loader.dataset)} | 测试集={len(test_loader.dataset)}")

    model = DrugRNAModel(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5) if config["use_lr_scheduler"] else None
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["use_amp"]) and device.type == "cuda")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")

    best_val_aupr, early_stop_count = -float("inf"), 0
    train_losses, val_losses = [], []
    best_model_path = os.path.join(config["save_model_dir"], f"best_{config.get('run_name', config['rna_type'])}.pt")

    for epoch in tqdm(range(config["epochs"]), desc="Total Progress", ncols=100):
        train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config, scaler)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, config, split_name="Val")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if scheduler is not None:
            scheduler.step(val_metrics["aupr"])

        if val_metrics["aupr"] > best_val_aupr:
            best_val_aupr = val_metrics["aupr"]
            early_stop_count = 0
            torch.save({"model_state_dict": model.state_dict(), "config": config, "val_metrics": val_metrics}, best_model_path)
            print(f"保存最佳模型: {best_model_path} | Val AUPR={best_val_aupr:.4f}")
        else:
            early_stop_count += 1

        if epoch + 1 >= config["min_epochs"] and early_stop_count >= config["patience"]:
            print(f"早停触发：连续 {config['patience']} 个epoch验证集AUPR未提升")
            break

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    _, test_metrics = evaluate(model, test_loader, criterion, device, config, split_name="Test")
    plot_results(train_losses, val_losses, test_metrics, config)
    print("\n=== 最终测试结果 ===")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "aupr"]:
        print(f"{key}: {test_metrics[key]:.4f}")


if __name__ == "__main__":
    main()
