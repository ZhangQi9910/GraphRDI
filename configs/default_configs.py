"""Default configurations for unified GraphRDI training.

The code supports two data modes:
1. random: one labeled file is split into train/val/test by ratio.
2. presplit: explicit train/val/test files are used. This is useful for
   benchmark settings such as ROBIN/PDB where the split is predefined.
"""
from __future__ import annotations

from copy import deepcopy

BASE_CONFIG = {
    # data
    "data_path": None,
    "train_data_path": None,
    "val_data_path": None,
    "test_data_path": None,
    "sheet_name": 0,
    "rna_type": "miRNA",
    "dataset_preset": "custom",
    "split_mode": "random",  # random or presplit
    "run_name": "custom",
    "data_sample_ratio": 1.0,
    "seed": 42,

    # graph / feature extraction
    "max_drug_nodes": 80,
    "max_rna_nodes": 30,
    "rna_graph_mode": "mirna",  # mirna or long
    "cache_dir": "feature_cache",
    "rna_graph_batch_size": 100,
    "max_memory_usage": 0.80,
    "n_jobs": -1,

    # split / dataloader
    "train_ratio": 0.7,
    "val_ratio": 0.2,
    "test_ratio": 0.1,
    "batch_size": 256,
    "num_workers": 4,

    # sampling
    "enforce_balance": True,
    "balance_train": True,
    "balance_val": True,
    "balance_test": True,

    # model dimensions
    "drug_ecfp_dim": 1024,
    "drug_ecfp_hidden_dims": [256],
    "drug_ecfp_output_dim": 64,
    "rna_vec_dim": 174,  # NAC(4) + 3-mer(64) + DPCP(6) + RNA2Vec(100)
    "rna_vec_hidden_dims": [64],
    "rna_vec_output_dim": 32,
    "drug_node_dim": 17,  # atom type one-hot(10) + atom descriptors(7)
    "drug_gat_hidden_dim": 16,
    "drug_gat_output_dim": 32,
    "rna_node_dim": 4,
    "rna_gat_hidden_dim": 8,
    "rna_gat_output_dim": 16,
    "gat_num_heads": 2,
    "fusion_hidden_dims": [128, 64],
    "dropout": 0.5,
    "batch_norm": True,

    # train
    "epochs": 100,
    "patience": 15,
    "min_epochs": 10,
    "lr": 5e-3,
    "weight_decay": 1e-5,
    "grad_clip": 0.8,
    "use_amp": False,
    "use_lr_scheduler": False,
    "save_model_dir": "saved_models",
    "result_dir": "results",
}

RNA_TYPE_OVERRIDES = {
    "mirna": {
        "rna_type": "miRNA",
        "data_path": "data/miRNA_drug_interactions.csv",
        "max_rna_nodes": 30,
        "rna_graph_mode": "mirna",
        "batch_size": 256,
        "rna_gat_hidden_dim": 8,
        "rna_gat_output_dim": 16,
        "use_amp": False,
        "use_lr_scheduler": False,
        "run_name": "miRNA",
    },
    "circrna": {
        "rna_type": "circRNA",
        "data_path": "data/circRNA_drug_interactions.xlsx",
        "max_rna_nodes": 400,
        "rna_graph_mode": "long",
        "batch_size": 64,
        "rna_gat_hidden_dim": 16,
        "rna_gat_output_dim": 32,
        "use_amp": True,
        "use_lr_scheduler": True,
        "run_name": "circRNA",
    },
    "lncrna": {
        "rna_type": "lncRNA",
        "data_path": "data/lncRNA_drug_interactions.xlsx",
        "max_rna_nodes": 400,
        "rna_graph_mode": "long",
        "batch_size": 32,
        "rna_gat_hidden_dim": 16,
        "rna_gat_output_dim": 32,
        "use_amp": True,
        "use_lr_scheduler": True,
        "run_name": "lncRNA",
    },
}

DATASET_PRESETS = {
    "none": {},
    "custom": {},
    "robin": {
        "dataset_preset": "robin",
        "split_mode": "presplit",
        "run_name": "ROBIN",
        "train_data_path": "data/robin/netperturbation/data_train.csv",
        "val_data_path": "data/robin/netperturbation/data_val.csv",
        "test_data_path": "data/robin/netperturbation/data_test.csv",
        "max_rna_nodes": 400,
        "rna_graph_mode": "long",
        "batch_size": 128,
        "rna_gat_hidden_dim": 16,
        "rna_gat_output_dim": 32,
        "use_amp": True,
        "use_lr_scheduler": True,
        "min_epochs": 50,
    },
    "robin_netperturbation": {
        "dataset_preset": "robin_netperturbation",
        "split_mode": "presplit",
        "run_name": "ROBIN_netperturbation",
        "train_data_path": "data/robin/netperturbation/data_train.csv",
        "val_data_path": "data/robin/netperturbation/data_val.csv",
        "test_data_path": "data/robin/netperturbation/data_test.csv",
        "max_rna_nodes": 400,
        "rna_graph_mode": "long",
        "batch_size": 128,
        "rna_gat_hidden_dim": 16,
        "rna_gat_output_dim": 32,
        "use_amp": True,
        "use_lr_scheduler": True,
        "min_epochs": 50,
    },
    "pdb": {
        "dataset_preset": "pdb",
        "split_mode": "presplit",
        "run_name": "PDB",
        "train_data_path": "data/pdb/data_train.csv",
        "val_data_path": "data/pdb/data_val.csv",
        "test_data_path": "data/pdb/data_test.csv",
        "max_rna_nodes": 400,
        "rna_graph_mode": "long",
        "batch_size": 128,
        "rna_gat_hidden_dim": 16,
        "rna_gat_output_dim": 32,
        "use_amp": True,
        "use_lr_scheduler": True,
        "min_epochs": 50,
    },
}


def _normalize_key(value: str) -> str:
    return str(value).lower().replace("_", "").replace("-", "")


def get_config(rna_type: str = "miRNA", dataset_preset: str | None = None) -> dict:
    """Return merged RNA-type and optional dataset-preset config."""
    key = _normalize_key(rna_type)
    if key not in RNA_TYPE_OVERRIDES:
        raise ValueError(f"Unsupported rna_type={rna_type}. Choose from miRNA, circRNA, lncRNA.")
    config = deepcopy(BASE_CONFIG)
    config.update(RNA_TYPE_OVERRIDES[key])
    if dataset_preset:
        preset_key = str(dataset_preset).lower()
        if preset_key not in DATASET_PRESETS:
            raise ValueError(f"Unsupported dataset preset={dataset_preset}. Choose from {sorted(DATASET_PRESETS)}")
        config.update(DATASET_PRESETS[preset_key])
    return config


def available_presets() -> list[str]:
    return sorted(DATASET_PRESETS.keys())
