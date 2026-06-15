"""Unified GraphRDI model components."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class FeatureProcessor:
    """Feature standardization and graph padding/truncation."""

    def __init__(self, max_drug_nodes: int = 80, max_rna_nodes: int = 400, drug_node_dim: int = 17, rna_node_dim: int = 4):
        self.max_drug_nodes = max_drug_nodes
        self.max_rna_nodes = max_rna_nodes
        self.drug_node_dim = drug_node_dim
        self.rna_node_dim = rna_node_dim
        self.drug_ecfp_mean: np.ndarray | None = None
        self.drug_ecfp_std: np.ndarray | None = None
        self.rna_feat_mean: np.ndarray | None = None
        self.rna_feat_std: np.ndarray | None = None

    @staticmethod
    def _as_float_array(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32)
            if arr.size == 0 or np.isnan(arr).any():
                return None
            return arr
        except Exception:
            return None

    def pad_graph(self, adj_matrix: Any, node_features: Any, max_nodes: int, fixed_feat_dim: int):
        adj = self._as_float_array(adj_matrix)
        nodes = self._as_float_array(node_features)
        if adj is None or nodes is None or adj.ndim != 2:
            n = 0
        else:
            n = min(adj.shape[0], nodes.shape[0], max_nodes)

        adj_pad = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        feat_pad = np.zeros((max_nodes, fixed_feat_dim), dtype=np.float32)
        mask = np.zeros(max_nodes, dtype=np.float32)

        if n > 0:
            adj_pad[:n, :n] = adj[:n, :n]
            if nodes.ndim == 1:
                nodes = nodes.reshape(-1, 1)
            current_feat_dim = min(nodes.shape[1], fixed_feat_dim)
            feat_pad[:n, :current_feat_dim] = nodes[:n, :current_feat_dim]
            mask[:n] = 1.0
        return adj_pad, feat_pad, mask

    def _concat_rna_vector(self, row: pd.Series) -> np.ndarray | None:
        parts = [self._as_float_array(row.get("nac")), self._as_float_array(row.get("3mer")), self._as_float_array(row.get("dpcp")), self._as_float_array(row.get("rna2vec"))]
        if any(part is None for part in parts):
            return None
        return np.concatenate(parts).astype(np.float32)

    def fit(self, df: pd.DataFrame) -> None:
        valid_ecfps = [self._as_float_array(x) for x in df["ecfp"].tolist()]
        valid_ecfps = [x for x in valid_ecfps if x is not None]
        if not valid_ecfps:
            raise ValueError("无法fit FeatureProcessor：没有有效的ECFP特征")
        drug_ecfp = np.vstack(valid_ecfps).astype(np.float32)
        self.drug_ecfp_mean = np.mean(drug_ecfp, axis=0)
        self.drug_ecfp_std = np.maximum(np.std(drug_ecfp, axis=0), 1e-6)

        rna_feat_list = []
        for _, row in df.iterrows():
            rna_feat = self._concat_rna_vector(row)
            if rna_feat is not None:
                rna_feat_list.append(rna_feat)
        if not rna_feat_list:
            raise ValueError("无法fit FeatureProcessor：没有有效的RNA向量特征")
        rna_feat = np.vstack(rna_feat_list).astype(np.float32)
        self.rna_feat_mean = np.mean(rna_feat, axis=0)
        self.rna_feat_std = np.maximum(np.std(rna_feat, axis=0), 1e-6)

    def transform(self, df: pd.DataFrame) -> list[dict[str, np.ndarray | np.float32]]:
        if self.drug_ecfp_mean is None or self.rna_feat_mean is None:
            raise RuntimeError("FeatureProcessor must be fitted before transform().")

        processed_data: list[dict[str, np.ndarray | np.float32]] = []
        for idx, row in df.iterrows():
            try:
                drug_ecfp = self._as_float_array(row["ecfp"])
                rna_feat = self._concat_rna_vector(row)
                if drug_ecfp is None or rna_feat is None:
                    continue
                if len(drug_ecfp) != len(self.drug_ecfp_mean):
                    drug_ecfp = np.pad(drug_ecfp, (0, max(0, len(self.drug_ecfp_mean) - len(drug_ecfp))))[: len(self.drug_ecfp_mean)]
                drug_ecfp = np.clip((drug_ecfp - self.drug_ecfp_mean) / self.drug_ecfp_std, -5.0, 5.0).astype(np.float32)
                rna_feat = np.clip((rna_feat - self.rna_feat_mean) / self.rna_feat_std, -5.0, 5.0).astype(np.float32)

                drug_adj, drug_nodes, drug_mask = self.pad_graph(row["drug_adj_matrix"], row["drug_atom_features"], self.max_drug_nodes, self.drug_node_dim)
                rna_adj, rna_nodes, rna_mask = self.pad_graph(row["rna_adj_matrix"], row["rna_node_onehot"], self.max_rna_nodes, self.rna_node_dim)

                processed_data.append(
                    {
                        "drug_ecfp": drug_ecfp.astype(np.float32),
                        "drug_adj": drug_adj.astype(np.float32),
                        "drug_nodes": drug_nodes.astype(np.float32),
                        "drug_mask": drug_mask.astype(np.float32),
                        "rna_feat": rna_feat.astype(np.float32),
                        "rna_adj": rna_adj.astype(np.float32),
                        "rna_nodes": rna_nodes.astype(np.float32),
                        "rna_mask": rna_mask.astype(np.float32),
                        "label": np.float32(row["interaction"]),
                    }
                )
            except Exception as exc:
                print(f"处理样本 {idx} 时出错: {exc}")
        print(f"特征预处理完成，保留 {len(processed_data)} 条有效数据")
        return processed_data


class GATNet(nn.Module):
    """Single-layer GAT graph encoder with global mean pooling."""

    def __init__(self, input_dim: int, output_dim: int = 64, num_heads: int = 2, dropout: float = 0.2, batch_norm: bool = False):
        super().__init__()
        self.conv = GATConv(input_dim, output_dim, heads=num_heads, dropout=dropout)
        self.out_proj = nn.Linear(output_dim * num_heads, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.output_dim = output_dim
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, mask: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0)
        x = self.dropout(x)
        x = F.leaky_relu(self.conv(x, edge_index), negative_slope=0.1)
        x = self.out_proj(x)
        if self.batch_norm is not None and x.shape[0] > 1:
            x = self.batch_norm(x)
        x = x * mask.unsqueeze(-1)
        return global_mean_pool(x, batch=batch)


class FCNN(nn.Module):
    """Fully connected encoder."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.2, batch_norm: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            linear = nn.Linear(prev_dim, dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.extend([nn.LeakyReLU(0.1, inplace=True), nn.Dropout(dropout)])
            prev_dim = dim
        final_linear = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_uniform_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        layers.append(final_linear)
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(self.model(torch.nan_to_num(x, nan=0.0)), nan=0.0)


class DrugRNAModel(nn.Module):
    """Drug-RNA interaction model: ECFP + drug graph + RNA vector + RNA graph."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.drug_ecfp_fcnn = FCNN(config["drug_ecfp_dim"], config["drug_ecfp_hidden_dims"], config["drug_ecfp_output_dim"], config["dropout"], config["batch_norm"])
        self.rna_vec_fcnn = FCNN(config["rna_vec_dim"], config["rna_vec_hidden_dims"], config["rna_vec_output_dim"], config["dropout"], config["batch_norm"])
        self.drug_gat = GATNet(config["drug_node_dim"], config["drug_gat_output_dim"], config["gat_num_heads"], config["dropout"], config["batch_norm"])
        self.rna_gat = GATNet(config["rna_node_dim"], config["rna_gat_output_dim"], config["gat_num_heads"], config["dropout"], config["batch_norm"])
        fusion_input_dim = config["drug_ecfp_output_dim"] + config["rna_vec_output_dim"] + config["drug_gat_output_dim"] + config["rna_gat_output_dim"]
        self.fusion_mlp = FCNN(fusion_input_dim, config["fusion_hidden_dims"], 1, config["dropout"], config["batch_norm"])
        self.global_dropout = nn.Dropout(config["dropout"] * 0.3)

    @staticmethod
    def batch_adj_to_edge_index(adj_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_nodes = adj_matrix.shape[0], adj_matrix.shape[1]
        edge_indices = []
        max_edges_per_graph = min(max_nodes * 10, 4000)
        for i in range(batch_size):
            edges = torch.nonzero(adj_matrix[i] > 0, as_tuple=False).t().contiguous()
            if edges.shape[1] > max_edges_per_graph:
                chosen = torch.randperm(edges.shape[1], device=adj_matrix.device)[:max_edges_per_graph]
                edges = edges[:, chosen]
            if edges.numel() > 0:
                edge_indices.append(edges + i * max_nodes)
        node_batch = torch.repeat_interleave(torch.arange(batch_size, device=adj_matrix.device), max_nodes)
        if edge_indices:
            return torch.cat(edge_indices, dim=1).long(), node_batch.long()
        return torch.zeros((2, 0), dtype=torch.long, device=adj_matrix.device), node_batch.long()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = {key: torch.nan_to_num(value, nan=0.0) for key, value in batch.items()}
        batch_size = batch["drug_ecfp"].shape[0]

        drug_ecfp_feat = self.drug_ecfp_fcnn(batch["drug_ecfp"])
        rna_vec_feat = self.rna_vec_fcnn(batch["rna_feat"])

        drug_edge_index, drug_node_batch = self.batch_adj_to_edge_index(batch["drug_adj"])
        drug_nodes = batch["drug_nodes"].reshape(-1, batch["drug_nodes"].shape[-1])
        drug_mask = batch["drug_mask"].reshape(-1)
        drug_gat_feat = self.drug_gat(drug_nodes, drug_edge_index, drug_mask, drug_node_batch)
        if drug_gat_feat.shape[0] != batch_size:
            drug_gat_feat = torch.zeros(batch_size, self.drug_gat.output_dim, device=drug_nodes.device)

        rna_edge_index, rna_node_batch = self.batch_adj_to_edge_index(batch["rna_adj"])
        rna_nodes = batch["rna_nodes"].reshape(-1, batch["rna_nodes"].shape[-1])
        rna_mask = batch["rna_mask"].reshape(-1)
        rna_gat_feat = self.rna_gat(rna_nodes, rna_edge_index, rna_mask, rna_node_batch)
        if rna_gat_feat.shape[0] != batch_size:
            rna_gat_feat = torch.zeros(batch_size, self.rna_gat.output_dim, device=rna_nodes.device)

        combined = torch.cat([drug_ecfp_feat, rna_vec_feat, drug_gat_feat, rna_gat_feat], dim=1)
        return self.fusion_mlp(self.global_dropout(combined)).squeeze(1)


class DrugRNADataSet(torch.utils.data.Dataset):
    """Torch Dataset wrapper for processed feature dictionaries."""

    def __init__(self, processed_data: list[dict[str, np.ndarray | np.float32]]):
        if not processed_data:
            raise ValueError("processed_data为空，无法创建数据集")
        self.data = processed_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.data[idx]
        return {
            "drug_ecfp": torch.tensor(sample["drug_ecfp"], dtype=torch.float32),
            "drug_adj": torch.tensor(sample["drug_adj"], dtype=torch.float32),
            "drug_nodes": torch.tensor(sample["drug_nodes"], dtype=torch.float32),
            "drug_mask": torch.tensor(sample["drug_mask"], dtype=torch.float32),
            "rna_feat": torch.tensor(sample["rna_feat"], dtype=torch.float32),
            "rna_adj": torch.tensor(sample["rna_adj"], dtype=torch.float32),
            "rna_nodes": torch.tensor(sample["rna_nodes"], dtype=torch.float32),
            "rna_mask": torch.tensor(sample["rna_mask"], dtype=torch.float32),
            "label": torch.tensor(sample["label"], dtype=torch.float32),
        }
