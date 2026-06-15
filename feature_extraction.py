"""Unified feature extraction for miRNA / circRNA / lncRNA drug-interaction prediction.

This module merges the short-RNA and long-RNA versions. It keeps the long-RNA
version's disk cache, incremental saving, parallel RNA graph generation, memory
monitoring, and tolerant CSV/Excel input handling.
"""
from __future__ import annotations

import gc
import itertools
import os
import pickle
import time
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import psutil
from gensim.models import Word2Vec
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem

from RNA_2D import build_rna_graph, normalize_rna_sequence

FEATURE_CONFIG = {
    "max_rna_nodes": 400,
    "rna_graph_mode": "long",
    "cache_dir": "feature_cache",
    "rna_graph_batch_size": 100,
    "max_memory_usage": 0.80,
    "n_jobs": -1,
}

# In-memory caches. They are loaded by load_all_caches(), usually called from main.py.
drug_feature_cache: dict[str, dict[Any, Any]] = {"ecfp": {}, "graph_feature": {}}
rna_feature_cache: dict[str, dict[Any, Any]] = {
    "nac": {},
    "3mer": {},
    "dpcp": {},
    "rna2vec": {},
    "graph_feature": {},
}


def configure_feature_extraction(config: dict[str, Any]) -> None:
    """Update feature-extraction settings and load caches."""
    FEATURE_CONFIG["max_rna_nodes"] = int(config.get("max_rna_nodes", FEATURE_CONFIG["max_rna_nodes"]))
    FEATURE_CONFIG["rna_graph_mode"] = str(config.get("rna_graph_mode", FEATURE_CONFIG["rna_graph_mode"]))
    FEATURE_CONFIG["cache_dir"] = str(config.get("cache_dir", FEATURE_CONFIG["cache_dir"]))
    FEATURE_CONFIG["rna_graph_batch_size"] = int(config.get("rna_graph_batch_size", FEATURE_CONFIG["rna_graph_batch_size"]))
    FEATURE_CONFIG["max_memory_usage"] = float(config.get("max_memory_usage", FEATURE_CONFIG["max_memory_usage"]))
    FEATURE_CONFIG["n_jobs"] = int(config.get("n_jobs", FEATURE_CONFIG["n_jobs"]))
    os.makedirs(FEATURE_CONFIG["cache_dir"], exist_ok=True)
    load_all_caches()


def get_cache_file_path(feature_type: str) -> str:
    os.makedirs(FEATURE_CONFIG["cache_dir"], exist_ok=True)
    return os.path.join(FEATURE_CONFIG["cache_dir"], f"{feature_type}_cache.pkl")


def load_cache(feature_type: str) -> dict[Any, Any]:
    cache_file = get_cache_file_path(feature_type)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return data
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            print(f"警告：加载 {feature_type} 缓存失败，将使用空缓存。错误：{exc}")
            try:
                os.rename(cache_file, f"{cache_file}.corrupted.{int(time.time())}")
            except OSError:
                pass
    return {}


def save_cache(feature_type: str, cache_data: dict[Any, Any], incremental: bool = True) -> None:
    cache_file = get_cache_file_path(feature_type)
    if incremental and os.path.exists(cache_file):
        existing_cache = load_cache(feature_type)
        existing_cache.update(cache_data)
        cache_data = existing_cache
    try:
        temp_file = f"{cache_file}.tmp"
        with open(temp_file, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_file, cache_file)
        print(f"已保存 {feature_type} 缓存，共 {len(cache_data)} 条记录")
    except OSError as exc:
        print(f"保存 {feature_type} 缓存失败：{exc}")


def load_all_caches() -> None:
    print("正在加载已有的特征缓存...")
    drug_feature_cache["ecfp"] = load_cache("drug_ecfp")
    drug_feature_cache["graph_feature"] = load_cache("drug_graph")
    rna_feature_cache["nac"] = load_cache("rna_nac")
    rna_feature_cache["3mer"] = load_cache("rna_3mer")
    rna_feature_cache["dpcp"] = load_cache("rna_dpcp")
    rna_feature_cache["rna2vec"] = load_cache("rna_rna2vec")
    rna_feature_cache["graph_feature"] = load_cache(f"rna_graph_{FEATURE_CONFIG['rna_graph_mode']}")
    print(
        f"加载完成 - 药物ECFP缓存: {len(drug_feature_cache['ecfp'])}条, "
        f"RNA图特征缓存({FEATURE_CONFIG['rna_graph_mode']}): {len(rna_feature_cache['graph_feature'])}条"
    )


def save_all_caches() -> None:
    print("\n=== 保存特征缓存 ===")
    save_cache("drug_ecfp", drug_feature_cache["ecfp"])
    save_cache("drug_graph", drug_feature_cache["graph_feature"])
    save_cache("rna_nac", rna_feature_cache["nac"])
    save_cache("rna_3mer", rna_feature_cache["3mer"])
    save_cache("rna_dpcp", rna_feature_cache["dpcp"])
    save_cache("rna_rna2vec", rna_feature_cache["rna2vec"])
    save_cache(f"rna_graph_{FEATURE_CONFIG['rna_graph_mode']}", rna_feature_cache["graph_feature"])


# ----------------------
# Data loading and column normalization
# ----------------------
COLUMN_ALIASES = {
    "smiles": ["smiles", "sm_smiles", "SMILES", "drug_smiles", "Drug_SMILES", "canonical_smiles"],
    "Sequence": ["Sequence", "sequence", "rna_sequence", "RNA_sequence", "rna_seq", "seq", "RNA"],
    "interaction": ["interaction", "label", "Label", "y", "target", "class", "binding", "bind"],
}


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column names to smiles / Sequence / interaction."""
    rename_map: dict[str, str] = {}
    missing_required: list[str] = []
    for target, aliases in COLUMN_ALIASES.items():
        found = _find_column(df, aliases)
        if found is None:
            if target in {"smiles", "Sequence"}:
                missing_required.append(target)
        else:
            rename_map[found] = target

    if missing_required:
        raise ValueError(
            f"数据缺少必要列: {missing_required}。可识别的列名包括: "
            f"smiles/sm_smiles/drug_smiles，Sequence/rna_sequence/rna_seq，interaction/label。"
        )

    df = df.rename(columns=rename_map).copy()
    if "interaction" in df.columns:
        df["interaction"] = pd.to_numeric(df["interaction"], errors="coerce")
        df = df.dropna(subset=["interaction"])
        df["interaction"] = (df["interaction"].astype(float) > 0).astype(int)
    return df


def load_data(file_path: str, sheet_name: int | str = 0, sample_ratio: float = 1.0, seed: int = 42) -> pd.DataFrame | None:
    """Load CSV/XLS/XLSX data and normalize columns.

    The returned DataFrame uses canonical columns: smiles, Sequence, interaction.
    """
    try:
        file_path = str(file_path)
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

        print(f"成功读取数据: {file_path}，共 {df.shape[0]} 行，{df.shape[1]} 列")
        df = normalize_columns(df)

        for col in ["smiles", "Sequence"]:
            df[col] = df[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
        initial_count = len(df)
        required = ["smiles", "Sequence"] + (["interaction"] if "interaction" in df.columns else [])
        df = df.dropna(subset=required).copy()
        if len(df) < initial_count:
            print(f"移除了 {initial_count - len(df)} 行缺失 smiles/Sequence/interaction 的数据")

        df["Sequence"] = df["Sequence"].map(lambda x: normalize_rna_sequence(x))
        df = df[df["Sequence"].str.len() > 0].copy()

        if 0 < sample_ratio < 1.0:
            np.random.seed(seed)
            if "interaction" in df.columns and df["interaction"].nunique() > 1:
                df = df.groupby("interaction", group_keys=False).apply(lambda x: x.sample(frac=sample_ratio, random_state=seed))
            else:
                df = df.sample(frac=sample_ratio, random_state=seed)
            print(f"采样后数据形状: {df.shape}，采样比例 {sample_ratio:.0%}")

        print(f"药物序列重复情况: {df['smiles'].nunique()} 个独特 SMILES，共 {len(df)} 条记录")
        print(f"RNA序列重复情况: {df['Sequence'].nunique()} 个独特 RNA 序列，共 {len(df)} 条记录")
        if "interaction" in df.columns:
            print("标签分布:")
            print(df["interaction"].value_counts().sort_index().to_string())
        return df.reset_index(drop=True)
    except Exception as exc:
        print(f"读取数据失败: {exc}")
        return None


# ----------------------
# Drug features
# ----------------------
def generate_ecfp(smiles: str, radius: int = 2, nBits: int = 1024) -> np.ndarray | None:
    if not isinstance(smiles, str) or pd.isna(smiles) or smiles.strip() == "":
        return None
    if smiles in drug_feature_cache["ecfp"]:
        return drug_feature_cache["ecfp"][smiles]
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            drug_feature_cache["ecfp"][smiles] = None
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                Chem.SanitizeMol(
                    mol,
                    Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                    | Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
                )
            except Exception as exc:
                print(f"ECFP生成时分子验证失败(smiles={smiles[:40]}...): {exc}")
                drug_feature_cache["ecfp"][smiles] = None
                return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.asarray(fp, dtype=np.float32)
        drug_feature_cache["ecfp"][smiles] = arr
        return arr
    except Exception as exc:
        print(f"生成ECFP失败(smiles={smiles[:40]}...): {exc}")
        drug_feature_cache["ecfp"][smiles] = None
        return None


def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    atom_types = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "OTHER"]
    type_onehot = [0.0] * len(atom_types)
    symbol = atom.GetSymbol()
    type_onehot[atom_types.index(symbol) if symbol in atom_types else -1] = 1.0
    features = [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetTotalValence()),
        float(atom.GetFormalCharge()),
        float(atom.GetNumRadicalElectrons()),
        1.0 if atom.GetIsAromatic() else 0.0,
        float(int(atom.GetHybridization())),
    ]
    return np.asarray(type_onehot + features, dtype=np.float32)


def generate_drug_graph_feature(smiles: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not isinstance(smiles, str) or pd.isna(smiles) or smiles.strip() == "":
        return None, None
    if smiles in drug_feature_cache["graph_feature"]:
        return drug_feature_cache["graph_feature"][smiles]
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            drug_feature_cache["graph_feature"][smiles] = (None, None)
            return None, None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                flags = (
                    Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                    | Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                    | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                    | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                )
                Chem.SanitizeMol(mol, flags)
            except Exception:
                drug_feature_cache["graph_feature"][smiles] = (None, None)
                return None, None

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            drug_feature_cache["graph_feature"][smiles] = (None, None)
            return None, None
        adj = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.SINGLE:
                weight = 1.0
            elif bond_type == Chem.BondType.DOUBLE:
                weight = 2.0
            elif bond_type == Chem.BondType.TRIPLE:
                weight = 3.0
            elif bond_type == Chem.BondType.AROMATIC:
                weight = 1.5
            else:
                weight = 1.0
            adj[i, j] = adj[j, i] = weight
        nodes = np.vstack([get_atom_features(atom) for atom in mol.GetAtoms()]).astype(np.float32)
        drug_feature_cache["graph_feature"][smiles] = (adj, nodes)
        return adj, nodes
    except Exception as exc:
        print(f"生成药物图特征失败(smiles={smiles[:40]}...): {exc}")
        drug_feature_cache["graph_feature"][smiles] = (None, None)
        return None, None


# ----------------------
# RNA vector features
# ----------------------
def calculate_nac(sequence: str) -> list[float] | None:
    if sequence in rna_feature_cache["nac"]:
        return rna_feature_cache["nac"][sequence]
    seq = normalize_rna_sequence(sequence)
    if not seq:
        rna_feature_cache["nac"][sequence] = None
        return None
    total = len(seq)
    result = [seq.count(base) / total for base in ["A", "U", "C", "G"]]
    rna_feature_cache["nac"][sequence] = result
    return result


def calculate_3mer(sequence: str) -> list[float] | None:
    if sequence in rna_feature_cache["3mer"]:
        return rna_feature_cache["3mer"][sequence]
    seq = normalize_rna_sequence(sequence)
    if len(seq) < 3:
        rna_feature_cache["3mer"][sequence] = None
        return None
    if not hasattr(calculate_3mer, "all_3mers"):
        calculate_3mer.all_3mers = ["".join(c) for c in itertools.product(["A", "U", "C", "G"], repeat=3)]
    counts = defaultdict(int)
    total_valid = 0
    for i in range(len(seq) - 2):
        kmer = seq[i : i + 3]
        if "N" in kmer:
            continue
        counts[kmer] += 1
        total_valid += 1
    if total_valid == 0:
        rna_feature_cache["3mer"][sequence] = None
        return None
    result = [counts[k] / total_valid for k in calculate_3mer.all_3mers]
    rna_feature_cache["3mer"][sequence] = result
    return result


def calculate_dpcp(sequence: str) -> list[float] | None:
    if sequence in rna_feature_cache["dpcp"]:
        return rna_feature_cache["dpcp"][sequence]
    seq = normalize_rna_sequence(sequence)
    if len(seq) < 2:
        rna_feature_cache["dpcp"][sequence] = None
        return None
    dinuc_properties = {
        "AA": [0.5773884923447732, 0.6531915653378907, 0.6124592000985356, 0.8402684612384332, 0.5856582729115565, 0.5476708282666789],
        "AU": [0.7512077598863804, 0.6036675879079278, 0.6737051546096536, 0.39069870063063133, 1.0, 0.76847598772376],
        "AG": [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764, 0.45903777008667923],
        "AC": [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944, 0.7467063799220581],
        "UA": [0.3539063797840531, 0.15795248106354978, 0.48996729107629966, 0.1795369895818257, 0.3059118434042811, 0.32686549630327577],
        "UU": [0.5773884923447732, 0.6531915653378907, 0.0, 0.8402684612384332, 0.5856582729115565, 0.5476708282666789],
        "UG": [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195, 0.3501900760908136],
        "UC": [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026, 0.6891727614587756],
        "GA": [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026, 0.6891727614587756],
        "GU": [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944, 0.7467063799220581],
        "GG": [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261, 0.6083143907016332],
        "GC": [0.5525570698352168, 0.6036675879079278, 0.7961968911255676, 0.5064970193495165, 0.6780274730118172, 0.8400043540595654],
        "CA": [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195, 0.3501900760908136],
        "CU": [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764, 0.45903777008667923],
        "CG": [0.2794124572680277, 0.3560480457707574, 0.48996729107629966, 0.4247569687810134, 0.5170412957708868, 0.32686549630327577],
        "CC": [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261, 0.6083143907016332],
    }
    properties = [dinuc_properties[seq[i : i + 2]] for i in range(len(seq) - 1) if seq[i : i + 2] in dinuc_properties]
    if not properties:
        rna_feature_cache["dpcp"][sequence] = None
        return None
    result = np.mean(properties, axis=0).astype(float).tolist()
    rna_feature_cache["dpcp"][sequence] = result
    return result


def train_rna2vec_model(sequences: list[str], embedding_size: int = 100, window: int = 5, min_count: int = 1) -> Word2Vec:
    unique_sequences = list({normalize_rna_sequence(seq) for seq in sequences if isinstance(seq, str) and len(seq) >= 3})
    print(f"使用 {len(unique_sequences)} 个独特序列训练 RNA2Vec 模型")
    corpus = []
    for seq in unique_sequences:
        kmers = [seq[i : i + 3] for i in range(len(seq) - 2) if "N" not in seq[i : i + 3]]
        if kmers:
            corpus.append(kmers)
    if not corpus:
        raise ValueError("没有足够的有效RNA k-mer用于训练RNA2Vec")
    return Word2Vec(sentences=corpus, vector_size=embedding_size, window=window, min_count=min_count, workers=4)


def get_rna2vec_embedding(sequence: str, model: Word2Vec, k: int = 3, use_cache: bool = True) -> list[float] | None:
    if use_cache and sequence in rna_feature_cache["rna2vec"]:
        return rna_feature_cache["rna2vec"][sequence]
    seq = normalize_rna_sequence(sequence)
    if len(seq) < k:
        rna_feature_cache["rna2vec"][sequence] = None
        return None
    vectors = [model.wv[seq[i : i + k]] for i in range(len(seq) - k + 1) if "N" not in seq[i : i + k] and seq[i : i + k] in model.wv]
    if not vectors:
        rna_feature_cache["rna2vec"][sequence] = None
        return None
    result = np.mean(vectors, axis=0).astype(float).tolist()
    rna_feature_cache["rna2vec"][sequence] = result
    return result


# ----------------------
# RNA graph features
# ----------------------
def process_single_rna_graph(sequence: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if sequence in rna_feature_cache["graph_feature"]:
        return rna_feature_cache["graph_feature"][sequence]
    try:
        result = build_rna_graph(
            sequence,
            max_nodes=FEATURE_CONFIG["max_rna_nodes"],
            mode=FEATURE_CONFIG["rna_graph_mode"],
        )
        return result
    except Exception as exc:
        print(f"[RNA图特征] 处理RNA序列失败: {str(sequence)[:30]}... | 错误: {exc}")
        return None, None


def generate_rna_graph_features_parallel(sequences: list[str]) -> list[tuple[np.ndarray | None, np.ndarray | None]]:
    print(f"使用并行处理生成RNA图特征，共 {len(sequences)} 个序列")
    if not sequences:
        return []
    results: list[Any] = [None] * len(sequences)
    batch_size = max(1, int(FEATURE_CONFIG["rna_graph_batch_size"]))
    num_batches = (len(sequences) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        while True:
            memory_usage = psutil.virtual_memory().percent / 100
            if memory_usage < FEATURE_CONFIG["max_memory_usage"]:
                break
            print(f"内存使用率过高 ({memory_usage:.1%})，等待10秒...")
            time.sleep(10)

        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start:end]
        print(f"处理RNA图批次 {batch_idx + 1}/{num_batches}: {start}-{end}")
        batch_results = Parallel(n_jobs=FEATURE_CONFIG["n_jobs"], verbose=0)(
            delayed(process_single_rna_graph)(seq) for seq in batch_sequences
        )
        for offset, (seq, result) in enumerate(zip(batch_sequences, batch_results)):
            results[start + offset] = result
            rna_feature_cache["graph_feature"][seq] = result
        del batch_sequences, batch_results
        gc.collect()
    return results


# ----------------------
# Orchestration
# ----------------------
def _is_valid_feature(feature: Any) -> bool:
    if feature is None:
        return False
    if isinstance(feature, np.ndarray):
        return feature.size > 0 and not np.isnan(feature).any()
    if isinstance(feature, (list, tuple)):
        if len(feature) == 0:
            return False
        try:
            return not pd.isna(feature).any()
        except Exception:
            return True
    return not pd.isna(feature)


def generate_all_features(
    df: pd.DataFrame,
    rna2vec_model: Word2Vec | None = None,
    fit_rna2vec: bool = True,
    force_train_rna2vec: bool = False,
    use_rna2vec_cache: bool = True,
) -> tuple[pd.DataFrame | None, Word2Vec | None]:
    """Generate all model features with incremental cache reuse.

    For predefined train/val/test benchmark splits, pass the RNA2Vec model
    learned on the training split into validation/test calls and set
    fit_rna2vec=False. This keeps all splits in one embedding space.
    """
    if df is None or df.empty:
        print("没有数据可处理")
        return None, None
    df = normalize_columns(df).copy()
    print(f"\n=== 特征提取开始 - 初始样本数: {df.shape[0]} ===")

    unique_smiles = df["smiles"].dropna().unique().tolist()
    unique_sequences = df["Sequence"].dropna().unique().tolist()
    print(f"数据集中包含 {len(unique_smiles)} 个独特药物和 {len(unique_sequences)} 个独特RNA序列")

    print("\n=== 生成药物ECFP指纹 ===")
    for smiles in unique_smiles:
        if smiles not in drug_feature_cache["ecfp"]:
            generate_ecfp(smiles)
    df["ecfp"] = df["smiles"].map(lambda x: drug_feature_cache["ecfp"].get(x))

    print("\n=== 生成药物图特征 ===")
    for smiles in unique_smiles:
        if smiles not in drug_feature_cache["graph_feature"]:
            generate_drug_graph_feature(smiles)
    drug_graph_values = df["smiles"].map(lambda x: drug_feature_cache["graph_feature"].get(x, (None, None))).tolist()
    df[["drug_adj_matrix", "drug_atom_features"]] = pd.DataFrame(drug_graph_values, index=df.index)

    print("\n=== 生成RNA NAC / 3-mer / DPCP特征 ===")
    for seq in unique_sequences:
        if seq not in rna_feature_cache["nac"]:
            calculate_nac(seq)
        if seq not in rna_feature_cache["3mer"]:
            calculate_3mer(seq)
        if seq not in rna_feature_cache["dpcp"]:
            calculate_dpcp(seq)
    df["nac"] = df["Sequence"].map(lambda x: rna_feature_cache["nac"].get(x))
    df["3mer"] = df["Sequence"].map(lambda x: rna_feature_cache["3mer"].get(x))
    df["dpcp"] = df["Sequence"].map(lambda x: rna_feature_cache["dpcp"].get(x))

    print("\n=== 处理RNA2Vec特征 ===")
    valid_sequences = [seq for seq in unique_sequences if isinstance(seq, str) and len(seq) >= 3]
    if rna2vec_model is None and (fit_rna2vec or force_train_rna2vec):
        rna2vec_model = train_rna2vec_model(valid_sequences)

    if rna2vec_model is None:
        # Fallback for compatibility: rely only on existing cache.
        print("警告：未提供RNA2Vec模型，将仅使用已有缓存；未缓存序列会被过滤。")
    else:
        if force_train_rna2vec or not use_rna2vec_cache:
            need_rna2vec = valid_sequences
        else:
            need_rna2vec = [seq for seq in valid_sequences if seq not in rna_feature_cache["rna2vec"]]
        for seq in need_rna2vec:
            get_rna2vec_embedding(seq, rna2vec_model, use_cache=(use_rna2vec_cache and not force_train_rna2vec))
    df["rna2vec"] = df["Sequence"].map(lambda x: rna_feature_cache["rna2vec"].get(x))

    print("\n=== 生成RNA图特征 ===")
    need_graph = [seq for seq in unique_sequences if seq not in rna_feature_cache["graph_feature"]]
    if need_graph:
        generate_rna_graph_features_parallel(need_graph)
    graph_values = df["Sequence"].map(lambda x: rna_feature_cache["graph_feature"].get(x, (None, None))).tolist()
    df[["rna_adj_matrix", "rna_node_onehot"]] = pd.DataFrame(graph_values, index=df.index)

    save_all_caches()

    print("\n=== 过滤无效样本 ===")
    required_features = [
        "ecfp",
        "drug_adj_matrix",
        "drug_atom_features",
        "nac",
        "3mer",
        "dpcp",
        "rna2vec",
        "rna_adj_matrix",
        "rna_node_onehot",
    ]
    initial_count = len(df)
    for feature_name in required_features:
        missing = sum(not _is_valid_feature(x) for x in df[feature_name])
        print(f"  {feature_name}: 缺失/无效 {missing}")
        df = df[df[feature_name].map(_is_valid_feature)].copy()

    print(f"过滤后保留 {len(df)} 条样本 (原始 {initial_count} 条)")
    if df.empty:
        return None, rna2vec_model
    return df.reset_index(drop=True), rna2vec_model
