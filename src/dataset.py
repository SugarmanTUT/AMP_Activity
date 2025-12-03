import torch
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import config

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import standard_aa_names

# --- 全局常量 (保持不变) ---
STANDARD_AA_3_LETTERS = set(standard_aa_names)
HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TRP', 'MET'}
POSITIVE_RESIDUES = {'ARG', 'LYS', 'HIS'}
NEGATIVE_RESIDUES = {'ASP', 'GLU'}
AA_TO_INDEX = {aa: i for i, aa in enumerate(sorted(list(STANDARD_AA_3_LETTERS)))}
NUM_AMINO_ACIDS = len(AA_TO_INDEX)


# --- Helper Classes and Functions (保持不变) ---
class GaussianDistance(torch.nn.Module):
    def __init__(self, start=0.0, stop=15.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def get_sidechain_centroid(residue):
    sidechain_atoms = [atom for atom in residue.get_atoms() if atom.get_id() not in ['N', 'CA', 'C', 'O', 'H']]
    if not sidechain_atoms:
        if 'CA' in residue:
            return residue['CA'].get_coord()
        else:
            return None
    coords = [atom.get_coord() for atom in sidechain_atoms]
    return np.mean(coords, axis=0)


def parse_pdb_to_chemical_graph(pdb_path, max_nodes=None, is_train=False):
    # 此函数保持原样，不需要修改
    gbf = GaussianDistance(num_gaussians=50)
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("peptide", pdb_path)
    except Exception:
        return None
    try:
        model = structure[0]
        chain = next(model.get_chains())
        residues = [res for res in chain.get_residues() if
                    res.get_resname() in STANDARD_AA_3_LETTERS and 'CA' in res and 'N' in res and 'C' in res]
    except Exception:
        return None
    if not residues: return None
    if max_nodes and len(residues) > max_nodes: residues = residues[:max_nodes]
    num_residues = len(residues)
    pos_coords, scalar_features, vector_features = [], [], []
    for res in residues:
        ca_coord, n_coord, c_coord = res['CA'].get_coord(), res['N'].get_coord(), res['C'].get_coord()
        pos_coords.append(ca_coord)
        aa_type = res.get_resname()
        aa_index = AA_TO_INDEX.get(aa_type, -1)
        if aa_index == -1: return None
        scalar_features.append(F.one_hot(torch.tensor(aa_index), num_classes=NUM_AMINO_ACIDS))
        v_n, v_c = n_coord - ca_coord, c_coord - ca_coord
        vector_features.append(np.stack([v_n, v_c], axis=0))
    pos, node_s, node_v = torch.tensor(np.array(pos_coords), dtype=torch.float), torch.stack(scalar_features,
                                                                                             dim=0).float(), torch.tensor(
        np.array(vector_features), dtype=torch.float)
    edges_with_attrs = set()
    for i in range(num_residues):
        for j in range(i, num_residues):
            dist = np.linalg.norm(pos_coords[i] - pos_coords[j])
            if i != j and dist < config.INTERACTION_THRESHOLDS['proximity']: edges_with_attrs.add(
                (i, j, config.RELATION_TYPES['proximity'], dist))
            if j == i + 1: edges_with_attrs.add((i, j, config.RELATION_TYPES['peptide_bond'], dist))
            if i == j: continue
            res_i, res_j = residues[i], residues[j]
            res_i_type, res_j_type = res_i.get_resname(), res_j.get_resname()
            if res_i_type == 'CYS' and res_j_type == 'CYS' and 'SG' in res_i and 'SG' in res_j:
                if np.linalg.norm(res_i['SG'].get_coord() - res_j['SG'].get_coord()) < config.INTERACTION_THRESHOLDS[
                    'disulfide_bond']: edges_with_attrs.add((i, j, config.RELATION_TYPES['disulfide_bond'], dist))
            sc_centroid_i, sc_centroid_j = get_sidechain_centroid(res_i), get_sidechain_centroid(res_j)
            if sc_centroid_i is not None and sc_centroid_j is not None:
                sc_dist = np.linalg.norm(sc_centroid_i - sc_centroid_j)
                if ((res_i_type in POSITIVE_RESIDUES and res_j_type in NEGATIVE_RESIDUES) or (
                        res_i_type in NEGATIVE_RESIDUES and res_j_type in POSITIVE_RESIDUES)):
                    if sc_dist < config.INTERACTION_THRESHOLDS['salt_bridge']: edges_with_attrs.add(
                        (i, j, config.RELATION_TYPES['salt_bridge'], dist))
                if res_i_type in HYDROPHOBIC_RESIDUES and res_j_type in HYDROPHOBIC_RESIDUES:
                    if sc_dist < config.INTERACTION_THRESHOLDS['hydrophobic']: edges_with_attrs.add(
                        (i, j, config.RELATION_TYPES['hydrophobic'], dist))
            if 'N' in res_i and 'O' in res_j and np.linalg.norm(res_i['N'].get_coord() - res_j['O'].get_coord()) < \
                    config.INTERACTION_THRESHOLDS['hydrogen_bond']: edges_with_attrs.add(
                (i, j, config.RELATION_TYPES['hydrogen_bond'], dist))
            if 'O' in res_i and 'N' in res_j and np.linalg.norm(res_i['O'].get_coord() - res_j['N'].get_coord()) < \
                    config.INTERACTION_THRESHOLDS['hydrogen_bond']: edges_with_attrs.add(
                (i, j, config.RELATION_TYPES['hydrogen_bond'], dist))
    if not edges_with_attrs and num_residues > 0: edges_with_attrs.add((0, 0, config.RELATION_TYPES['proximity'], 0.0))
    edge_list = list(edges_with_attrs)
    if not edge_list: return None
    edge_src_temp, edge_dst_temp, edge_type_temp, edge_dist_temp = [e[0] for e in edge_list], [e[1] for e in
                                                                                               edge_list], [e[2] for e
                                                                                                            in
                                                                                                            edge_list], [
        e[3] for e in edge_list]
    edge_src, edge_dst = edge_src_temp + [d for s, d in zip(edge_src_temp, edge_dst_temp) if s != d], edge_dst_temp + [s
                                                                                                                       for
                                                                                                                       s, d
                                                                                                                       in
                                                                                                                       zip(edge_src_temp,
                                                                                                                           edge_dst_temp)
                                                                                                                       if
                                                                                                                       s != d]
    edge_type, edge_dist = edge_type_temp + [t for s, d, t in zip(edge_src_temp, edge_dst_temp, edge_type_temp) if
                                             s != d], edge_dist_temp + [e for s, d, e in
                                                                        zip(edge_src_temp, edge_dst_temp,
                                                                            edge_dist_temp) if s != d]
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type_one_hot = F.one_hot(torch.tensor(edge_type, dtype=torch.long),
                                  num_classes=len(config.RELATION_TYPES)).float()
    expanded_dist_features = gbf(torch.tensor(edge_dist, dtype=torch.float))
    edge_attr = torch.cat([edge_type_one_hot, expanded_dist_features], dim=-1)
    return Data(pos=pos, node_s=node_s, node_v=node_v, edge_index=edge_index, edge_attr=edge_attr)


class UnifiedPeptideDataset(Dataset):
    def __init__(self, dataframe, processed_dir, tokenizer, max_token_len):
        self.df = dataframe
        self.processed_dir = processed_dir
        self.tokenizer = tokenizer
        self.max_len = max_token_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        original_index = row.name  # Pandas索引
        graph_path = os.path.join(self.processed_dir, f"graph_{original_index}.pt")

        if not os.path.exists(graph_path): return None
        try:
            graph_data = torch.load(graph_path)
        except Exception:
            return None

        if not all(hasattr(graph_data, key) for key in
                   ['node_s', 'node_v', 'pos']) or graph_data.num_nodes == 0 or graph_data.num_edges == 0:
            return None

        # ==================== 核心修改：读取 'Sequence' 列 ====================
        sequence = str(row['Sequence'])
        # ====================================================================

        encoding = self.tokenizer.encode_plus(
            sequence, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt',
            return_token_type_ids=False
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'graph_data': graph_data,
            'binary_label': torch.tensor(row['Label'], dtype=torch.long),
            # 使用 config.LABEL_COLUMNS 自动适配 AGgP
            'multilabel_vector': torch.tensor(row[config.LABEL_COLUMNS].values.astype(float), dtype=torch.float),
        }


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'graph_data': Batch.from_data_list([item['graph_data'] for item in batch]),
        'binary_label': torch.stack([item['binary_label'] for item in batch]),
        'multilabel_vector': torch.stack([item['multilabel_vector'] for item in batch]),
    }


def calculate_weights(train_df):
    """
    根据训练集DataFrame计算加权采样器和加权损失的权重。
    """
    print("Calculating weights for sampler and loss...")

    amp_df = train_df[train_df['Label'] == 1]
    pos_counts = amp_df[config.LABEL_COLUMNS].sum().values

    beta = 0.999
    effective_num = 1.0 - np.power(beta, pos_counts)
    weights = (1.0 - beta) / effective_num
    loss_weights = torch.tensor(weights / np.sum(weights) * len(config.LABEL_COLUMNS), dtype=torch.float)

    print("Calculated loss weights for labels:")
    for i, label in enumerate(config.LABEL_COLUMNS):
        print(f"- {label}: {loss_weights[i]:.4f} (based on {int(pos_counts[i])} samples)")

    class_weights = 1.0 / (pos_counts + 1e-8)
    sample_weights = np.zeros(len(train_df))

    for i, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Calculating sample weights"):
        if row['Label'] == 1:
            active_class_indices = np.where(row[config.LABEL_COLUMNS].values == 1)[0]
            if len(active_class_indices) > 0:
                max_weight = class_weights[active_class_indices].max()
                sample_weights[train_df.index.get_loc(i)] = max_weight
            else:
                sample_weights[train_df.index.get_loc(i)] = np.mean(class_weights)
        else:
            sample_weights[train_df.index.get_loc(i)] = np.min(class_weights)

    return torch.DoubleTensor(sample_weights), loss_weights


def create_data_loaders():
    df = pd.read_csv(config.DATA_CSV_PATH)

    train_val_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=42, stratify=df['Label']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=config.VALIDATION_SIZE, random_state=42, stratify=train_val_df['Label']
    )

    print("--- 数据集划分 ---")
    print(f"训练集样本数: {len(train_df)}")
    print(f"验证集样本数: {len(val_df)}")
    print(f"测试集样本数: {len(test_df)}")
    print("--------------------")

    sample_weights, loss_weights = calculate_weights(train_df)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dataset = UnifiedPeptideDataset(train_df, config.PROCESSED_DATA_DIR, config.TOKENIZER, config.MAX_TOKEN_LEN)
    val_dataset = UnifiedPeptideDataset(val_df, config.PROCESSED_DATA_DIR, config.TOKENIZER, config.MAX_TOKEN_LEN)
    test_dataset = UnifiedPeptideDataset(test_df, config.PROCESSED_DATA_DIR, config.TOKENIZER, config.MAX_TOKEN_LEN)

    num_workers = 4
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler,
        num_workers=num_workers, collate_fn=custom_collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=num_workers, collate_fn=custom_collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=num_workers, collate_fn=custom_collate_fn, pin_memory=True
    )

    return train_loader, val_loader, test_loader, loss_weights.to(config.DEVICE)