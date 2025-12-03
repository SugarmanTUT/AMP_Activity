import torch
from transformers import AutoTokenizer
import os

# --- 项目路径配置 ---
DATA_DIR = "../data"
# [修改] 现在的核心数据就是 dataset.csv，不需要合并了
DATA_CSV_PATH = f"{DATA_DIR}/dataset.csv"
PDB_FILES_DIR = f"{DATA_DIR}/pdb/"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed/"
VISUALIZATION_DIR = "../visualizations/"

MODEL_TYPE = "multimodal"
BEST_MODEL_SAVE_PATH = "../models/best_model.bin"

# --- 模型和分词器配置 ---
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- 数据集划分配置 ---
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# --- 训练配置 ---
DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"
MAX_TOKEN_LEN = 128
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 10

# --- 损失与权重配置 ---
TASK_LOSS_WEIGHT = 0.1
CONTRASTIVE_LOSS_WEIGHT = 0.1
CONTRASTIVE_TEMP = 0.07

# --- Mixup 配置 ---
USE_MIXUP = True
MIXUP_ALPHA = 0.4

# --- ASL 损失函数配置 ---
ASL_GAMMA_NEG = 4
ASL_GAMMA_POS = 0
ASL_CLIP = 0.05

# --- 标签配置 ---
# [核心修改] 根据你的新数据集更新列名
# 现在的列是: Sequence, Label, ABP, AFP, AGnP, AGpP, AVP, pLDDT
# 我们只取多标签的列
LABEL_COLUMNS = ["ABP", "AFP", "AGnP", "AGpP", "AVP"]
NUM_LABELS = len(LABEL_COLUMNS)

# --- 模型维度配置 ---
ESM2_EMBEDDING_DIM = 1280
FUSION_DIM = 256

# --- 多关系化学图配置 ---
RELATION_TYPES = {
    'peptide_bond': 0, 'hydrogen_bond': 1, 'hydrophobic': 2,
    'salt_bridge': 3, 'disulfide_bond': 4, 'proximity': 5
}
NUM_RELATIONS = len(RELATION_TYPES)
INTERACTION_THRESHOLDS = {
    'hydrogen_bond': 3.5, 'hydrophobic': 5.0, 'salt_bridge': 4.0,
    'disulfide_bond': 2.2, 'proximity': 12.0
}