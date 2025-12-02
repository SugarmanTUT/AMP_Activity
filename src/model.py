import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
import config

from gvp import GVP, GVPConvLayer, LayerNorm
from dataset import NUM_AMINO_ACIDS


# ==========================================
# 1. 基础组件 (GVP, CoAttention)
# ==========================================

class GVPEncoder(nn.Module):
    '''
    基于 GVP-GNN 的蛋白质结构编码器
    '''

    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, num_layers=3, drop_rate=0.1):
        super(GVPEncoder, self).__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.W_v = nn.Sequential(GVP(node_in_dim, node_h_dim, activations=(None, None)), LayerNorm(node_h_dim))
        self.W_e = nn.Sequential(nn.Linear(edge_in_dim, edge_h_dim), nn.ReLU(), nn.Linear(edge_h_dim, edge_h_dim),
                                 nn.ReLU())
        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, (edge_h_dim, 0), activations=(F.relu, None), vector_gate=True) for _ in
            range(num_layers))
        self.dropout = nn.Dropout(drop_rate)
        self.norm = nn.ModuleList(LayerNorm(node_h_dim) for _ in range(num_layers))

    def forward(self, node_s, node_v, edge_index, edge_attr, batch):
        h = (node_s, node_v)
        h = self.W_v(h)
        e = self.W_e(edge_attr)
        for i, layer in enumerate(self.layers):
            h_in = h
            v_edge_empty = torch.zeros(e.shape[0], 0, 3, device=e.device)
            h = layer(h, edge_index, (e, v_edge_empty))
            h = self.norm[i](h)
            s, v = h
            s = self.dropout(s)
            h = (s, v)
            h_s = h[0] + h_in[0]
            h_v = h[1] + h_in[1]
            h = (h_s, h_v)
        out = global_mean_pool(h[0], batch)
        return out


class CoAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CoAttention, self).__init__()
        self.feature_dim = feature_dim
        self.W = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, seq_features, struct_features, struct_mask):
        affinity_matrix = torch.bmm(self.W(seq_features), struct_features.transpose(1, 2))
        attn_s2x = F.softmax(affinity_matrix, dim=2)
        attended_seq_features = torch.bmm(attn_s2x, struct_features)
        extended_struct_mask = struct_mask.unsqueeze(1).expand_as(affinity_matrix.transpose(1, 2))
        mask_value = torch.finfo(affinity_matrix.dtype).min
        affinity_matrix_masked = affinity_matrix.transpose(1, 2).masked_fill(extended_struct_mask == 0, mask_value)
        attn_x2s = F.softmax(affinity_matrix_masked, dim=2)
        attended_struct_features = torch.bmm(attn_x2s, seq_features)
        return attended_seq_features, attended_struct_features


# ==========================================
# 2. 特征提取器 (支持对比学习)
# ==========================================

class FeatureExtractor(nn.Module):
    def __init__(self, esm_model_name):
        super(FeatureExtractor, self).__init__()
        self.esm2 = EsmModel.from_pretrained(esm_model_name)

        esm_dim = config.ESM2_EMBEDDING_DIM
        self.fusion_dim = config.FUSION_DIM  # 256

        # [显存优化关键]
        # 在进入 CNN/LSTM 之前，先将 1280 维降到 256 维。
        # 这将大大减少后续层的显存占用。
        self.dimension_reducer = nn.Sequential(
            nn.Linear(esm_dim, self.fusion_dim),
            nn.GELU(),
            nn.LayerNorm(self.fusion_dim)
        )

        # 序列特征细化 (现在输入是 256 维，而不是 1280)
        self.refiner_conv = nn.Conv1d(
            in_channels=self.fusion_dim,
            out_channels=self.fusion_dim,
            kernel_size=5, padding=2, stride=1
        )
        self.refiner_relu = nn.ReLU()
        self.refiner_dropout = nn.Dropout(0.4)

        # LSTM 输入输出现在也是 256
        self.refiner_lstm = nn.LSTM(
            input_size=self.fusion_dim,
            hidden_size=self.fusion_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # 这里的投影层现在只是为了规范化，因为维度已经匹配了
        self.seq_proj = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.struct_proj = nn.Linear(3, self.fusion_dim)

        self.co_attention = CoAttention(self.fusion_dim)
        self.fusion_gate = nn.Linear(self.fusion_dim * 2, self.fusion_dim)

        # GVP 结构编码
        NUM_GAUSSIANS = 50
        edge_attr_dim = len(config.RELATION_TYPES) + NUM_GAUSSIANS
        self.structure_encoder = GVPEncoder(
            node_in_dim=(NUM_AMINO_ACIDS + self.fusion_dim, 2),
            node_h_dim=(128, 16),
            edge_in_dim=edge_attr_dim,
            edge_h_dim=64,
            num_layers=4,
            drop_rate=0.5
        )
        self.gvp_output_proj = nn.Linear(128, 512)

        # 投影头，用于对比学习
        self.cl_seq_proj = nn.Linear(self.fusion_dim, 128)
        self.cl_struct_proj = nn.Linear(128, 128)

    def forward(self, input_ids, attention_mask, graph_data=None):
        # 1. ESM 序列特征 [Batch, Seq_Len, 1280]
        esm_outputs = self.esm2(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = esm_outputs.last_hidden_state

        # [显存优化] 立即降维 [Batch, Seq_Len, 256]
        token_embeddings = self.dimension_reducer(token_embeddings)

        # CNN 处理
        embeddings_for_cnn = token_embeddings.permute(0, 2, 1)  # [B, 256, Seq]
        refined_embeddings = self.refiner_conv(embeddings_for_cnn)
        refined_embeddings = self.refiner_relu(refined_embeddings)
        refined_embeddings = self.refiner_dropout(refined_embeddings)

        # LSTM 处理
        embeddings_for_lstm = refined_embeddings.permute(0, 2, 1)  # [B, Seq, 256]
        # 这里 LSTM 的显存占用会比之前少非常多
        refined_embeddings, _ = self.refiner_lstm(embeddings_for_lstm)

        # 2. 图数据准备
        pos, node_s, node_v = graph_data.pos, graph_data.node_s, graph_data.node_v
        pos_batch, coord_mask = to_dense_batch(pos, graph_data.batch)
        max_nodes = pos_batch.size(1)

        if refined_embeddings.size(1) >= max_nodes + 1:
            seq_embeddings_for_graph = refined_embeddings[:, 1:max_nodes + 1, :]
        else:
            pad_len = max_nodes + 1 - refined_embeddings.size(1)
            padded_refined_embeddings = F.pad(refined_embeddings, (0, 0, 0, pad_len))
            seq_embeddings_for_graph = padded_refined_embeddings[:, 1:max_nodes + 1, :]

        # 3. Co-Attention
        seq_proj = self.seq_proj(seq_embeddings_for_graph)
        struct_proj_for_coattention = self.struct_proj(pos_batch)
        attended_seq, attended_struct = self.co_attention(seq_proj, struct_proj_for_coattention, coord_mask)

        # 4. 提取独立表征用于对比学习
        mask_float = coord_mask.unsqueeze(-1).float()
        seq_rep_raw = (seq_proj * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)
        seq_rep_cl = F.normalize(self.cl_seq_proj(seq_rep_raw), dim=-1)

        # 5. 模态融合
        seq_gate = torch.sigmoid(self.fusion_gate(torch.cat([seq_proj, attended_seq], dim=-1)))
        fused_seq = seq_gate * seq_proj + (1 - seq_gate) * attended_seq
        fused_seq_flat = fused_seq[coord_mask]
        final_node_s = torch.cat([node_s, fused_seq_flat], dim=-1)

        # 6. GVP 编码
        gvp_out = self.structure_encoder(node_s=final_node_s, node_v=node_v, edge_index=graph_data.edge_index,
                                         edge_attr=graph_data.edge_attr, batch=graph_data.batch)
        struct_rep_cl = F.normalize(self.cl_struct_proj(gvp_out), dim=-1)

        # 7. 最终投影
        fused_features = self.gvp_output_proj(gvp_out)

        return fused_features, seq_rep_cl, struct_rep_cl


# ==========================================
# 3. 标签交互头 (Label Interaction Head)
# ==========================================

class LabelInteractionHead(nn.Module):
    def __init__(self, feature_dim, num_labels):
        super(LabelInteractionHead, self).__init__()
        self.num_labels = num_labels
        self.feature_dim = feature_dim

        # 1. 标签嵌入
        self.label_embeddings = nn.Parameter(torch.Tensor(num_labels, feature_dim))
        nn.init.xavier_uniform_(self.label_embeddings)

        # 2. 投影层
        self.proj_feat = nn.Linear(feature_dim, feature_dim)
        self.proj_label = nn.Linear(feature_dim, feature_dim)

        # 3. 标签偏置 (Bias)
        self.label_bias = nn.Parameter(torch.zeros(num_labels))

        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, fused_features):
        # 归一化与正则化
        features = self.layer_norm(fused_features)
        features = self.dropout(features)

        # 1. 投影
        feat_proj = self.proj_feat(features)  # [Batch, Dim]
        label_proj = self.proj_label(self.label_embeddings)  # [Num_Labels, Dim]

        # 2. 计算点积相似度 (Scaled Dot-Product)
        scale_factor = self.feature_dim ** -0.5
        logits = torch.matmul(feat_proj, label_proj.T) * scale_factor

        # 3. 加上偏置
        logits = logits + self.label_bias

        return logits


# ==========================================
# 4. 模型主类
# ==========================================

class SimplifiedMultimodalClassifier(nn.Module):
    def __init__(self, esm_model_name, num_labels):
        super(SimplifiedMultimodalClassifier, self).__init__()
        self.feature_extractor = FeatureExtractor(esm_model_name)
        fused_dim = 512

        self.multilabel_classifier_head = LabelInteractionHead(feature_dim=fused_dim, num_labels=num_labels)
        self.binary_classifier_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim // 4), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(fused_dim // 4, 2)
        )

    def unfreeze_feature_extractor(self):
        for param in self.parameters():
            param.requires_grad = True

    # 专门提取特征的方法，方便在 Train 中做 Mixup
    def extract_features(self, input_ids, attention_mask, graph_data):
        return self.feature_extractor(input_ids, attention_mask, graph_data)

    # 独立的分类方法
    def classify(self, fused_features):
        multilabel_output = self.multilabel_classifier_head(fused_features)
        binary_output = self.binary_classifier_head(fused_features)
        return {"multilabel": multilabel_output, "binary": binary_output}

    # 保持标准的 forward 接口用于评估
    def forward(self, input_ids, attention_mask, graph_data):
        fused_features, _, _ = self.extract_features(input_ids, attention_mask, graph_data)
        if fused_features is None: return None
        return self.classify(fused_features)


def build_model():
    model = SimplifiedMultimodalClassifier(esm_model_name=config.MODEL_NAME, num_labels=config.NUM_LABELS)
    return model.to(config.DEVICE)