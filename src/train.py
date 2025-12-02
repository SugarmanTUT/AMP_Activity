import torch
import torch.nn as nn
import numpy as np
import os
import sys

# [核心修复] 设置显存分配策略，减少碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import config
from model import build_model
from dataset import create_data_loaders
from losses import AsymmetricLossOptimized


# --- 辅助函数：对比学习损失 (NT-Xent) ---
def contrastive_loss_fn(seq_rep, struct_rep, temperature=0.07):
    # 如果权重是0，直接返回0，避免计算
    if config.CONTRASTIVE_LOSS_WEIGHT == 0:
        return torch.tensor(0.0, device=seq_rep.device)

    logits = torch.matmul(seq_rep, struct_rep.T) / temperature
    batch_size = seq_rep.size(0)
    labels = torch.arange(batch_size).to(seq_rep.device)
    loss_seq = nn.CrossEntropyLoss()(logits, labels)
    loss_struct = nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_seq + loss_struct) / 2


# --- 辅助函数：Mixup ---
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, data_loader, optimizer, device, scaler, scheduler, loss_weights):
    model.train()
    total_loss, processed_batches = 0, 0
    binary_loss_fn = nn.CrossEntropyLoss()

    multilabel_loss_fn = AsymmetricLossOptimized(
        gamma_neg=config.ASL_GAMMA_NEG,
        gamma_pos=config.ASL_GAMMA_POS,
        clip=config.ASL_CLIP,
        disable_torch_grad_focal_loss=True
    )

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(data_loader, desc="Training (Mixup)")):
        if batch is None: continue

        input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        graph_data, binary_labels = batch["graph_data"].to(device), batch["binary_label"].to(device)
        multilabel_vectors = batch["multilabel_vector"].to(device)

        with autocast():
            fused_features, seq_rep, struct_rep = model.extract_features(input_ids, attention_mask, graph_data)
            if fused_features is None: continue

            # Loss 1: Contrastive (如果在config里设为0，这里几乎不消耗时间)
            loss_cl = contrastive_loss_fn(seq_rep, struct_rep, temperature=config.CONTRASTIVE_TEMP)

            # Loss 2: Binary
            outputs_orig = model.classify(fused_features)
            loss_b = binary_loss_fn(outputs_orig["binary"], binary_labels)

            # Loss 3: Multilabel (with Mixup)
            amp_mask = (binary_labels == 1)
            # 只有当Batch里至少有两个AMP样本时才做Mixup，否则无法置换
            if amp_mask.sum() > 1 and config.USE_MIXUP:
                features_amp = fused_features[amp_mask]
                targets_amp = multilabel_vectors[amp_mask]

                mixed_features, targets_a, targets_b, lam = mixup_data(features_amp, targets_amp,
                                                                       config.MIXUP_ALPHA)
                mixed_logits = model.multilabel_classifier_head(mixed_features)
                loss_ml = mixup_criterion(multilabel_loss_fn, mixed_logits, targets_a, targets_b, lam)

            elif amp_mask.sum() >= 1:
                # 只有1个AMP，或者不使用Mixup
                features_amp = fused_features[amp_mask]
                targets_amp = multilabel_vectors[amp_mask]
                logits = model.multilabel_classifier_head(features_amp)
                loss_ml = multilabel_loss_fn(logits, targets_amp)
            else:
                # 没有AMP样本
                loss_ml = torch.tensor(0.0, device=device)

            combined_loss = (config.TASK_LOSS_WEIGHT * loss_b) + \
                            ((1 - config.TASK_LOSS_WEIGHT) * loss_ml) + \
                            (config.CONTRASTIVE_LOSS_WEIGHT * loss_cl)

            # 梯度累积标准化
            combined_loss = combined_loss / config.GRADIENT_ACCUMULATION_STEPS

        if torch.isnan(combined_loss) or torch.isinf(combined_loss):
            continue

        scaler.scale(combined_loss).backward()

        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += combined_loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        processed_batches += 1

    return total_loss / processed_batches if processed_batches > 0 else 0


def find_best_thresholds(model, data_loader, device):
    model.eval()
    all_scores, all_targets = [], []
    print("\nFinding best thresholds...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Threshold Tuning"):
            if batch is None: continue
            amp_mask = (batch["binary_label"] == 1)
            if amp_mask.sum() == 0: continue

            input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            graph_data = batch["graph_data"].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask, graph_data)

            scores = torch.sigmoid(outputs["multilabel"][amp_mask.to(device)]).cpu()
            targets = batch["multilabel_vector"][amp_mask].cpu()
            all_scores.append(scores)
            all_targets.append(targets)

    if not all_scores: return [0.5] * len(config.LABEL_COLUMNS)

    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    best_thresholds = []
    search_range = np.arange(0.05, 0.96, 0.01)

    for i in range(len(config.LABEL_COLUMNS)):
        best_f1 = -1
        best_thresh = 0.5
        for thresh in search_range:
            preds = (all_scores[:, i] > thresh).astype(int)
            f1 = f1_score(all_targets[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        best_thresholds.append(best_thresh)

    print("Best Thresholds:", [f"{t:.2f}" for t in best_thresholds])
    return best_thresholds


def evaluate(model, data_loader, device, thresholds=None, is_test=False):
    model.eval()
    b_targets, b_preds = [], []
    ml_targets, ml_outputs = [], []

    if thresholds is None: thresholds = [0.5] * config.NUM_LABELS

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if batch is None: continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            graph_data = batch["graph_data"].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask, graph_data)
            if outputs is None: continue

            # Binary
            binary_labels = batch["binary_label"].cpu().numpy()
            b_targets.extend(binary_labels)
            b_preds.extend(torch.argmax(outputs["binary"], dim=1).cpu().numpy())

            # Multilabel
            amp_mask = (batch["binary_label"] == 1)
            if amp_mask.sum() > 0:
                ml_targets.extend(batch["multilabel_vector"][amp_mask].cpu().numpy())
                scores = torch.sigmoid(outputs["multilabel"][amp_mask]).cpu().numpy()
                preds = (scores > np.array(thresholds)).astype(int)
                ml_outputs.extend(preds)

    res = {}
    if b_targets:
        res['binary'] = {'balanced_accuracy': balanced_accuracy_score(b_targets, b_preds)}
        if is_test:
            print(classification_report(b_targets, b_preds, target_names=['Non-AMP', 'AMP']))

    if ml_targets:
        t_np, o_np = np.array(ml_targets), np.array(ml_outputs)
        label_accs = [balanced_accuracy_score(t_np[:, i], o_np[:, i]) for i in range(t_np.shape[1])]
        res['multilabel'] = {
            'avg_balanced_accuracy': np.mean(label_accs) * 100,
            'macro_f1': f1_score(t_np, o_np, average='macro', zero_division=0)
        }
        if is_test:
            print(classification_report(t_np, o_np, target_names=config.LABEL_COLUMNS, zero_division=0))

    return res


def run_training_pipeline():
    print("--- Data Loading ---")
    train_loader, val_loader, test_loader, _ = create_data_loaders()

    print("--- Model Init ---")
    model = build_model()

    # ==================================================================================
    # [优化] 冻结策略：冻结 ESM-2 的大部分层，只训练最后几层
    # ==================================================================================
    FREEZE_LAYERS = 28
    print(f"--- Optimizing: Freezing the first {FREEZE_LAYERS} layers of ESM-2 ---")

    frozen_params = 0
    total_params = 0

    for name, param in model.feature_extractor.esm2.named_parameters():
        total_params += param.numel()
        if "encoder.layer." in name:
            layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
            if layer_num < FREEZE_LAYERS:
                param.requires_grad = False
                frozen_params += param.numel()
        elif "embeddings" in name:  # 同时也冻结底层的 Embedding
            param.requires_grad = False
            frozen_params += param.numel()

    print(f"Total Params: {total_params / 1e6:.2f}M")
    print(f"Frozen Params: {frozen_params / 1e6:.2f}M")
    print(f"Trainable Params: {(total_params - frozen_params) / 1e6:.2f}M")

    # 确保没有冻结后续的融合层和分类头
    model.unfreeze_feature_extractor()

    # ==================================================================================

    # 优化器配置
    optimizer = torch.optim.AdamW([
        # ESM-2 剩余的微调层
        {"params": filter(lambda p: p.requires_grad, model.feature_extractor.esm2.parameters()), "lr": 1e-5},
        # 其他部分
        {"params": [p for n, p in model.named_parameters() if "esm2" not in n], "lr": 2e-4}
    ], weight_decay=1e-3)

    steps = len(train_loader) * config.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader) * 1,
                                                num_training_steps=steps)
    scaler = GradScaler()

    best_metric = 0
    patience_counter = 0
    history = []

    print(f"--- Starting Training (Batch={config.BATCH_SIZE}, Accum={config.GRADIENT_ACCUMULATION_STEPS}) ---")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        loss = train_one_epoch(model, train_loader, optimizer, config.DEVICE, scaler, scheduler, None)
        print(f"Train Loss: {loss:.4f}")

        # 强制清理显存
        torch.cuda.empty_cache()

        metrics = evaluate(model, val_loader, config.DEVICE)
        ml_metrics = metrics.get("multilabel", {})

        curr_ba = ml_metrics.get("avg_balanced_accuracy", 0)
        curr_f1 = ml_metrics.get('macro_f1', 0)
        print(f"Val Multi-label BA: {curr_ba:.2f}% | Macro F1: {curr_f1:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': loss,
            'val_ba': curr_ba,
            'val_f1': curr_f1
        })

        if curr_ba > best_metric:
            best_metric = curr_ba
            patience_counter = 0
            torch.save(model.state_dict(), config.BEST_MODEL_SAVE_PATH)
            print(">>> New Best Model Saved!")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping.")
                break

    pd.DataFrame(history).to_csv("../models/training_history.csv", index=False)

    print("\n--- Final Evaluation ---")
    if os.path.exists(config.BEST_MODEL_SAVE_PATH):
        print("Loading best model for testing...")
        model.load_state_dict(torch.load(config.BEST_MODEL_SAVE_PATH))
        best_thresh = find_best_thresholds(model, val_loader, config.DEVICE)
        evaluate(model, test_loader, config.DEVICE, thresholds=best_thresh, is_test=True)


if __name__ == '__main__':
    os.makedirs(os.path.dirname(config.BEST_MODEL_SAVE_PATH), exist_ok=True)
    run_training_pipeline()