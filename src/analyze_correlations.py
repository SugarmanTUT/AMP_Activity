import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import config


def analyze_label_correlations():
    # 1. 加载数据
    if not os.path.exists(config.DATA_CSV_PATH):
        print(f"错误: 找不到数据文件 {config.DATA_CSV_PATH}")
        return

    df = pd.read_csv(config.DATA_CSV_PATH)

    # 2. 关键步骤：只筛选 AMP 阳性样本进行分析
    # 原因：如果我们包含大量的 Non-AMP (全0标签)，它们会导致所有标签看起来都有极高的正相关性（因为都是0），
    # 这会掩盖 AMP 功能内部的真实关系。
    amp_df = df[df['Label'] == 1][config.LABEL_COLUMNS]

    print(f"正在分析 {len(amp_df)} 个 AMP 阳性样本的标签相关性...")

    # 创建可视化目录
    os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)

    # --- 分析维度 1: Pearson 相关系数矩阵 ---
    # 范围 [-1, 1]，1表示完全正相关，0表示无关
    corr_matrix = amp_df.corr(method='pearson')

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Label Correlation Matrix (Pearson)\n(Calculated on AMP samples only)')
    plt.tight_layout()
    plt.savefig(f"{config.VISUALIZATION_DIR}/label_correlation_pearson.png")
    print(f"1. 相关系数矩阵已保存至 {config.VISUALIZATION_DIR}/label_correlation_pearson.png")

    # --- 分析维度 2: 共现矩阵 (Co-occurrence Matrix) ---
    # 对角线是该标签的总数，非对角线是两个标签同时为1的次数
    cooccurrence_matrix = np.dot(amp_df.T, amp_df)
    cooccurrence_df = pd.DataFrame(cooccurrence_matrix, index=config.LABEL_COLUMNS, columns=config.LABEL_COLUMNS)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cooccurrence_df, annot=True, fmt="d", cmap='YlGnBu')
    plt.title('Label Co-occurrence Matrix (Raw Counts)')
    plt.tight_layout()
    plt.savefig(f"{config.VISUALIZATION_DIR}/label_cooccurrence.png")
    print(f"2. 共现矩阵已保存至 {config.VISUALIZATION_DIR}/label_cooccurrence.png")

    # --- 分析维度 3: 条件概率矩阵 P(Row|Col) ---
    # 含义：已知具备 Column 功能，具备 Row 功能的概率是多少？
    # 例如：P(AGnP | ABP) -> 如果它是抗菌肽(ABP)，它是抗革兰氏阴性菌(AGnP)的概率？
    # 这对于理解层级关系非常重要。

    cond_prob_matrix = np.zeros((len(config.LABEL_COLUMNS), len(config.LABEL_COLUMNS)))

    for i, label_row in enumerate(config.LABEL_COLUMNS):
        for j, label_col in enumerate(config.LABEL_COLUMNS):
            # 分母：具备 Col 功能的样本数
            col_count = amp_df[label_col].sum()
            # 分子：同时具备 Row 和 Col 功能的样本数
            intersect_count = len(amp_df[(amp_df[label_row] == 1) & (amp_df[label_col] == 1)])

            if col_count > 0:
                cond_prob_matrix[i, j] = intersect_count / col_count
            else:
                cond_prob_matrix[i, j] = 0.0

    cond_prob_df = pd.DataFrame(cond_prob_matrix, index=config.LABEL_COLUMNS, columns=config.LABEL_COLUMNS)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cond_prob_df, annot=True, fmt=".2f", cmap='Blues', vmin=0, vmax=1)
    plt.title('Conditional Probability P(Row | Column)\n"Given Column=1, probability that Row=1"')
    plt.xlabel("Given Condition (Column)")
    plt.ylabel("Target Probability (Row)")
    plt.tight_layout()
    plt.savefig(f"{config.VISUALIZATION_DIR}/label_conditional_prob.png")
    print(f"3. 条件概率矩阵已保存至 {config.VISUALIZATION_DIR}/label_conditional_prob.png")

    # --- 简单文本报告 ---
    print("\n=== 简要分析报告 ===")
    print("观察生成的 conditional_prob 图：")
    print("1. ABP (抗菌) 与 AGnP/AGpP 通常有极强的双向或单向依赖。")
    print("2. 关注 APP (抗寄生虫/穿孔) 列：查看其他标签对 APP 的概率。")
    print("   如果 P(APP | AVP) 较高，说明抗病毒肽通常也具有穿孔活性。")
    print("3. 如果矩阵大部分区域颜色很深(>0.5)，说明相关性很强 -> 必须引入 Label Attention。")
    print("4. 如果矩阵大部分区域接近0 (除了对角线)，说明标签非常独立 -> 不需要 Label Attention。")


if __name__ == "__main__":
    analyze_label_correlations()