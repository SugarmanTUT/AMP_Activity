import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import mutual_info_score

import config  # 用你的 config 里的 LABEL_COLUMNS, DATA_CSV_PATH


def compute_label_relationships(csv_path=None, label_columns=None, min_support=0):
    """
    计算多标签之间的共现频率、皮尔逊相关系数和互信息。

    参数:
        csv_path: 数据集 CSV 路径，默认用 config.DATA_CSV_PATH
        label_columns: 多标签列名列表，默认用 config.LABEL_COLUMNS
        min_support: 最小共现次数阈值（小于该次数的关系可认为不可靠）

    返回:
        cooccurrence_df: 共现计数矩阵 (DataFrame)
        correlation_df: 皮尔逊相关矩阵 (DataFrame)
        mi_df: 互信息矩阵 (DataFrame)
        pair_list: 每对标签的详细关系列表（便于后续构图）
    """
    if csv_path is None:
        csv_path = config.DATA_CSV_PATH
    if label_columns is None:
        label_columns = config.LABEL_COLUMNS

    df = pd.read_csv(csv_path)

    # 只保留 AMP 样本（Label == 1），因为功能标签只在 AMP 上有意义
    if 'Label' in df.columns:
        df_amp = df[df['Label'] == 1].copy()
    else:
        df_amp = df.copy()

    # 取多标签矩阵，确保是 0/1
    Y = df_amp[label_columns].astype(int)

    # 1) 共现计数矩阵
    cooccurrence = pd.DataFrame(
        np.zeros((len(label_columns), len(label_columns)), dtype=int),
        index=label_columns,
        columns=label_columns
    )

    for i, lab_i in enumerate(label_columns):
        for j, lab_j in enumerate(label_columns):
            if i <= j:
                # 两者都为 1 的样本数
                c = int(((Y[lab_i] == 1) & (Y[lab_j] == 1)).sum())
                cooccurrence.loc[lab_i, lab_j] = c
                cooccurrence.loc[lab_j, lab_i] = c

    # 2) 皮尔逊相关系数矩阵（基于 0/1 标签）
    correlation_df = Y.corr(method="pearson")

    # 3) 互信息矩阵
    mi_mat = np.zeros_like(cooccurrence, dtype=float)

    for i, lab_i in enumerate(label_columns):
        for j, lab_j in enumerate(label_columns):
            if i <= j:
                mi = mutual_info_score(Y[lab_i], Y[lab_j])
                mi_mat[i, j] = mi
                mi_mat[j, i] = mi

    mi_df = pd.DataFrame(mi_mat, index=label_columns, columns=label_columns)

    # 4) 打平成 pair 列表，方便后续画图或构建标签图
    pair_list = []
    for lab_i, lab_j in combinations(label_columns, 2):
        co = cooccurrence.loc[lab_i, lab_j]
        if co < min_support:
            continue
        corr = correlation_df.loc[lab_i, lab_j]
        mi = mi_df.loc[lab_i, lab_j]
        pair_list.append({
            "label_i": lab_i,
            "label_j": lab_j,
            "cooccurrence": int(co),
            "pearson_corr": float(corr),
            "mutual_info": float(mi),
        })

    return cooccurrence, correlation_df, mi_df, pair_list


if __name__ == "__main__":
    co_df, corr_df, mi_df, pairs = compute_label_relationships(min_support=10)

    print("\n=== 共现计数矩阵 (Co-occurrence) ===")
    print(co_df)

    print("\n=== 皮尔逊相关矩阵 (Pearson Correlation) ===")
    print(corr_df.round(3))

    print("\n=== 互信息矩阵 (Mutual Information) ===")
    print(mi_df.round(3))

    print("\n=== 标签对关系 (共现 >= 10) ===")
    for p in pairs:
        print(
            f"{p['label_i']} - {p['label_j']}: "
            f"co={p['cooccurrence']}, "
            f"corr={p['pearson_corr']:.3f}, "
            f"MI={p['mutual_info']:.3f}"
        )