import os
import pandas as pd
import torch
from tqdm import tqdm
import config
from dataset import parse_pdb_to_chemical_graph
import multiprocessing


def process_single_item(args):
    """
    处理单个数据样本
    """
    index, row = args

    # [匹配逻辑]
    # 假设 PDB 文件名对应 CSV 的行号。
    # 第 0 行 -> sequence_1.pdb
    pdb_filename = f"sequence_{index + 1}.pdb"

    pdb_path = os.path.join(config.PDB_FILES_DIR, pdb_filename)
    output_path = os.path.join(config.PROCESSED_DATA_DIR, f"graph_{index}.pt")

    if os.path.exists(output_path):
        return True

    if not os.path.exists(pdb_path):
        # 调试用：如果找不到，返回错误信息
        return f"MISSING:{pdb_path}"

    graph_data = parse_pdb_to_chemical_graph(
        pdb_path,
        max_nodes=config.MAX_TOKEN_LEN - 2,
        is_train=False
    )

    if graph_data is None or graph_data.num_nodes == 0:
        return f"PARSE_ERROR:{pdb_path}"

    try:
        torch.save(graph_data, output_path)
        return True
    except Exception as e:
        return f"SAVE_ERROR:{e}"


def preprocess_and_save_graphs_mp():
    print("\n--- 开始数据预处理 ---")

    if not os.path.exists(config.DATA_CSV_PATH):
        print(f"❌ 错误: 找不到数据集文件 {config.DATA_CSV_PATH}")
        return

    df = pd.read_csv(config.DATA_CSV_PATH)

    # 强制重置 processed 目录
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    if not os.path.exists(config.PDB_FILES_DIR):
        print(f"❌ 严重错误: PDB 文件夹不存在！路径: {config.PDB_FILES_DIR}")
        return

    print(f"数据集样本数: {len(df)}")
    print(f"标签列: {config.LABEL_COLUMNS}")

    tasks = list(df.iterrows())

    # ==================== 修改部分 ====================
    # 固定使用 4 个进程
    num_processes = 4
    print(f"将使用 {num_processes} 个 CPU 核心进行并行处理...")
    # =================================================

    success_count = 0
    missing_count = 0
    debug_prints = 0

    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap(process_single_item, tasks), total=len(tasks), desc="生成图数据"):
            if result is True:
                success_count += 1
            else:
                if isinstance(result, str) and result.startswith("MISSING"):
                    missing_count += 1
                    if debug_prints < 3:
                        print(f"\n[调试] 找不到: {result.split(':', 1)[1]}")
                        debug_prints += 1

    print("\n--- 处理结果统计 ---")
    print(f"成功: {success_count}")
    print(f"缺失 PDB: {missing_count}")

    if success_count == 0:
        print("⚠️ 警告: 没有生成任何图文件，请检查 PDB 文件名是否真的是 sequence_1.pdb 这种格式，以及是否与 CSV 行号对应。")


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    preprocess_and_save_graphs_mp()