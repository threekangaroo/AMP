"""
步骤0：候选肽段生成
功能：从候选蛋白序列（CSV格式）中通过滑动窗口切片 + 单点突变生成候选肽库
输入：data/input/candidate_proteins.csv（必须包含 'sequence' 列）
输出：output/generated_peptides/candidate_peptides.csv
"""

import os
import pandas as pd
import logging
from datetime import datetime

# 导入配置
from config import INPUT_DATA_DIR, GENERATED_PEPTIDES_DIR, DATASET_FILES

# 配置路径
INPUT_CANDIDATE_PROTEINS = os.path.join(INPUT_DATA_DIR, "candidate_proteins.csv")
OUTPUT_PEPTIDES = os.path.join(GENERATED_PEPTIDES_DIR, DATASET_FILES['candidate_peptides'])

# 参数配置
MIN_PEPTIDE_LENGTH = 2  # 2 ≤ m < n
MAX_PEPTIDE_LENGTH = None  # 自动根据序列长度确定最大长度（n-1）


def setup_logging():
    """设置日志"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"step0_generate_peptides_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("Step0_GeneratePeptides")


logger = setup_logging()


def peptide_slicing(sequence):
    """滑动窗口切片生成肽段"""
    n = len(sequence)
    min_length = MIN_PEPTIDE_LENGTH
    max_length = n - 1 if MAX_PEPTIDE_LENGTH is None else min(MAX_PEPTIDE_LENGTH, n - 1)

    if max_length < min_length:
        logger.warning(f"序列过短（长度{n}），无法生成长度≥{min_length}的肽段")
        return []

    logger.debug(f"对长度 {n} 的序列进行切片，肽段长度范围: {min_length}-{max_length}")

    peptides = []
    for length in range(min_length, max_length + 1):
        for start in range(n - length + 1):
            peptide = sequence[start:start + length]
            peptides.append(peptide)

    unique_peptides = list(set(peptides))
    logger.debug(f"从该序列生成 {len(unique_peptides)} 个唯一肽段")

    return unique_peptides


def in_silico_mutation(peptides):
    """对肽段进行单点突变"""
    logger.info(f"开始对 {len(peptides)} 个肽段进行单点突变")
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20种标准氨基酸
    mutated_peptides = []

    for i, peptide in enumerate(peptides):
        for pos in range(len(peptide)):
            for aa in amino_acids:
                if aa != peptide[pos]:
                    mutated = peptide[:pos] + aa + peptide[pos + 1:]
                    mutated_peptides.append(mutated)

        if (i + 1) % 100 == 0:
            logger.info(f"已处理 {i + 1}/{len(peptides)} 个肽段的突变")

    unique_mutated = list(set(mutated_peptides))
    logger.info(f"突变完成，生成 {len(unique_mutated)} 个唯一突变肽段")

    return unique_mutated


def generate_candidate_peptides():
    """生成候选肽库：切片 + 单点突变"""
    logger.info("开始步骤0: 候选肽段生成")
    logger.info(f"输入文件: {INPUT_CANDIDATE_PROTEINS}")
    logger.info(f"输出文件: {OUTPUT_PEPTIDES}")

    # 检查输入文件
    if not os.path.exists(INPUT_CANDIDATE_PROTEINS):
        logger.error(f"输入文件不存在: {INPUT_CANDIDATE_PROTEINS}")
        logger.info(f"请将候选蛋白CSV文件放在: {INPUT_CANDIDATE_PROTEINS}")
        logger.info("文件格式要求：必须包含 'sequence' 列，每行一个蛋白序列")
        return False

    sequences = []
    try:
        df = pd.read_csv(INPUT_CANDIDATE_PROTEINS)
        logger.info(f"成功读取CSV文件，共 {len(df)} 行")

        if 'sequence' not in df.columns:
            logger.error(f"CSV文件缺少 'sequence' 列，当前列名为: {list(df.columns)}")
            return False

        # 提取并清洗序列
        raw_sequences = df['sequence'].astype(str).tolist()
        for seq in raw_sequences:
            seq = seq.strip()
            if not seq:
                continue
            if not all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in seq.upper()):
                logger.warning(f"跳过无效序列（含非标准氨基酸）: {seq[:30]}...")
                continue
            sequences.append(seq.upper())

        logger.info(f"共加载 {len(sequences)} 个有效候选蛋白序列")

    except Exception as e:
        logger.error(f"读取或解析CSV文件失败: {e}")
        return False

    if len(sequences) == 0:
        logger.error("未加载到任何有效蛋白序列，请检查输入文件内容")
        return False

    # 1. 滑动窗口切片
    logger.info("=== 执行滑动窗口切片 ===")
    all_sliced_peptides = []
    for i, seq in enumerate(sequences):
        logger.info(f"处理第 {i + 1}/{len(sequences)} 个蛋白 (长度: {len(seq)})")
        sliced_peptides = peptide_slicing(seq)
        all_sliced_peptides.extend(sliced_peptides)

    all_sliced_peptides = list(set(all_sliced_peptides))
    logger.info(f"切片完成，共生成 {len(all_sliced_peptides)} 个唯一肽段")

    # 2. 单点突变
    logger.info("=== 执行单点突变 ===")
    if len(all_sliced_peptides) == 0:
        logger.warning("无肽段可用于突变，跳过突变步骤")
        mutated_peptides = []
    else:
        mutated_peptides = in_silico_mutation(all_sliced_peptides)

    # 3. 合并结果
    logger.info("=== 组合切片与突变结果 ===")
    all_candidates = list(set(all_sliced_peptides + mutated_peptides))
    logger.info(f"最终候选肽库大小: {len(all_candidates)} 个肽段")

    # 保存结果
    try:
        os.makedirs(os.path.dirname(OUTPUT_PEPTIDES), exist_ok=True)
        result_df = pd.DataFrame(all_candidates, columns=['sequence'])
        result_df.to_csv(OUTPUT_PEPTIDES, index=False)
        logger.info(f"✅ 候选肽库已成功保存至: {OUTPUT_PEPTIDES}")
    except Exception as e:
        logger.error(f"保存输出文件失败: {e}")
        return False

    return True


def main():
    """供 main.py 调用的入口函数"""
    logger.info("=" * 60)
    logger.info("AMP发现管道 - 步骤0: 候选肽段生成")
    logger.info("功能: 从候选蛋白CSV生成所有子肽 + 单点突变体")
    logger.info("=" * 60)

    success = generate_candidate_peptides()
    if success:
        logger.info("✅ 步骤0完成!")
    else:
        logger.error("❌ 步骤0失败!")
    return success


if __name__ == "__main__":
    main()