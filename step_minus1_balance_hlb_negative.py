# src/step_minus1_balance_hlb_negative.py

import os
import pandas as pd
import logging
from datetime import datetime

# ✅ 使用 config 中已定义的路径
from config import TRAINING_DATA_DIR, DATASET_FILES

# 构建文件路径
HLB_POSITIVE = os.path.join(TRAINING_DATA_DIR, DATASET_FILES['hlb_positive'])
HLB_NEGATIVE = os.path.join(TRAINING_DATA_DIR, DATASET_FILES['hlb_negative'])
GENERAL_AMP = os.path.join(TRAINING_DATA_DIR, DATASET_FILES['amp_positive'])


def setup_logging():
    from config import LOGS_DIR
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR,
                            f"step_minus1_balance_hlb_negative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("StepMinus1_BalanceHLBNegative")


logger = setup_logging()


def balance_hlb_negative():
    logger.info("🔍 开始步骤 -1: 平衡 HLB 负样本数量")
    logger.info(f"正样本文件: {HLB_POSITIVE}")
    logger.info(f"负样本文件: {HLB_NEGATIVE}")
    logger.info(f"候选池文件: {GENERAL_AMP}")

    # 检查文件是否存在
    for path in [HLB_POSITIVE, HLB_NEGATIVE, GENERAL_AMP]:
        if not os.path.exists(path):
            logger.error(f"❌ 文件不存在: {path}")
            return False

    try:
        df_pos = pd.read_csv(HLB_POSITIVE)
        df_neg = pd.read_csv(HLB_NEGATIVE)
        df_gen = pd.read_csv(GENERAL_AMP)
    except Exception as e:
        logger.error(f"读取CSV失败: {e}")
        return False

    # 简单清洗
    for df in [df_pos, df_neg, df_gen]:
        if 'sequence' not in df.columns:
            logger.error("缺少 'sequence' 列")
            return False
        df['sequence'] = df['sequence'].astype(str).str.strip().str.upper()

    pos_set = set(df_pos['sequence'])
    neg_set = set(df_neg['sequence'])

    # 候选池：是 AMP，但不是 HLB 正样本，也不是已有负样本
    candidate_pool = df_gen[
        (~df_gen['sequence'].isin(pos_set)) &
        (~df_gen['sequence'].isin(neg_set))
        ]

    target_neg_count = len(df_pos)
    current_neg_count = len(df_neg)
    need_add = max(0, target_neg_count - current_neg_count)

    logger.info(f"✅ HLB 正样本数量: {target_neg_count}")
    logger.info(f"✅ 当前负样本数量: {current_neg_count}")
    logger.info(f"📌 需补充负样本: {need_add} 条")

    if need_add == 0:
        logger.info("✅ 负样本数量已平衡，无需补充")
        return True

    if len(candidate_pool) < need_add:
        logger.warning(f"⚠️ 候选池不足！需要 {need_add}，但只有 {len(candidate_pool)} 可用")
        selected = candidate_pool
    else:
        selected = candidate_pool.sample(n=need_add, random_state=42, replace=False)

    # 合并并保存新负样本
    new_neg_df = pd.concat([df_neg, selected], ignore_index=True)
    try:
        new_neg_df.to_csv(HLB_NEGATIVE, index=False)
        logger.info(f"✅ 已更新负样本文件，新总数: {len(new_neg_df)}")
    except Exception as e:
        logger.error(f"保存文件失败: {e}")
        return False

    logger.info("🎉 步骤 -1 成功完成：HLB 负样本已平衡")
    return True


def main():
    logger.info("=" * 60)
    logger.info("⚖️  步骤 -1: 平衡HLB负样本数量")
    logger.info("   功能: 从 general_amp 扩充 hlb_negative 数量")
    logger.info("=" * 60)

    success = balance_hlb_negative()
    if success:
        logger.info("✅ 步骤 -1 成功完成")
    else:
        logger.error("❌ 步骤 -1 执行失败")
    return success


if __name__ == "__main__":
    main()