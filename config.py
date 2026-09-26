# src/config.py
import os

# 项目根目录
#__file__当前目录，os.path.abspath(__file__))转成绝对路径，
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'training')
INPUT_DATA_DIR = os.path.join(DATA_DIR, 'input')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# 模型目录
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
ESM_MODEL_DIR = os.path.join(MODELS_DIR, 'esm')
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, 'trained')

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
GENERATED_PEPTIDES_DIR = os.path.join(OUTPUT_DIR, 'generated_peptides')
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')

# 日志目录
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# ESM模型配置
ESM_MODEL_NAME = "esm2_t6_8M_UR50D"
ESM_MODEL_PATH = os.path.join(ESM_MODEL_DIR, "esm2_t6_8M_UR50D.pt")

# 训练参数
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# 数据集文件命名
DATASET_FILES = {
    # 抗菌肽数据
    'amp_positive': 'general_amp_data.csv',
    'amp_negative': 'negative_data_balanced.csv',
    
    # HLB数据
    'hlb_positive': 'hlb_positive_data.csv',
    'hlb_negative': 'hlb_negative_data.csv',
    
    # 候选肽数据
    'candidate_peptides': 'candidate_peptides.csv'
}

# 模型权重（综合评分）
MODEL_WEIGHTS = {
    'amp_classification': 0.4,  # 抗菌肽分类权重
    'hlb_classification': 0.3,  # HLB分类权重
    'hlb_regression': 0.3       # HLB回归权重
}

# 高活性阈值（前20%分位数）
HIGH_ACTIVITY_PERCENTILE = 0.8

# 创建目录
for dir_path in [DATA_DIR, TRAINING_DATA_DIR, INPUT_DATA_DIR, PROCESSED_DATA_DIR,
                 MODELS_DIR, ESM_MODEL_DIR, TRAINED_MODELS_DIR,
                 OUTPUT_DIR, GENERATED_PEPTIDES_DIR, PREDICTIONS_DIR,
                 LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)
