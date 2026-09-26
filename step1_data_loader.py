# src/step1_data_loader.py
import pandas as pd
import numpy as np
import torch
import esm
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *

class DataLoader:
    def __init__(self):
        self.setup_logging()
        self.model, self.alphabet = self.load_esm_model()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(LOGS_DIR, f"data_loading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_esm_model(self):
        """加载ESM模型"""
        print("🔧 加载ESM-2模型...")
        try:
            model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
            print(f"✅ ESM模型加载成功: {ESM_MODEL_NAME}")
            return model, alphabet
        except Exception as e:
            print(f"❌ ESM模型加载失败: {e}")
            raise

    def load_dataset(self, filename: str) -> pd.DataFrame:
        """加载单个数据集"""
        file_path = os.path.join(TRAINING_DATA_DIR, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"✅ 加载 {filename}: {len(df)} 条序列")
                return df
            except Exception as e:
                print(f"❌ 加载 {filename} 失败: {e}")
                return pd.DataFrame()
        else:
            print(f"⚠️  文件不存在: {file_path}")
            return pd.DataFrame()

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """加载所有数据集"""
        print("📥 加载所有数据集...")
        
        datasets = {}
        for dataset_name, filename in DATASET_FILES.items():
            df = self.load_dataset(filename)
            if not df.empty:
                datasets[dataset_name] = df
        
        return datasets

    def extract_features(self, sequences: List[str], batch_size: int = 32) -> np.ndarray:
        """提取ESM特征"""
        self.logger.info(f"开始提取 {len(sequences)} 条序列的特征...")
        
        all_features = []
        sequence_data = [(str(i), seq) for i, seq in enumerate(sequences)]
        
        for i in range(0, len(sequence_data), batch_size):
            batch_data = sequence_data[i:i + batch_size]
            
            try:
                batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
                batch_tokens = batch_tokens.to(self.device)
                
                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[6], return_contacts=False)
                    token_representations = results["representations"][6]
                
                sequence_representations = []
                for j, (_, seq) in enumerate(batch_data):
                    seq_representation = token_representations[j, 1:len(seq)+1].mean(dim=0)
                    sequence_representations.append(seq_representation.cpu().numpy())
                
                all_features.extend(sequence_representations)
                self.logger.info(f"处理批次 {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}")
                
            except Exception as e:
                self.logger.error(f"批次 {i//batch_size + 1} 处理失败: {e}")
                for _ in range(len(batch_data)):
                    all_features.append(np.zeros(320))
        
        return np.array(all_features)

    def prepare_amp_classification_data(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """准备抗菌肽分类数据"""
        print("\n🦠 准备抗菌肽分类数据...")
        
        if 'amp_positive' not in datasets or 'amp_negative' not in datasets:
            print("❌ 缺少抗菌肽正负样本数据")
            return {}
        
        pos_df = datasets['amp_positive']
        neg_df = datasets['amp_negative']
        
        # 合并正负样本
        pos_df['label'] = 1
        neg_df['label'] = 0
        
        combined_data = pd.concat([pos_df, neg_df], ignore_index=True)
        combined_data = combined_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        
        sequences = combined_data['sequence'].tolist()
        features = self.extract_features(sequences)
        labels = combined_data['label'].values
        
        print(f"✅ 抗菌肽分类数据: {len(combined_data)} 条样本")
        print(f"   正样本: {len(pos_df)} 条")
        print(f"   负样本: {len(neg_df)} 条")
        
        return {
            'features': features,
            'labels': labels,
            'sequences': sequences,
            'dataframe': combined_data
        }

    def prepare_hlb_classification_data(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """准备HLB分类数据（混合负样本）"""
        print("\n🍊 准备HLB分类数据（混合负样本）...")

        if 'hlb_positive' not in datasets or 'hlb_negative' not in datasets:
            print("❌ 缺少HLB正负样本数据")
            return {}

        pos_df = datasets['hlb_positive']
        neg_df = datasets['hlb_negative']

        # 从AMP正样本中抽取21条作为额外负样本
        if 'amp_positive' in datasets:
            amp_pos_df = datasets['amp_positive']
            # 确保不重复抽取已经在HLB正样本中的序列
            hlb_pos_sequences = set(pos_df['sequence'].tolist())
            amp_candidates = amp_pos_df[~amp_pos_df['sequence'].isin(hlb_pos_sequences)]

            if len(amp_candidates) >= 21:
                extra_neg_df = amp_candidates.sample(n=21, random_state=RANDOM_STATE)
                print(f"✅ 从AMP正样本中抽取21条作为额外负样本")
            else:
                extra_neg_df = amp_candidates
                print(f"⚠️  AMP正样本中可用序列不足，只抽取{len(extra_neg_df)}条")

            # 合并负样本
            neg_df = pd.concat([neg_df, extra_neg_df], ignore_index=True)
        else:
            print("⚠️  未找到AMP正样本数据，使用原有负样本")

        # 合并正负样本
        pos_df['label'] = 1
        neg_df['label'] = 0

        combined_data = pd.concat([pos_df, neg_df], ignore_index=True)
        combined_data = combined_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        sequences = combined_data['sequence'].tolist()
        features = self.extract_features(sequences)
        labels = combined_data['label'].values

        print(f"✅ HLB分类数据: {len(combined_data)} 条样本")
        print(f"   正样本: {len(pos_df)} 条")
        print(f"   负样本: {len(neg_df)} 条")
        print(
            f"   📊 负样本组成: 原有{len(datasets['hlb_negative'])}条 + AMP抽取{len(neg_df) - len(datasets['hlb_negative'])}条")

        return {
            'features': features,
            'labels': labels,
            'sequences': sequences,
            'dataframe': combined_data
        }

    def prepare_hlb_regression_data(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """准备HLB回归数据（扩展负样本）"""
        print("\n📈 准备HLB回归数据（扩展负样本）...")

        if 'hlb_positive' not in datasets or 'hlb_negative' not in datasets:
            print("❌ 缺少HLB数据")
            return {}

        pos_df = datasets['hlb_positive']
        neg_df = datasets['hlb_negative']

        # 从AMP正样本中抽取21条作为额外样本（赋予低efficiency值）
        if 'amp_positive' in datasets:
            amp_pos_df = datasets['amp_positive']
            hlb_pos_sequences = set(pos_df['sequence'].tolist())
            amp_candidates = amp_pos_df[~amp_pos_df['sequence'].isin(hlb_pos_sequences)]

            if len(amp_candidates) >= 21:
                extra_df = amp_candidates.sample(n=21, random_state=RANDOM_STATE)
                # 为这些样本赋予低efficiency值（0.1-0.3范围）
                extra_df = extra_df.copy()
                # 找到efficiency列
                efficiency_col = None
                for col in pos_df.columns:
                    if 'efficiency' in col.lower():
                        efficiency_col = col
                        break

                if efficiency_col:
                    # 赋予随机低效率值
                    np.random.seed(RANDOM_STATE)
                    extra_df[efficiency_col] = np.random.uniform(0.1, 0.3, len(extra_df))
                    print(f"✅ 从AMP正样本中抽取21条，赋予低efficiency值")
                else:
                    print("⚠️  未找到efficiency列，无法扩展回归数据")
                    extra_df = pd.DataFrame()
            else:
                extra_df = amp_candidates
                print(f"⚠️  AMP正样本中可用序列不足，只抽取{len(extra_df)}条")

            # 合并数据
            combined_data = pd.concat([pos_df, neg_df, extra_df], ignore_index=True)
        else:
            combined_data = pd.concat([pos_df, neg_df], ignore_index=True)
            print("⚠️  未找到AMP正样本数据，使用原有数据")

        # 检查efficiency列
        efficiency_col = None
        for col in combined_data.columns:
            if 'efficiency' in col.lower() or 'efficency' in col.lower():
                efficiency_col = col
                break

        if efficiency_col is None:
            print("❌ 未找到efficiency列")
            return {}

        # 移除efficiency为空的行
        combined_data = combined_data.dropna(subset=[efficiency_col])
        combined_data = combined_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        sequences = combined_data['sequence'].tolist()
        features = self.extract_features(sequences)
        targets = combined_data[efficiency_col].values

        print(f"✅ HLB回归数据: {len(combined_data)} 条样本")
        print(f"   efficiency范围: {targets.min():.3f} - {targets.max():.3f}")
        print(f"   efficiency均值: {targets.mean():.3f} ± {targets.std():.3f}")

        return {
            'features': features,
            'targets': targets,
            'sequences': sequences,
            'dataframe': combined_data,
            'efficiency_column': efficiency_col
        }

    def prepare_candidate_data(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """准备候选肽数据"""
        print("\n🎯 准备候选肽数据...")
        
        if 'candidate_peptides' not in datasets:
            print("❌ 缺少候选肽数据")
            return {}
        
        candidate_df = datasets['candidate_peptides']
        sequences = candidate_df['sequence'].tolist()
        features = self.extract_features(sequences)
        
        print(f"✅ 候选肽数据: {len(candidate_df)} 条序列")
        
        return {
            'features': features,
            'sequences': sequences,
            'dataframe': candidate_df
        }

    def save_training_data(self, training_data: Dict[str, Any]):
        """保存训练数据"""
        print("\n💾 保存训练数据...")
        
        for data_name, data_dict in training_data.items():
            output_file = os.path.join(PROCESSED_DATA_DIR, f'{data_name}_features.npz')
            
            if 'labels' in data_dict:
                np.savez_compressed(
                    output_file,
                    features=data_dict['features'],
                    sequences=data_dict['sequences'],
                    labels=data_dict['labels']
                )
            elif 'targets' in data_dict:
                np.savez_compressed(
                    output_file,
                    features=data_dict['features'],
                    sequences=data_dict['sequences'],
                    targets=data_dict['targets']
                )
            else:
                np.savez_compressed(
                    output_file,
                    features=data_dict['features'],
                    sequences=data_dict['sequences']
                )
            
            # 保存数据信息
            info_file = os.path.join(PROCESSED_DATA_DIR, f'{data_name}_info.csv')
            data_dict['dataframe'].to_csv(info_file, index=False)
            
            print(f"✅ 保存 {data_name}: {output_file}")

    def run_data_loading(self) -> Dict[str, Any]:
        """运行数据加载流程"""
        print("=" * 60)
        print("📥 步骤1: 数据加载和特征提取")
        print("=" * 60)
        
        try:
            # 1. 加载所有数据集
            datasets = self.load_all_datasets()
            
            if not datasets:
                raise ValueError("未成功加载任何数据集")
            
            # 2. 准备训练数据
            training_data = {}
            
            # 抗菌肽分类数据
            amp_data = self.prepare_amp_classification_data(datasets)
            if amp_data:
                training_data['amp_classification'] = amp_data
            
            # HLB分类数据
            hlb_class_data = self.prepare_hlb_classification_data(datasets)
            if hlb_class_data:
                training_data['hlb_classification'] = hlb_class_data
            
            # HLB回归数据
            hlb_reg_data = self.prepare_hlb_regression_data(datasets)
            if hlb_reg_data:
                training_data['hlb_regression'] = hlb_reg_data
            
            # 候选肽数据
            candidate_data = self.prepare_candidate_data(datasets)
            if candidate_data:
                training_data['candidate_peptides'] = candidate_data
            
            if not training_data:
                raise ValueError("未成功准备任何训练数据")
            
            # 3. 保存训练数据
            self.save_training_data(training_data)
            
            print(f"\n✅ 数据加载完成!")
            print(f"📊 加载的数据集: {list(training_data.keys())}")
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            return {}

def main():
    """主函数"""
    loader = DataLoader()
    training_data = loader.run_data_loading()
    
    if training_data:
        print(f"\n🎉 数据准备完成!")
        print("💡 下一步: 开始模型训练")
        return training_data
    else:
        print("\n❌ 数据准备失败")
        return {}

if __name__ == "__main__":
    main()
