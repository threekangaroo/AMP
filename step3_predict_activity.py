# src/step3_predict_activity.py
import pandas as pd
import numpy as np
import pickle
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import torch
import esm
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *

class ActivityPredictor:
    def __init__(self):
        self.setup_logging()
        self.models = {}
        self.high_activity_thresholds = {}
        self.esm_model = None
        self.alphabet = None
        self.batch_converter = None
        
    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(LOGS_DIR, f"activity_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_models_and_thresholds(self):
        """加载模型和高活性阈值"""
        print("📥 加载模型和高活性阈值...")
        
        # 加载模型
        model_files = [
            'lgb_amp.pkl', 'rf_amp.pkl', 'xgb_amp.pkl',  # 抗菌肽模型
            'logistic_hlb_class.pkl',  # HLB逻辑回归
            'bayesian_ridge_hlb_reg.pkl'     # HLB回归模型（贝叶斯岭回归）
        ]
        
        for model_file in model_files:
            file_path = os.path.join(TRAINED_MODELS_DIR, model_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        model = pickle.load(f)
                    model_name = model_file.replace('.pkl', '')
                    self.models[model_name] = model
                    print(f"✅ 加载 {model_name}")
                except Exception as e:
                    print(f"❌ 加载 {model_file} 失败: {e}")
        
        # 加载高活性阈值
        threshold_file = os.path.join(PROCESSED_DATA_DIR, 'high_activity_thresholds.pkl')
        if os.path.exists(threshold_file):
            with open(threshold_file, 'rb') as f:
                self.high_activity_thresholds = pickle.load(f)
            print(f"✅ 加载高活性阈值")
        
        print(f"📊 成功加载 {len(self.models)} 个模型")

    def load_candidate_data(self) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        """加载候选肽数据"""
        print("📥 加载候选肽数据...")
        
        candidate_file = os.path.join(PROCESSED_DATA_DIR, 'candidate_peptides_features.npz')
        if os.path.exists(candidate_file):
            data = np.load(candidate_file)
            features = data['features']
            sequences = data['sequences']
            
            # 加载原始数据框
            info_file = os.path.join(PROCESSED_DATA_DIR, 'candidate_peptides_info.csv')
            if os.path.exists(info_file):
                df = pd.read_csv(info_file)
            else:
                df = pd.DataFrame({'sequence': sequences})
            
            print(f"✅ 加载候选肽: {len(sequences)} 条序列")
            return features, sequences, df
        else:
            raise FileNotFoundError(f"候选肽数据不存在: {candidate_file}")

    def predict_with_models(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """使用所有模型进行预测"""
        print("\n🔮 使用所有模型进行预测...")

        predictions = {}

        for model_name, model in self.models.items():
            try:
                if 'bayesian_ridge_hlb_reg' in model_name:
                    # 贝叶斯岭回归：返回预测值和标准差
                    y_pred, y_std = model.predict(features, return_std=True)
                    predictions[model_name] = y_pred
                    predictions[f'{model_name}_std'] = y_std  # 保存不确定性
                    print(f"✅ {model_name} 预测完成 (±{np.mean(y_std):.3f})")
                elif 'reg' in model_name:
                    # 其他回归模型
                    predictions[model_name] = model.predict(features)
                    print(f"✅ {model_name} 预测完成")
                else:
                    # 分类模型 - 获取正类概率
                    pred_proba = model.predict_proba(features)
                    if pred_proba.shape[1] == 2:
                        predictions[model_name] = pred_proba[:, 1]
                    else:
                        predictions[model_name] = pred_proba
                    print(f"✅ {model_name} 预测完成")

            except Exception as e:
                print(f"❌ {model_name} 预测失败: {e}")

        return predictions

    def normalize_scores(self, scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """标准化分数到0-1范围"""
        normalized_scores = {}
        
        for model_name, score in scores.items():
            if 'reg' in model_name:
                # 回归模型：Min-Max标准化
                min_val = np.min(score)
                max_val = np.max(score)
                if max_val > min_val:
                    normalized_scores[model_name] = (score - min_val) / (max_val - min_val)
                else:
                    normalized_scores[model_name] = np.zeros_like(score)
            else:
                # 分类模型：已经是0-1范围
                normalized_scores[model_name] = score
        
        return normalized_scores

    def calculate_consensus_scores(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """计算多模型一致性评分"""
        print("\n🤝 计算多模型一致性评分...")
        
        consensus_scores = []
        
        for i in range(len(next(iter(predictions.values())))):
            high_activity_count = 0
            total_models = 0
            
            for model_name, preds in predictions.items():
                score = preds[i]
                
                # 判断是否为高活性
                if 'reg' in model_name:
                    # 回归模型：与阈值比较
                    threshold = self.high_activity_thresholds.get('hlb_regression', 0)
                    if score >= threshold:
                        high_activity_count += 1
                else:
                    # 分类模型：概率>0.7
                    if score > 0.7:
                        high_activity_count += 1
                
                total_models += 1
            
            if total_models > 0:
                consensus_scores.append(high_activity_count / total_models)
            else:
                consensus_scores.append(0)
        
        return np.array(consensus_scores)

    def calculate_priority_scores(self, normalized_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """计算综合优先级评分"""
        print("\n🎯 计算综合优先级评分...")
        
        priority_scores = []
        
        for i in range(len(next(iter(normalized_scores.values())))):
            score = 0
            
            # 抗菌肽分类模型平均分
            amp_scores = [normalized_scores[name][i] for name in normalized_scores if 'amp' in name]
            if amp_scores:
                score += np.mean(amp_scores) * MODEL_WEIGHTS['amp_classification']
            
            # HLB分类模型分数
            hlb_class_scores = [normalized_scores[name][i] for name in normalized_scores if 'hlb_class' in name]
            if hlb_class_scores:
                score += np.mean(hlb_class_scores) * MODEL_WEIGHTS['hlb_classification']
            
            # HLB回归模型分数
            hlb_reg_scores = [normalized_scores[name][i] for name in normalized_scores if 'hlb_reg' in name]
            if hlb_reg_scores:
                score += np.mean(hlb_reg_scores) * MODEL_WEIGHTS['hlb_regression']
            
            priority_scores.append(score)
        
        return np.array(priority_scores)

    def generate_prediction_results(self, predictions: Dict[str, np.ndarray], sequences: List[str], 
                                  candidate_df: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
        """生成预测结果"""
        print(f"\n📊 生成预测结果 (选择前 {top_k} 个)...")
        
        # 创建结果数据框
        result_df = candidate_df.copy()
        
        # 添加所有模型的原始预测结果
        for model_name, preds in predictions.items():
            result_df[f'{model_name}_score'] = preds
        
        # 标准化分数
        normalized_scores = self.normalize_scores(predictions)
        
        # 计算综合优先级评分
        priority_scores = self.calculate_priority_scores(normalized_scores)
        result_df['priority_score'] = priority_scores
        
        # 计算一致性评分
        consensus_scores = self.calculate_consensus_scores(predictions)
        result_df['consensus_score'] = consensus_scores
        
        # 按优先级排序
        result_df = result_df.sort_values('priority_score', ascending=False)
        
        # 选择前top_k个
        top_peptides = result_df.head(top_k)
        
        print(f"✅ 选择前 {len(top_peptides)} 个高优先级肽段")
        print(f"📊 优先级分数范围: {top_peptides['priority_score'].min():.3f} - {top_peptides['priority_score'].max():.3f}")
        print(f"🤝 一致性评分范围: {top_peptides['consensus_score'].min():.3f} - {top_peptides['consensus_score'].max():.3f}")
        
        return result_df, top_peptides

    def save_predictions(self, all_predictions: pd.DataFrame, top_peptides: pd.DataFrame):
        """保存预测结果"""
        print("\n💾 保存预测结果...")
        
        # 保存所有预测结果
        all_pred_file = os.path.join(PREDICTIONS_DIR, 'all_candidate_predictions.csv')
        all_predictions.to_csv(all_pred_file, index=False)
        print(f"✅ 保存所有预测结果: {all_pred_file}")
        
        # 保存高优先级肽段
        top_pred_file = os.path.join(PREDICTIONS_DIR, 'top_priority_peptides.csv')
        top_peptides.to_csv(top_pred_file, index=False)
        print(f"✅ 保存高优先级肽段: {top_pred_file}")
        
        # 保存简化的候选列表
        simple_file = os.path.join(PREDICTIONS_DIR, 'final_candidate_list.csv')
        simple_df = top_peptides[['sequence', 'priority_score', 'consensus_score']].copy()
        simple_df['rank'] = range(1, len(simple_df) + 1)
        simple_df.to_csv(simple_file, index=False)
        print(f"✅ 保存简化候选列表: {simple_file}")

    def generate_prediction_report(self, all_predictions: pd.DataFrame, top_peptides: pd.DataFrame):
        """生成预测报告"""
        report_file = os.path.join(LOGS_DIR, 'activity_prediction_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("=== 候选肽活性预测报告 ===\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("📊 预测统计:\n")
            f.write(f"   总候选肽数量: {len(all_predictions)}\n")
            f.write(f"   选择高活性肽数量: {len(top_peptides)}\n")
            f.write(f"   选择比例: {len(top_peptides)/len(all_predictions)*100:.1f}%\n\n")
            
            f.write("🎯 使用的模型:\n")
            for model_name in self.models.keys():
                f.write(f"   - {model_name}\n")
            f.write("\n")
            
            f.write("🏆 前10个高优先级肽段:\n")
            for i, (idx, row) in enumerate(top_peptides.head(10).iterrows()):
                f.write(f"   {i+1}. {row['sequence']} (优先级: {row['priority_score']:.3f}, 一致性: {row['consensus_score']:.2f})\n")
            f.write("\n")
            
            f.write("💡 实验建议:\n")
            f.write(f"   1. 优先合成前 {len(top_peptides)} 个肽段进行实验验证\n")
            f.write(f"   2. 关注高一致性评分的肽段（多个模型一致预测）\n")
            f.write(f"   3. 可根据具体需求调整选择标准\n")

    def run_prediction_pipeline(self, top_k: int = 100, custom_file: Optional[str] = None) -> bool:
        """运行预测流程（支持自定义序列）"""
        print("=" * 60)
        print("🔮 步骤3: 候选肽活性预测")
        if custom_file:
            print("   (包含自定义序列预测)")
        print("=" * 60)

        try:
            # 1. 加载模型和阈值
            self.load_models_and_thresholds()

            if not self.models:
                raise ValueError("未找到任何训练好的模型")

            # 2. 加载候选肽
            features, sequences, candidate_df = self.load_candidate_data()

            # 3. 多模型预测候选肽
            predictions = self.predict_with_models(features)

            if not predictions:
                raise ValueError("所有模型预测失败")

            # 4. 生成候选肽预测结果
            all_predictions, top_peptides = self.generate_prediction_results(
                predictions, sequences, candidate_df, top_k
            )

            # 5. 保存候选肽结果
            self.save_predictions(all_predictions, top_peptides)

            # 6. 生成候选肽报告
            self.generate_prediction_report(all_predictions, top_peptides)

            # 7. 如果有自定义文件，预测自定义序列
            if custom_file:
                success = self.predict_custom_sequences(custom_file)
                if not success:
                    print("⚠️  自定义序列预测失败，但候选肽预测已完成")

            print(f"\n✅ 候选肽预测完成!")
            print(f"📊 总候选肽: {len(all_predictions)} 条")
            print(f"🏆 选择高优先级肽段: {len(top_peptides)} 条")

            if custom_file:
                print(f"🔮 自定义序列预测: 已完成")

            return True

        except Exception as e:
            self.logger.error(f"预测流程失败: {e}")
            return False

    def load_esm_model(self):
        """加载ESM模型用于自定义序列特征提取"""
        if self.esm_model is None:
            print("🔧 加载ESM模型用于自定义序列...")
            try:
                self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
                self.batch_converter = self.alphabet.get_batch_converter()
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.esm_model.eval()
                self.esm_model = self.esm_model.to(self.device)
                print(f"✅ ESM模型加载成功: {ESM_MODEL_NAME}")
            except Exception as e:
                print(f"❌ ESM模型加载失败: {e}")
                raise

    def extract_features_for_custom_sequences(self, sequences: List[str]) -> np.ndarray:
        """为自定义序列提取ESM特征"""
        self.load_esm_model()

        print(f"🔬 为 {len(sequences)} 条自定义序列提取特征...")
        features = []

        # 批量处理序列
        batch_size = 32
        sequence_data = [(str(i), seq) for i, seq in enumerate(sequences)]

        for i in range(0, len(sequence_data), batch_size):
            batch_data = sequence_data[i:i + batch_size]

            try:
                batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
                batch_tokens = batch_tokens.to(self.device)

                with torch.no_grad():
                    results = self.esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
                    token_representations = results["representations"][6]

                # 平均池化得到序列级表示
                for j, (_, seq) in enumerate(batch_data):
                    seq_representation = token_representations[j, 1:len(seq) + 1].mean(dim=0)
                    features.append(seq_representation.cpu().numpy())

            except Exception as e:
                print(f"❌ 批次处理失败: {e}")
                # 为失败的批次添加零向量
                for _ in range(len(batch_data)):
                    features.append(np.zeros(320))

        return np.array(features)

    def load_custom_sequences(self, custom_file: str):
        """加载自定义序列"""
        print(f"📥 加载自定义序列文件: {custom_file}")

        if not os.path.exists(custom_file):
            raise FileNotFoundError(f"自定义序列文件不存在: {custom_file}")

        custom_df = pd.read_csv(custom_file)

        if 'sequence' not in custom_df.columns:
            raise ValueError("自定义序列文件必须包含'sequence'列")

        sequences = custom_df['sequence'].tolist()
        print(f"✅ 加载自定义序列: {len(sequences)} 条")

        # 提取特征
        features = self.extract_features_for_custom_sequences(sequences)

        return features, sequences, custom_df

    def predict_custom_sequences(self, custom_file: str, output_dir: str = None):
        """预测自定义序列"""
        if output_dir is None:
            output_dir = PREDICTIONS_DIR

        print("\n" + "=" * 60)
        print("🔮 自定义序列预测")
        print("=" * 60)

        try:
            # 加载自定义序列
            features, sequences, custom_df = self.load_custom_sequences(custom_file)

            # 使用模型进行预测
            predictions = self.predict_with_models(features)

            # 生成预测结果
            all_predictions, top_predictions = self.generate_prediction_results(
                predictions, sequences, custom_df, top_k=len(custom_df)
            )

            # 保存结果
            custom_output_file = os.path.join(output_dir, 'custom_sequences_predictions.csv')
            all_predictions.to_csv(custom_output_file, index=False)

            # 保存简化版本
            simple_file = os.path.join(output_dir, 'custom_sequences_simple.csv')
            simple_df = top_predictions[['sequence', 'priority_score', 'consensus_score']].copy()
            simple_df['rank'] = range(1, len(simple_df) + 1)
            simple_df.to_csv(simple_file, index=False)

            print(f"✅ 自定义序列预测完成!")
            print(f"📊 总自定义序列: {len(all_predictions)} 条")
            print(f"💾 保存详细结果: {custom_output_file}")
            print(f"💾 保存简化结果: {simple_file}")

            # 显示前几个结果
            print(f"\n🏆 自定义序列前5个结果:")
            for i, (idx, row) in enumerate(top_predictions.head(5).iterrows()):
                print(f"   {i + 1}. {row['sequence']}")
                print(f"      优先级分数: {row['priority_score']:.3f}")
                print(f"      一致性评分: {row['consensus_score']:.2f}")

            return True

        except Exception as e:
            print(f"❌ 自定义序列预测失败: {e}")
            return False
def main():
    """主函数"""
    predictor = ActivityPredictor()
    success = predictor.run_prediction_pipeline(top_k=100)
    
    if success:
        print(f"\n🎉 候选肽预测完成!")
        print("💡 下一步: 实验验证高优先级肽段")
        return True
    else:
        print("\n❌ 预测流程失败")
        return False

if __name__ == "__main__":
    main()
