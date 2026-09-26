# src/step2_train_models.py
import pandas as pd
import numpy as np
import pickle
import os
import sys
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report, 
                           confusion_matrix, mean_squared_error, r2_score, 
                           mean_absolute_error)
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, Tuple, Any, List
# 在已有的import后面添加这两个
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, average_precision_score
from sklearn.metrics import explained_variance_score, max_error
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *

class ModelTrainer:
    def __init__(self):
        self.setup_logging()
        self.models = {}
        self.results = {}
        self.high_activity_thresholds = {}
        
    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(LOGS_DIR, f"model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_training_data(self) -> Dict[str, Any]:
        """加载训练数据"""
        print("📥 加载训练数据...")

        training_data = {}
        data_files = {
            'amp_classification': 'amp_classification_features.npz',
            'hlb_classification': 'hlb_classification_features.npz',
            'hlb_regression': 'hlb_regression_features.npz'
        }

        for data_name, filename in data_files.items():
            file_path = os.path.join(PROCESSED_DATA_DIR, filename)
            if os.path.exists(file_path):
                data = np.load(file_path)
                training_data[data_name] = data

                # 详细数据诊断
                n_samples = len(data['features'])
                if 'labels' in data:
                    labels = data['labels']
                    unique, counts = np.unique(labels, return_counts=True)
                    print(f"✅ 加载 {data_name}: {n_samples} 条样本")
                    print(f"   类别分布: {dict(zip(unique, counts))}")

                    # 检查特征相似度
                    if len(unique) == 2:  # 二分类问题
                        features_0 = data['features'][labels == 0]
                        features_1 = data['features'][labels == 1]
                        if len(features_0) > 0 and len(features_1) > 0:
                            from sklearn.metrics.pairwise import cosine_similarity
                            # 计算类间平均相似度
                            similarity = cosine_similarity(features_0.mean(axis=0).reshape(1, -1),
                                                           features_1.mean(axis=0).reshape(1, -1))[0][0]
                            print(f"   类间特征相似度: {similarity:.4f}")
                elif 'targets' in data:
                    targets = data['targets']
                    print(f"✅ 加载 {data_name}: {n_samples} 条样本")
                    print(f"   目标值范围: {targets.min():.3f} - {targets.max():.3f}")
                    print(f"   目标值均值: {targets.mean():.3f} ± {targets.std():.3f}")
            else:
                print(f"⚠️  数据文件不存在: {file_path}")

        return training_data

    def calculate_high_activity_threshold(self, targets: np.ndarray) -> float:
        """计算高活性阈值（前20%分位数）"""
        threshold = np.percentile(targets, HIGH_ACTIVITY_PERCENTILE * 100)
        print(f"📊 高活性阈值: {threshold:.3f} (前{ HIGH_ACTIVITY_PERCENTILE * 100}%分位数)")
        return threshold

    def train_amp_classification_models(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """训练抗菌肽分类模型"""
        print("\n🦠 训练抗菌肽分类模型...")
        
        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
        )
        
        models = {}
        results = {}
        
        # 1. LightGBM
        print("🌲 训练LightGBM...")
        lgb_model = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_train, y_train)
        models['lgb_amp'] = lgb_model
        results['lgb_amp'] = self.extended_classification_metrics(lgb_model, 'LightGBM-AMP', X_test, y_test)
        
        # 2. Random Forest
        print("🌳 训练Random Forest...")
        rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        models['rf_amp'] = rf_model
        results['rf_amp'] = self.extended_classification_metrics(rf_model, 'RF-AMP', X_test, y_test)
        
        # 3. XGBoost
        print("🚀 训练XGBoost...")
        xgb_model = xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss', verbosity=0)
        xgb_model.fit(X_train, y_train)
        models['xgb_amp'] = xgb_model
        results['xgb_amp'] = self.extended_classification_metrics(xgb_model, 'XGBoost-AMP', X_test, y_test)
        
        return {'models': models, 'results': results, 'test_data': (X_test, y_test)}

    def train_hlb_classification_model(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """训练HLB分类模型（逻辑回归 + 分层K折交叉验证）"""
        print("\n🍊 训练HLB分类模型（逻辑回归）...")
        print(f"⚠️  注意: 仅使用 {len(labels)} 条样本，模型预测仅供参考")

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

        # 检查类别分布
        unique, counts = np.unique(labels, return_counts=True)
        print(f"📊 类别分布: {dict(zip(unique, counts))}")

        # 根据样本量选择合适的交叉验证策略
        if len(labels) < 10 or min(counts) < 3:
            # 样本太少，使用简单划分
            print("🔍 样本量过少，使用简单训练测试划分...")
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.3, random_state=RANDOM_STATE, stratify=labels
            )

            model = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000)
            model.fit(X_train, y_train)

            # 在测试集上评估
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            evaluation_method = 'Train-Test-Split'

        else:
            # 使用分层K折交叉验证（确保每个类别在每个折中都有代表）
            n_splits = min(5, min(counts))  # 确保每个类别在每个折中至少有1个样本
            print(f"🔍 使用分层 {n_splits} 折交叉验证评估...")

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            model = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000)

            y_pred_cv = cross_val_predict(model, features, labels, cv=cv)
            y_pred_proba_cv = cross_val_predict(model, features, labels, cv=cv, method='predict_proba')[:, 1]

            # 计算CV指标
            accuracy = accuracy_score(labels, y_pred_cv)
            auc_score = roc_auc_score(labels, y_pred_proba_cv)
            f1 = f1_score(labels, y_pred_cv, zero_division=0)

            evaluation_method = f'Stratified-{n_splits}-Fold-CV'

        # 训练最终模型（使用全部数据）
        model.fit(features, labels)

        models = {'logistic_hlb_class': model}
        results = {
            'logistic_hlb_class': {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'f1_score': f1,
                'evaluation_method': evaluation_method,
                'sample_size': len(labels),
                'class_distribution': dict(zip(unique, counts))
            }
        }

        print(f"✅ 逻辑回归 - 准确率: {accuracy:.4f}, AUC: {auc_score:.4f}, F1: {f1:.4f}")
        print(f"   📊 基于 {len(labels)} 条样本的{evaluation_method}")
        print(f"   ⚠️  样本量较少，预测不确定性较大")

        return {'models': models, 'results': results, 'test_data': (features, labels)}

    def train_hlb_regression_model(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """训练HLB回归模型（贝叶斯岭回归 + K折交叉验证）"""
        print("\n📈 训练HLB回归模型（贝叶斯岭回归）...")
        print(f"⚠️  注意: 仅使用 {len(targets)} 条样本，模型预测仅供参考")

        from sklearn.linear_model import BayesianRidge
        from sklearn.model_selection import cross_val_predict, KFold
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        # 根据样本量选择合适的交叉验证策略
        if len(targets) < 10:
            # 样本太少，使用简单划分
            print("🔍 样本量过少，使用简单训练测试划分...")
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.3, random_state=RANDOM_STATE
            )

            model = BayesianRidge()
            model.fit(X_train, y_train)

            # 在测试集上评估
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            evaluation_method = 'Train-Test-Split'

        else:
            # 使用K折交叉验证
            n_splits = min(5, len(targets))
            print(f"🔍 使用 {n_splits} 折交叉验证评估...")

            cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            model = BayesianRidge()

            y_pred_cv = cross_val_predict(model, features, targets, cv=cv)

            # 计算CV指标
            r2 = r2_score(targets, y_pred_cv)
            rmse = np.sqrt(mean_squared_error(targets, y_pred_cv))
            mae = mean_absolute_error(targets, y_pred_cv)

            evaluation_method = f'{n_splits}-Fold-CV'

        # 训练最终模型（使用全部数据）
        model.fit(features, targets)

        # 计算高活性阈值
        self.high_activity_thresholds['hlb_regression'] = self.calculate_high_activity_threshold(targets)

        models = {'bayesian_ridge_hlb_reg': model}
        results = {
            'bayesian_ridge_hlb_reg': {
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae,
                'evaluation_method': evaluation_method,
                'sample_size': len(targets)
            }
        }

        print(f"✅ 贝叶斯岭回归 - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        print(f"   📊 基于 {len(targets)} 条样本的{evaluation_method}")
        print(f"   ⚠️  样本量较少，预测不确定性较大")

        return {'models': models, 'results': results, 'test_data': (features, targets)}

    def evaluate_classification_model(self, model: Any, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估分类模型"""
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # 计算其他指标
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            results = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1_score': report['1']['f1-score'],
                'confusion_matrix': cm.tolist(),
                'classification_report': report
            }
            
            print(f"✅ {model_name} - 准确率: {accuracy:.4f}, AUC: {auc_score:.4f}, F1: {results['f1_score']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ {model_name} 评估失败: {e}")
            return {'error': str(e)}

    def extended_classification_metrics(self, model: Any, model_name: str, X_test: np.ndarray,
                                        y_test: np.ndarray) -> Dict:
        """扩展分类模型评估指标"""
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # 基础指标
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # 扩展指标
            avg_precision = average_precision_score(y_test, y_pred_proba)
            log_loss_val = log_loss(y_test, y_pred_proba)

            # 分类报告和混淆矩阵
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            results = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'average_precision': avg_precision,
                'log_loss': log_loss_val,
                'confusion_matrix': cm.tolist(),
                'classification_report': report
            }

            print(f"✅ {model_name} - 准确率: {accuracy:.4f}, AUC: {auc_score:.4f}, F1: {f1:.4f}")
            print(f"   精确率: {precision:.4f}, 召回率: {recall:.4f}, AP: {avg_precision:.4f}")

            return results

        except Exception as e:
            print(f"❌ {model_name} 评估失败: {e}")
            return {'error': str(e)}

    def extended_regression_metrics(self, model: Any, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """扩展回归模型评估指标"""
        try:
            y_pred = model.predict(X_test)

            # 基础回归指标
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # 扩展指标
            mape = np.mean(np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1e-8, None))) * 100
            explained_variance = explained_variance_score(y_test, y_pred)
            max_err = max_error(y_test, y_pred)

            results = {
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'mae': mae,
                'mape': mape,
                'explained_variance': explained_variance,
                'max_error': max_err,
                'predictions': y_pred.tolist(),
                'actual': y_test.tolist()
            }

            print(f"✅ {model_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            print(f"   MAPE: {mape:.2f}%, 解释方差: {explained_variance:.4f}")

            return results

        except Exception as e:
            print(f"❌ {model_name} 评估失败: {e}")
            return {'error': str(e)}


    def evaluate_regression_model(self, model: Any, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估回归模型"""
        try:
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results = {
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'mae': mae,
                'predictions': y_pred.tolist(),
                'actual': y_test.tolist()
            }
            
            print(f"✅ {model_name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ {model_name} 评估失败: {e}")
            return {'error': str(e)}

    def save_models(self):
        """保存所有模型"""
        print("\n💾 保存所有模型...")
        
        for model_name, model in self.models.items():
            model_file = os.path.join(TRAINED_MODELS_DIR, f'{model_name}.pkl')
            try:
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                print(f"✅ 保存 {model_name}: {model_file}")
            except Exception as e:
                print(f"❌ 保存 {model_name} 失败: {e}")

    def save_high_activity_thresholds(self):
        """保存高活性阈值"""
        threshold_file = os.path.join(PROCESSED_DATA_DIR, 'high_activity_thresholds.pkl')
        with open(threshold_file, 'wb') as f:
            pickle.dump(self.high_activity_thresholds, f)
        print(f"✅ 保存高活性阈值: {threshold_file}")

    def generate_training_report(self):
        """生成训练报告"""
        report_file = os.path.join(LOGS_DIR, 'model_training_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("=== 模型训练和评估报告 ===\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("🎯 训练的模型:\n")
            for model_name in self.models.keys():
                f.write(f"   - {model_name}\n")
            f.write("\n")
            
            f.write("📊 模型性能汇总:\n")
            for task_name, task_results in self.results.items():
                f.write(f"   {task_name}:\n")
                for model_name, results in task_results.items():
                    if 'error' in results:
                        f.write(f"     {model_name}: 错误 - {results['error']}\n")
                    else:
                        if 'auc_score' in results:
                            # 分类模型
                            f.write(f"     {model_name}: 准确率={results['accuracy']:.4f}, AUC={results['auc_score']:.4f}, F1={results['f1_score']:.4f}\n")
                        else:
                            # 回归模型
                            f.write(f"     {model_name}: R²={results['r2_score']:.4f}, RMSE={results['rmse']:.4f}, MAE={results['mae']:.4f}\n")
            
            f.write(f"\n🎯 高活性阈值:\n")
            for model_type, threshold in self.high_activity_thresholds.items():
                f.write(f"   {model_type}: {threshold:.3f}\n")
            
            f.write(f"\n💡 使用建议:\n")
            f.write(f"   1. 所有模型保存在: {TRAINED_MODELS_DIR}\n")
            f.write(f"   2. 下一步运行候选肽预测\n")
            f.write(f"   3. 高活性阈值用于筛选多模型一致预测的肽段\n")

    def run_training_pipeline(self) -> bool:
        """运行训练流程"""
        print("=" * 60)
        print("🎯 步骤2: 模型训练和评估")
        print("=" * 60)
        
        try:
            # 1. 加载数据
            training_data = self.load_training_data()
            
            if not training_data:
                raise ValueError("未找到训练数据")
            
            # 2. 训练各个任务的模型
            self.results = {}
            
            # 抗菌肽分类模型
            if 'amp_classification' in training_data:
                data = training_data['amp_classification']
                amp_results = self.train_amp_classification_models(data['features'], data['labels'])
                self.models.update(amp_results['models'])
                self.results['amp_classification'] = amp_results['results']
            
            # HLB分类模型
            if 'hlb_classification' in training_data:
                data = training_data['hlb_classification']
                hlb_class_results = self.train_hlb_classification_model(data['features'], data['labels'])
                self.models.update(hlb_class_results['models'])
                self.results['hlb_classification'] = hlb_class_results['results']
            
            # HLB回归模型
            if 'hlb_regression' in training_data:
                data = training_data['hlb_regression']
                hlb_reg_results = self.train_hlb_regression_model(data['features'], data['targets'])
                self.models.update(hlb_reg_results['models'])
                self.results['hlb_regression'] = hlb_reg_results['results']
            
            if not self.models:
                raise ValueError("没有成功训练任何模型")
            
            # 3. 保存模型和阈值
            self.save_models()
            self.save_high_activity_thresholds()
            
            # 4. 生成报告
            self.generate_training_report()
            
            print(f"\n✅ 模型训练完成!")
            print(f"📊 成功训练模型: {len(self.models)} 个")
            print(f"🎯 高活性阈值: {self.high_activity_thresholds}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            return False

    # 在step2中添加这个诊断函数
    def diagnose_hlb_data_issues(self):
        """诊断HLB数据问题"""
        print("\n🔍 诊断HLB数据问题...")

        # 从训练数据中获取序列
        hlb_data = np.load(os.path.join(PROCESSED_DATA_DIR, 'hlb_classification_features.npz'))
        sequences = hlb_data['sequences']
        labels = hlb_data['labels']

        positive_seqs = [sequences[i] for i in range(len(sequences)) if labels[i] == 1]
        negative_seqs = [sequences[i] for i in range(len(sequences)) if labels[i] == 0]

        print(f"正样本数量: {len(positive_seqs)}")
        print(f"负样本数量: {len(negative_seqs)}")

        # 分析序列特征
        self.analyze_current_data_issues(positive_seqs, negative_seqs)

def main():
    """主函数"""
    trainer = ModelTrainer()
    success = trainer.run_training_pipeline()
    
    if success:
        print(f"\n🎉 模型训练完成!")
        print("💡 下一步: 运行候选肽预测")
        return True
    else:
        print("\n❌ 模型训练失败")
        return False

if __name__ == "__main__":
    main()
