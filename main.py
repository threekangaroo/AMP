# src/main.py
import os
import sys
import argparse
from datetime import datetime


def run_balance_hlb_negative():
    """运行步骤 -1：平衡 HLB 负样本数量"""
    from step_minus1_balance_hlb_negative import main as step_minus1_main
    print("\n" + "="*60)
    print("⚖️  步骤-1: 平衡HLB负样本数量")
    print("   功能: 从 general_amp 中补充 hlb_negative 数据")
    print("="*60)
    return step_minus1_main()


def run_peptide_generation():
    """运行步骤0：候选肽段生成"""
    from step0_generate_peptides import main as step0_main
    print("\n" + "="*60)
    print("🧩 步骤0: 候选肽段生成")
    print("   功能: 从候选蛋白CSV生成所有子肽 + 单点突变体")
    print("="*60)
    return step0_main()


def run_data_loading():
    """运行步骤1：数据加载和特征提取"""
    from step1_data_loader import main as data_main
    print("\n" + "="*60)
    print("📥 步骤1: 数据加载和特征提取")
    print("   功能: 加载抗菌肽数据、HLB数据，提取ESM特征")
    print("="*60)
    return data_main()


def run_model_training():
    """运行步骤2：模型训练和评估"""
    from step2_train_models import main as training_main
    print("\n" + "="*60)
    print("🎯 步骤2: 模型训练和评估")
    print("   功能: 训练AMP分类、HLB分类与回归模型")
    print("="*60)
    return training_main()


def run_activity_prediction(custom_file: str = None, top_k: int = 100):
    """运行步骤3：候选肽活性预测"""
    from step3_predict_activity import ActivityPredictor
    print("\n" + "="*60)
    print("🔮 步骤3: 候选肽活性预测")
    if custom_file:
        print(f"   (使用自定义序列文件: {custom_file})")
    print("   功能: 对候选肽进行综合评分与排序")
    print("="*60)

    predictor = ActivityPredictor()
    success = predictor.run_prediction_pipeline(top_k=top_k, custom_file=custom_file)
    if success:
        print("✅ 候选肽预测完成")
    else:
        print("❌ 候选肽预测失败")
    return success


def main():
    """主函数：AMP发现管道入口"""
    parser = argparse.ArgumentParser(description='🧬 AMP发现管道')
    parser.add_argument('--step', type=int, choices=[-1, 0, 1, 2, 3],
                        help='运行特定步骤: -1=平衡HLB负样本, 0=生成肽库, 1=加载, 2=训练, 3=预测')
    parser.add_argument('--all', action='store_true', help='运行全部流程（从 step0 开始）')
    parser.add_argument('--top_k', type=int, default=100, help='预测时返回前K个高优先级肽段（默认: 100）')
    parser.add_argument('--custom', type=str, help='指定自定义候选肽文件路径（CSV格式，含sequence列）')

    args = parser.parse_args()

    print("="*60)
    print("🧬 AMP发现管道启动")
    print("="*60)

    start_time = datetime.now()
    success = True
    steps_ran = False

    try:
        # 注意：--all 不包含 step -1，因为它是预处理
        if args.step == -1:
            steps_ran = True
            if not run_balance_hlb_negative():
                print("❌ 步骤 -1 失败")
                success = False

        elif args.step == 0 or args.all:
            steps_ran = True
            if not run_peptide_generation():
                print("❌ 步骤0失败：候选肽生成中断")
                success = False

        elif args.step == 1 or args.all:
            steps_ran = True
            if not run_data_loading():
                print("❌ 步骤1失败：数据加载中断")
                success = False

        elif args.step == 2 or args.all:
            steps_ran = True
            if not run_model_training():
                print("❌ 步骤2失败：模型训练中断")
                success = False

        elif args.step == 3 or args.all:
            steps_ran = True
            if not run_activity_prediction(custom_file=args.custom, top_k=args.top_k):
                print("❌ 步骤3失败：活性预测中断")
                success = False

        if not steps_ran:
            parser.print_help()
            print("\n可用命令示例:")
            print("  python main.py --step -1                    # 平衡HLB负样本")
            print("  python main.py --step 0                     # 生成候选肽库")
            print("  python main.py --step 1                     # 数据加载")
            print("  python main.py --all                        # 完整流程（从step0开始）")
            print("  python main.py --all --top_k 50             # 完整流程，返回前50个")
            print("  python main.py --step 3 --custom my.csv     # 预测自定义序列")
            return False

    except Exception as e:
        print(f"❌ 流程执行异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    end_time = datetime.now()
    duration = end_time - start_time

    if success:
        print(f"\n🎉 所有请求步骤已完成!")
    else:
        print(f"\n⚠️  流程未全部完成，请检查错误信息。")

    print(f"⏱️  总耗时: {duration}")
    return success


if __name__ == "__main__":
    main()