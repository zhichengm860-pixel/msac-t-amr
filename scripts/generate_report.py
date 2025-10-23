#!/usr/bin/env python3
"""
实验报告生成脚本
汇总所有实验结果并生成综合报告
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

def find_latest_experiments():
    """查找最新的实验结果"""
    experiments = {}
    
    # 查找简化训练实验
    simple_dirs = glob.glob("experiments/simple_training_*")
    if simple_dirs:
        latest_simple = sorted(simple_dirs)[-1]
        experiments['simple_training'] = latest_simple
    
    # 查找评估实验
    eval_dirs = glob.glob("experiments/evaluation_*")
    if eval_dirs:
        latest_eval = sorted(eval_dirs)[-1]
        experiments['evaluation'] = latest_eval
    
    # 查找基线对比实验
    baseline_dirs = glob.glob("experiments/baseline_comparison_*")
    if baseline_dirs:
        latest_baseline = sorted(baseline_dirs)[-1]
        experiments['baseline_comparison'] = latest_baseline
    
    # 查找预训练实验
    pretrain_dirs = glob.glob("experiments/pretrain_lite_*")
    if pretrain_dirs:
        latest_pretrain = sorted(pretrain_dirs)[-1]
        experiments['pretrain'] = latest_pretrain
    
    return experiments

def load_experiment_results(experiments):
    """加载实验结果"""
    results = {}
    
    # 加载简化训练结果
    if 'simple_training' in experiments:
        simple_path = os.path.join(experiments['simple_training'], 'results.json')
        if os.path.exists(simple_path):
            with open(simple_path, 'r') as f:
                results['simple_training'] = json.load(f)
    
    # 加载评估结果
    if 'evaluation' in experiments:
        eval_path = os.path.join(experiments['evaluation'], 'evaluation_results.json')
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                results['evaluation'] = json.load(f)
    
    # 加载基线对比结果
    if 'baseline_comparison' in experiments:
        baseline_path = os.path.join(experiments['baseline_comparison'], 'comparison_results.json')
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r') as f:
                results['baseline_comparison'] = json.load(f)
    
    # 加载预训练结果
    if 'pretrain' in experiments:
        pretrain_path = os.path.join(experiments['pretrain'], 'results.json')
        if os.path.exists(pretrain_path):
            with open(pretrain_path, 'r') as f:
                results['pretrain'] = json.load(f)
    
    return results

def create_report_directory():
    """创建报告目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"experiments/final_report_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(os.path.join(report_dir, "plots"), exist_ok=True)
    return report_dir

def plot_model_comparison(results, report_dir):
    """绘制模型对比图"""
    if 'baseline_comparison' not in results:
        print("警告: 未找到基线对比结果")
        return
    
    baseline_results = results['baseline_comparison']
    
    # 准备数据
    models = []
    accuracies = []
    parameters = []
    training_times = []
    
    for model_name, model_data in baseline_results.items():
        models.append(model_name)
        accuracies.append(model_data['test_results']['test_accuracy'] * 100)
        parameters.append(model_data['parameters'])
        training_times.append(model_data['training_time'])
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 准确率对比
    bars1 = ax1.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_ylim(0, max(accuracies) * 1.2)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom')
    
    # 2. 参数数量对比
    bars2 = ax2.bar(models, [p/1000000 for p in parameters], color=['skyblue', 'lightcoral', 'lightgreen'])
    ax2.set_title('Model Parameters Comparison')
    ax2.set_ylabel('Parameters (Millions)')
    for i, v in enumerate(parameters):
        ax2.text(i, v/1000000 + 0.1, f'{v/1000000:.2f}M', ha='center', va='bottom')
    
    # 3. 训练时间对比
    bars3 = ax3.bar(models, training_times, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax3.set_title('Training Time Comparison')
    ax3.set_ylabel('Training Time (seconds)')
    for i, v in enumerate(training_times):
        ax3.text(i, v + 1, f'{v:.1f}s', ha='center', va='bottom')
    
    # 4. 效率对比 (准确率/参数数量)
    efficiency = [acc / (param / 1000000) for acc, param in zip(accuracies, parameters)]
    bars4 = ax4.bar(models, efficiency, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax4.set_title('Model Efficiency (Accuracy/Million Parameters)')
    ax4.set_ylabel('Efficiency Score')
    for i, v in enumerate(efficiency):
        ax4.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "plots", "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"模型对比图保存到: {os.path.join(report_dir, 'plots', 'model_comparison.png')}")

def plot_training_curves(results, report_dir):
    """绘制训练曲线"""
    if 'simple_training' not in results:
        print("警告: 未找到训练结果")
        return
    
    train_history = results['simple_training']['train_history']
    
    epochs = [h['epoch'] for h in train_history]
    train_losses = [h['train_loss'] for h in train_history]
    val_losses = [h['val_loss'] for h in train_history]
    train_accs = [h['train_acc'] for h in train_history]
    val_accs = [h['val_acc'] for h in train_history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "plots", "training_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练曲线图保存到: {os.path.join(report_dir, 'plots', 'training_curves.png')}")

def generate_summary_table(results):
    """生成结果汇总表"""
    summary = {}
    
    # 预训练结果
    if 'pretrain' in results:
        pretrain_data = results['pretrain']
        summary['预训练实验'] = {
            '最佳验证损失': f"{pretrain_data.get('best_val_loss', 'N/A'):.4f}" if isinstance(pretrain_data.get('best_val_loss'), (int, float)) else 'N/A',
            '最终训练损失': f"{pretrain_data.get('final_train_loss', 'N/A'):.4f}" if isinstance(pretrain_data.get('final_train_loss'), (int, float)) else 'N/A',
            '训练轮数': pretrain_data.get('epochs', 'N/A')
        }
    
    # 主模型训练结果
    if 'simple_training' in results:
        simple_data = results['simple_training']
        summary['主模型训练'] = {
            '最佳验证准确率': f"{simple_data.get('best_val_accuracy', 'N/A'):.2f}%" if isinstance(simple_data.get('best_val_accuracy'), (int, float)) else 'N/A',
            '测试准确率': f"{simple_data['test_results']['test_accuracy']*100:.2f}%" if 'test_results' in simple_data else 'N/A',
            '测试损失': f"{simple_data['test_results']['test_loss']:.4f}" if 'test_results' in simple_data else 'N/A'
        }
    
    # 详细评估结果
    if 'evaluation' in results:
        eval_data = results['evaluation']
        summary['详细评估'] = {
            '整体准确率': f"{eval_data.get('overall_accuracy', 0)*100:.2f}%",
            '宏平均F1': f"{eval_data.get('macro_f1', 0):.4f}",
            '微平均F1': f"{eval_data.get('micro_f1', 0):.4f}",
            '宏平均精确度': f"{eval_data.get('macro_precision', 0):.4f}",
            '宏平均召回率': f"{eval_data.get('macro_recall', 0):.4f}"
        }
    
    # 基线对比结果
    if 'baseline_comparison' in results:
        baseline_data = results['baseline_comparison']
        summary['基线模型对比'] = {}
        for model_name, model_data in baseline_data.items():
            summary['基线模型对比'][model_name] = {
                '测试准确率': f"{model_data['test_results']['test_accuracy']*100:.2f}%",
                '参数数量': f"{model_data['parameters']:,}",
                '训练时间': f"{model_data['training_time']:.2f}s"
            }
    
    return summary

def generate_markdown_report(results, summary, report_dir):
    """生成Markdown格式的报告"""
    report_path = os.path.join(report_dir, "experiment_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 无线电调制识别模型实验报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验概述\n\n")
        f.write("本报告汇总了融合多尺度分析与复数注意力机制的鲁棒无线电调制识别模型的实验结果。\n\n")
        
        f.write("## 实验结果汇总\n\n")
        
        for section, data in summary.items():
            f.write(f"### {section}\n\n")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        f.write(f"**{key}**:\n")
                        for subkey, subvalue in value.items():
                            f.write(f"- {subkey}: {subvalue}\n")
                        f.write("\n")
                    else:
                        f.write(f"- **{key}**: {value}\n")
            f.write("\n")
        
        f.write("## 主要发现\n\n")
        
        # 分析基线对比结果
        if 'baseline_comparison' in results:
            baseline_data = results['baseline_comparison']
            best_model = max(baseline_data.items(), 
                           key=lambda x: x[1]['test_results']['test_accuracy'])
            f.write(f"1. **最佳基线模型**: {best_model[0]}，测试准确率为 {best_model[1]['test_results']['test_accuracy']*100:.2f}%\n")
            
            # 参数效率分析
            efficiency_scores = {}
            for name, data in baseline_data.items():
                acc = data['test_results']['test_accuracy'] * 100
                params = data['parameters'] / 1000000  # 转换为百万
                efficiency_scores[name] = acc / params
            
            best_efficiency = max(efficiency_scores.items(), key=lambda x: x[1])
            f.write(f"2. **最高效率模型**: {best_efficiency[0]}，效率分数为 {best_efficiency[1]:.2f} (准确率/百万参数)\n")
        
        # 分析训练表现
        if 'simple_training' in results:
            simple_data = results['simple_training']
            f.write(f"3. **训练收敛性**: 模型在 {len(simple_data['train_history'])} 轮训练后达到最佳验证准确率 {simple_data['best_val_accuracy']:.2f}%\n")
        
        f.write("\n## 可视化结果\n\n")
        f.write("- 模型对比图: `plots/model_comparison.png`\n")
        f.write("- 训练曲线图: `plots/training_curves.png`\n")
        if 'evaluation' in results:
            f.write("- 混淆矩阵: 详见评估实验目录\n")
            f.write("- SNR性能分析: 详见评估实验目录\n")
        
        f.write("\n## 结论\n\n")
        f.write("实验成功验证了模型的有效性，并通过与基线模型的对比展示了不同架构的性能特点。")
        f.write("详细的评估结果为进一步的模型优化提供了重要参考。\n")
    
    print(f"Markdown报告保存到: {report_path}")

def main():
    """主函数"""
    try:
        print("=== 生成实验报告 ===")
        
        # 查找实验结果
        experiments = find_latest_experiments()
        print(f"找到 {len(experiments)} 个实验:")
        for exp_type, exp_path in experiments.items():
            print(f"  {exp_type}: {exp_path}")
        
        if not experiments:
            print("错误: 未找到任何实验结果")
            return
        
        # 加载实验结果
        print("\n=== 加载实验结果 ===")
        results = load_experiment_results(experiments)
        print(f"成功加载 {len(results)} 个实验的结果")
        
        # 创建报告目录
        report_dir = create_report_directory()
        print(f"报告目录: {report_dir}")
        
        # 生成可视化
        print("\n=== 生成可视化 ===")
        plot_model_comparison(results, report_dir)
        plot_training_curves(results, report_dir)
        
        # 生成汇总表
        print("\n=== 生成结果汇总 ===")
        summary = generate_summary_table(results)
        
        # 保存汇总结果
        summary_path = os.path.join(report_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        print("\n=== 生成Markdown报告 ===")
        generate_markdown_report(results, summary, report_dir)
        
        print(f"\n=== 报告生成完成 ===")
        print(f"报告保存在: {report_dir}")
        print("包含文件:")
        print("  - experiment_report.md (主报告)")
        print("  - summary.json (结果汇总)")
        print("  - plots/model_comparison.png (模型对比图)")
        print("  - plots/training_curves.png (训练曲线图)")
        
    except Exception as e:
        print(f"报告生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()