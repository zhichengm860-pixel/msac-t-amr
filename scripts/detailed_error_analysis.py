#!/usr/bin/env python3
"""
详细错误分析脚本
分析模型性能问题，包括混淆矩阵、SNR分层分析、类别错误分析等
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
from pathlib import Path

from src.data import DatasetLoader
from src.data import get_radioml_config
from src.models import ResNet1D
from src.evaluation import ModelEvaluator, MetricsCalculator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_latest_model():
    """加载最新的训练模型"""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        raise FileNotFoundError("未找到experiments目录")
    
    # 找到最新的实验目录
    exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith("simple_training")]
    if not exp_dirs:
        raise FileNotFoundError("未找到训练实验目录")
    
    latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
    model_path = latest_exp / "checkpoints" / "best_model.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    
    print(f"加载模型: {model_path}")
    return str(model_path), str(latest_exp)

def analyze_model_predictions(model, test_loader, device, modulation_types):
    """分析模型预测结果"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_snrs = []
    all_probs = []
    
    with torch.no_grad():
        for signals, labels, snrs in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_snrs.extend(snrs.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_snrs), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, modulation_types, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=modulation_types,
                yticklabels=modulation_types)
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm

def analyze_snr_performance(y_true, y_pred, snrs, modulation_types, save_path):
    """分析不同SNR下的性能"""
    unique_snrs = sorted(np.unique(snrs))
    snr_accuracies = []
    
    for snr in unique_snrs:
        mask = snrs == snr
        if np.sum(mask) > 0:
            acc = np.mean(y_true[mask] == y_pred[mask])
            snr_accuracies.append(acc)
        else:
            snr_accuracies.append(0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(unique_snrs, snr_accuracies, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('信噪比 (dB)')
    plt.ylabel('准确率')
    plt.title('不同信噪比下的准确率')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return dict(zip(unique_snrs, snr_accuracies))

def analyze_class_performance(y_true, y_pred, modulation_types, save_path):
    """分析各类别性能"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # 创建性能对比图
    x = np.arange(len(modulation_types))
    width = 0.25
    
    plt.figure(figsize=(15, 6))
    plt.bar(x - width, precision, width, label='精确率', alpha=0.8)
    plt.bar(x, recall, width, label='召回率', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1分数', alpha=0.8)
    
    plt.xlabel('调制类型')
    plt.ylabel('分数')
    plt.title('各调制类型性能对比')
    plt.xticks(x, modulation_types, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }

def analyze_prediction_confidence(probs, y_true, y_pred, save_path):
    """分析预测置信度"""
    max_probs = np.max(probs, axis=1)
    correct_mask = y_true == y_pred
    
    plt.figure(figsize=(12, 5))
    
    # 正确预测的置信度分布
    plt.subplot(1, 2, 1)
    plt.hist(max_probs[correct_mask], bins=50, alpha=0.7, label='正确预测', color='green')
    plt.hist(max_probs[~correct_mask], bins=50, alpha=0.7, label='错误预测', color='red')
    plt.xlabel('最大预测概率')
    plt.ylabel('频次')
    plt.title('预测置信度分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 置信度与准确率关系
    plt.subplot(1, 2, 2)
    confidence_bins = np.linspace(0, 1, 11)
    bin_accuracies = []
    bin_centers = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i+1])
        if np.sum(mask) > 0:
            acc = np.mean(correct_mask[mask])
            bin_accuracies.append(acc)
            bin_centers.append((confidence_bins[i] + confidence_bins[i+1]) / 2)
    
    plt.plot(bin_centers, bin_accuracies, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('预测置信度')
    plt.ylabel('准确率')
    plt.title('置信度与准确率关系')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'avg_confidence_correct': np.mean(max_probs[correct_mask]),
        'avg_confidence_incorrect': np.mean(max_probs[~correct_mask]),
        'confidence_accuracy_relation': list(zip(bin_centers, bin_accuracies))
    }

def main():
    """主函数"""
    print("=== 详细错误分析 ===")
    
    # 设置设备
    device = torch.device('cpu')  # 强制使用CPU
    print(f"使用设备: {device}")
    
    # 加载数据集配置
    config, _, _ = get_radioml_config('2016.10a')
    modulation_types = config.modulation_types
    print(f"调制类型: {modulation_types}")
    
    # 加载数据
    print("加载测试数据...")
    loader = DatasetLoader(config)
    signals, labels, snrs, mod_types = loader.load_radioml2016(config.path)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = DatasetLoader.create_dataloaders(
        signals, labels, snrs,
        batch_size=64,
        test_size=0.15,
        val_size=0.15,
        augment_train=False,
        num_workers=0
    )
    
    # 加载模型
    model_path, exp_dir = load_latest_model()
    model = ResNet1D(
        input_channels=2,
        num_classes=len(modulation_types),
        layers=[1, 1, 1, 1]  # 与训练时保持一致
    ).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("模型加载完成")
    
    # 获取预测结果
    print("分析模型预测...")
    predictions, labels, snrs, probs = analyze_model_predictions(
        model, test_loader, device, modulation_types
    )
    
    # 创建分析结果目录
    analysis_dir = Path(exp_dir) / "error_analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # 1. 混淆矩阵分析
    print("生成混淆矩阵...")
    cm = plot_confusion_matrix(
        labels, predictions, modulation_types,
        analysis_dir / "confusion_matrix.png"
    )
    
    # 2. SNR性能分析
    print("分析SNR性能...")
    snr_performance = analyze_snr_performance(
        labels, predictions, snrs, modulation_types,
        analysis_dir / "snr_performance.png"
    )
    
    # 3. 类别性能分析
    print("分析类别性能...")
    class_performance = analyze_class_performance(
        labels, predictions, modulation_types,
        analysis_dir / "class_performance.png"
    )
    
    # 4. 预测置信度分析
    print("分析预测置信度...")
    confidence_analysis = analyze_prediction_confidence(
        probs, labels, predictions,
        analysis_dir / "confidence_analysis.png"
    )
    
    # 5. 生成详细报告
    overall_accuracy = np.mean(labels == predictions)
    report = classification_report(labels, predictions, target_names=modulation_types, output_dict=True)
    
    # 保存分析结果
    analysis_results = {
        'overall_accuracy': float(overall_accuracy),
        'snr_performance': {str(k): float(v) for k, v in snr_performance.items()},
        'class_performance': {
            'precision': [float(x) for x in class_performance['precision']],
            'recall': [float(x) for x in class_performance['recall']],
            'f1': [float(x) for x in class_performance['f1']],
            'support': [int(x) for x in class_performance['support']]
        },
        'confidence_analysis': {
            'avg_confidence_correct': float(confidence_analysis['avg_confidence_correct']),
            'avg_confidence_incorrect': float(confidence_analysis['avg_confidence_incorrect']),
            'confidence_accuracy_relation': [(float(x), float(y)) for x, y in confidence_analysis['confidence_accuracy_relation']]
        },
        'classification_report': {k: (v if isinstance(v, (str, int, float, np.number)) else {k2: float(v2) if isinstance(v2, (int, float, np.number)) else v2 for k2, v2 in v.items()}) for k, v in report.items()},
        'confusion_matrix': [[int(x) for x in row] for row in cm.tolist()]
    }
    
    with open(analysis_dir / "analysis_results.json", 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    # 打印关键发现
    print("\n=== 关键发现 ===")
    print(f"整体准确率: {overall_accuracy:.2%}")
    print(f"最佳SNR性能: {max(snr_performance.values()):.2%} (SNR: {max(snr_performance, key=snr_performance.get)} dB)")
    print(f"最差SNR性能: {min(snr_performance.values()):.2%} (SNR: {min(snr_performance, key=snr_performance.get)} dB)")
    
    # 找出表现最差的类别
    worst_class_idx = np.argmin(class_performance['f1'])
    best_class_idx = np.argmax(class_performance['f1'])
    print(f"表现最差类别: {modulation_types[worst_class_idx]} (F1: {class_performance['f1'][worst_class_idx]:.3f})")
    print(f"表现最好类别: {modulation_types[best_class_idx]} (F1: {class_performance['f1'][best_class_idx]:.3f})")
    
    print(f"\n分析结果保存在: {analysis_dir}")

if __name__ == "__main__":
    main()