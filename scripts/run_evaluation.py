#!/usr/bin/env python3
"""
模型评估脚本
对训练好的模型进行全面的性能评估
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from src.utils import Config
from src.models import ResNet1D
from src.data import DatasetLoader
from src.data import get_radioml_config

def load_trained_model(checkpoint_path, device):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")
    
    # 创建模型
    model = ResNet1D(
        input_channels=2,
        num_classes=11,  # RadioML2016有11个调制类型
        layers=[1, 1, 1, 1]  # 匹配训练时的层配置
    )
    
    # 加载权重
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("模型加载成功")
    else:
        print(f"警告: 未找到模型文件 {checkpoint_path}")
        return None
    
    model.to(device)
    model.eval()
    return model

def evaluate_comprehensive(model, test_loader, device, class_names):
    """全面评估模型性能"""
    print("\n=== 开始全面评估 ===")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_snrs = []
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (signals, labels, snrs) in enumerate(test_loader):
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 获取预测概率
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_snrs.extend(snrs.numpy())
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    all_snrs = np.array(all_snrs)
    
    # 计算基本指标
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_loss = total_loss / len(test_loader)
    
    # 计算精确度、召回率、F1分数
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # 计算宏平均和微平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='micro', zero_division=0
    )
    
    # 按SNR分析性能
    snr_performance = analyze_snr_performance(all_labels, all_predictions, all_snrs)
    
    results = {
        'overall_accuracy': float(accuracy),
        'average_loss': float(avg_loss),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'micro_precision': float(micro_precision),
        'micro_recall': float(micro_recall),
        'micro_f1': float(micro_f1),
        'per_class_metrics': {
            'precision': [float(p) for p in precision],
            'recall': [float(r) for r in recall],
            'f1': [float(f) for f in f1],
            'support': [int(s) for s in support]
        },
        'snr_performance': snr_performance,
        'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
        'class_names': class_names
    }
    
    return results, all_labels, all_predictions, all_probabilities, all_snrs

def analyze_snr_performance(labels, predictions, snrs):
    """分析不同SNR下的性能"""
    unique_snrs = np.unique(snrs)
    snr_results = {}
    
    for snr in unique_snrs:
        mask = snrs == snr
        if np.sum(mask) > 0:
            snr_labels = labels[mask]
            snr_predictions = predictions[mask]
            snr_accuracy = accuracy_score(snr_labels, snr_predictions)
            snr_results[int(snr)] = {
                'accuracy': float(snr_accuracy),
                'samples': int(np.sum(mask))
            }
    
    return snr_results

def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵保存到: {save_path}")

def plot_snr_performance(snr_performance, save_path):
    """绘制SNR性能图"""
    snrs = sorted(snr_performance.keys())
    accuracies = [snr_performance[snr]['accuracy'] for snr in snrs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, accuracies, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Model Performance vs SNR')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SNR性能图保存到: {save_path}")

def plot_class_performance(results, save_path):
    """绘制各类别性能图"""
    class_names = results['class_names']
    precision = results['per_class_metrics']['precision']
    recall = results['per_class_metrics']['recall']
    f1 = results['per_class_metrics']['f1']
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Modulation Types')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"类别性能图保存到: {save_path}")

def main():
    """主函数"""
    try:
        print("=== 模型评估开始 ===")
        
        # 设置设备
        device = torch.device('cpu')
        print(f"使用设备: {device}")
        
        # 创建评估目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = f"experiments/evaluation_{timestamp}"
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(os.path.join(eval_dir, "plots"), exist_ok=True)
        
        # 加载数据
        print("\n=== 加载测试数据 ===")
        dataset_config, preprocess_config, dataloader_config = get_radioml_config('2016.10a')
        loader = DatasetLoader(dataset_config)
        _, _, test_loader = loader.load_radioml_data(
            dataset_config.path,
            batch_size=32,
            train_ratio=0.8,
            val_ratio=0.1
        )
        
        # 定义类别名称
        class_names = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'BFSK', 'CPFSK', 'PAM4', 'WBFM', 'AM-SSB', 'AM-DSB']
        
        # 查找最新的训练模型
        simple_training_dirs = [d for d in os.listdir('experiments') if d.startswith('simple_training_')]
        if not simple_training_dirs:
            print("错误: 未找到训练好的模型")
            return
        
        latest_dir = sorted(simple_training_dirs)[-1]
        checkpoint_path = os.path.join('experiments', latest_dir, 'checkpoints', 'best_model.pth')
        
        # 加载模型
        model = load_trained_model(checkpoint_path, device)
        if model is None:
            print("错误: 模型加载失败")
            return
        
        # 执行评估
        results, labels, predictions, probabilities, snrs = evaluate_comprehensive(
            model, test_loader, device, class_names
        )
        
        # 打印结果
        print(f"\n=== 评估结果 ===")
        print(f"整体准确率: {results['overall_accuracy']*100:.2f}%")
        print(f"宏平均 F1: {results['macro_f1']:.4f}")
        print(f"微平均 F1: {results['micro_f1']:.4f}")
        
        # 生成可视化
        print(f"\n=== 生成可视化 ===")
        
        # 混淆矩阵
        cm_path = os.path.join(eval_dir, "plots", "confusion_matrix.png")
        plot_confusion_matrix(np.array(results['confusion_matrix']), class_names, cm_path)
        
        # SNR性能
        snr_path = os.path.join(eval_dir, "plots", "snr_performance.png")
        plot_snr_performance(results['snr_performance'], snr_path)
        
        # 类别性能
        class_path = os.path.join(eval_dir, "plots", "class_performance.png")
        plot_class_performance(results, class_path)
        
        # 保存结果
        results_path = os.path.join(eval_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== 评估完成 ===")
        print(f"结果保存在: {eval_dir}")
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()