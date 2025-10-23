#!/usr/bin/env python3
"""
基于错误分析结果的改进方案
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_error_results(results_path):
    """分析错误分析结果并提出改进方案"""
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("=== 错误分析总结 ===")
    print(f"整体准确率: {results['overall_accuracy']:.2%}")
    
    # SNR性能分析
    snr_performance = results['snr_performance']
    snr_values = [float(k) for k in snr_performance.keys()]
    accuracy_values = list(snr_performance.values())
    
    print(f"\nSNR性能范围: {min(accuracy_values):.2%} - {max(accuracy_values):.2%}")
    
    # 找出低SNR性能问题
    low_snr_threshold = 0.15
    low_snr_issues = [(snr, acc) for snr, acc in zip(snr_values, accuracy_values) if acc < low_snr_threshold]
    print(f"低SNR性能问题 (<{low_snr_threshold:.0%}): {len(low_snr_issues)} 个SNR点")
    
    # 类别性能分析
    class_performance = results['class_performance']
    f1_scores = class_performance['f1']
    modulation_types = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    
    # 找出表现差的类别
    poor_performance_threshold = 0.2
    poor_classes = [(mod, f1) for mod, f1 in zip(modulation_types, f1_scores) if f1 < poor_performance_threshold]
    print(f"\n表现差的类别 (F1 < {poor_performance_threshold:.0%}): {len(poor_classes)} 个")
    for mod, f1 in poor_classes:
        print(f"  {mod}: F1 = {f1:.3f}")
    
    # 置信度分析
    confidence_analysis = results['confidence_analysis']
    avg_conf_correct = confidence_analysis['avg_confidence_correct']
    avg_conf_incorrect = confidence_analysis['avg_confidence_incorrect']
    
    print(f"\n置信度分析:")
    print(f"  正确预测平均置信度: {avg_conf_correct:.3f}")
    print(f"  错误预测平均置信度: {avg_conf_incorrect:.3f}")
    print(f"  置信度差异: {avg_conf_correct - avg_conf_incorrect:.3f}")
    
    return {
        'overall_accuracy': results['overall_accuracy'],
        'low_snr_issues': low_snr_issues,
        'poor_classes': poor_classes,
        'confidence_gap': avg_conf_correct - avg_conf_incorrect,
        'snr_performance': snr_performance,
        'class_f1_scores': dict(zip(modulation_types, f1_scores))
    }

def generate_improvement_recommendations(analysis_summary):
    """基于分析结果生成改进建议"""
    
    recommendations = []
    
    print("\n=== 改进建议 ===")
    
    # 1. 数据增强建议
    if len(analysis_summary['low_snr_issues']) > 5:
        recommendations.append({
            'category': 'data_augmentation',
            'priority': 'high',
            'title': '增强低SNR数据',
            'description': '针对低SNR场景增加数据增强技术',
            'specific_actions': [
                '添加噪声增强 (Noise Augmentation)',
                '信号衰减模拟 (Signal Attenuation)',
                '多径衰落模拟 (Multipath Fading)',
                '增加低SNR样本的权重'
            ]
        })
        print("1. 数据增强: 针对低SNR性能差的问题，建议增强数据增强技术")
    
    # 2. 模型架构建议
    if analysis_summary['overall_accuracy'] < 0.5:
        recommendations.append({
            'category': 'model_architecture',
            'priority': 'high',
            'title': '改进模型架构',
            'description': '当前模型容量可能不足，需要增强特征提取能力',
            'specific_actions': [
                '增加ResNet层数 (更深的网络)',
                '添加注意力机制 (Attention Mechanism)',
                '使用多尺度特征融合',
                '考虑使用预训练模型'
            ]
        })
        print("2. 模型架构: 整体准确率较低，建议改进模型架构")
    
    # 3. 类别不平衡处理
    if len(analysis_summary['poor_classes']) > 3:
        recommendations.append({
            'category': 'class_imbalance',
            'priority': 'medium',
            'title': '处理类别不平衡',
            'description': '多个类别表现差，可能存在类别不平衡问题',
            'specific_actions': [
                '使用加权损失函数 (Weighted Loss)',
                '焦点损失 (Focal Loss)',
                '类别平衡采样 (Balanced Sampling)',
                'SMOTE数据合成'
            ]
        })
        print("3. 类别平衡: 多个类别表现差，建议处理类别不平衡")
    
    # 4. 训练策略建议
    if analysis_summary['confidence_gap'] < 0.1:
        recommendations.append({
            'category': 'training_strategy',
            'priority': 'medium',
            'title': '改进训练策略',
            'description': '模型置信度区分度不够，需要改进训练策略',
            'specific_actions': [
                '标签平滑 (Label Smoothing)',
                '温度缩放 (Temperature Scaling)',
                '对比学习 (Contrastive Learning)',
                '知识蒸馏 (Knowledge Distillation)'
            ]
        })
        print("4. 训练策略: 置信度区分度不够，建议改进训练策略")
    
    # 5. 超参数优化建议
    recommendations.append({
        'category': 'hyperparameter_tuning',
        'priority': 'medium',
        'title': '超参数优化',
        'description': '系统性优化训练超参数',
        'specific_actions': [
            '学习率调度优化',
            '批次大小调整',
            '正则化参数调优',
            '优化器选择 (AdamW, SGD with momentum)'
        ]
    })
    print("5. 超参数: 建议系统性优化训练超参数")
    
    return recommendations

def create_quick_improvement_script(recommendations, output_dir):
    """创建快速改进实施脚本"""
    
    script_content = '''#!/usr/bin/env python3
"""
快速改进实施脚本
基于错误分析结果的针对性改进
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import json

# 导入现有模块
from src.data import DatasetLoader
from src.data import get_radioml_config
from src.models import ResNet1D

class ImprovedResNet1D(nn.Module):
    """改进的ResNet1D模型，添加注意力机制"""
    
    def __init__(self, input_channels=2, num_classes=11, layers=[2, 2, 2, 2]):
        super(ImprovedResNet1D, self).__init__()
        
        # 基础ResNet
        self.resnet = ResNet1D(input_channels=input_channels, num_classes=num_classes, layers=layers)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.attention_norm = nn.LayerNorm(512)
        
        # 改进的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # 获取ResNet特征 (需要修改ResNet以返回特征)
        features = self.resnet.features(x)  # 假设ResNet有features方法
        
        # 应用注意力
        attended_features, _ = self.attention(features, features, features)
        attended_features = self.attention_norm(attended_features + features)
        
        # 全局平均池化
        pooled_features = torch.mean(attended_features, dim=1)
        
        # 分类
        output = self.classifier(pooled_features)
        return output

class FocalLoss(nn.Module):
    """焦点损失，处理类别不平衡"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_weighted_sampler(labels):
    """创建加权采样器处理类别不平衡"""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def improved_training():
    """改进的训练流程"""
    
    print("=== 开始改进训练 ===")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据加载
    print("加载数据...")
    config = get_radioml_config('2016.10a')
    signals, labels, snrs = DatasetLoader.load_radioml_data(config['data_path'])
    
    # 数据预处理
    signals_normalized = DatasetLoader.normalize_signals(signals)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = DatasetLoader.create_dataloaders(
        signals=signals_normalized,
        labels=labels,
        snrs=snrs,
        batch_size=128,  # 增大批次大小
        test_size=0.15,
        val_size=0.15,
        augment_train=True,  # 启用数据增强
        num_workers=4
    )
    
    # 创建改进模型
    print("创建改进模型...")
    model = ImprovedResNet1D(input_channels=2, num_classes=11, layers=[3, 4, 6, 3])  # 更深的网络
    model = model.to(device)
    
    # 改进的损失函数
    criterion = FocalLoss(alpha=1, gamma=2)
    
    # 改进的优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 训练循环
    num_epochs = 20
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'improved_model_best.pth')
        
        scheduler.step()
    
    print(f"训练完成，最佳验证准确率: {best_val_acc:.2f}%")

if __name__ == "__main__":
    improved_training()
'''
    
    script_path = output_dir / "quick_improvement.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"\n快速改进脚本已保存: {script_path}")
    return script_path

def main():
    """主函数"""
    
    # 查找最新的错误分析结果
    experiments_dir = Path("experiments")
    latest_exp = max(experiments_dir.glob("simple_training_*"), key=lambda x: x.stat().st_mtime)
    results_path = latest_exp / "error_analysis" / "analysis_results.json"
    
    if not results_path.exists():
        print(f"错误分析结果文件不存在: {results_path}")
        return
    
    print(f"分析错误结果: {results_path}")
    
    # 分析错误结果
    analysis_summary = analyze_error_results(results_path)
    
    # 生成改进建议
    recommendations = generate_improvement_recommendations(analysis_summary)
    
    # 保存改进建议
    output_dir = latest_exp / "improvement_plan"
    output_dir.mkdir(exist_ok=True)
    
    recommendations_path = output_dir / "recommendations.json"
    with open(recommendations_path, 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)
    
    print(f"\n改进建议已保存: {recommendations_path}")
    
    # 创建快速改进脚本
    script_path = create_quick_improvement_script(recommendations, output_dir)
    
    print("\n=== 总结 ===")
    print(f"1. 错误分析完成，整体准确率: {analysis_summary['overall_accuracy']:.2%}")
    print(f"2. 识别出 {len(analysis_summary['low_snr_issues'])} 个低SNR性能问题")
    print(f"3. 识别出 {len(analysis_summary['poor_classes'])} 个表现差的类别")
    print(f"4. 生成了 {len(recommendations)} 个改进建议")
    print(f"5. 创建了快速改进实施脚本")
    
    print("\n建议优先级:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} (优先级: {rec['priority']})")

if __name__ == "__main__":
    main()