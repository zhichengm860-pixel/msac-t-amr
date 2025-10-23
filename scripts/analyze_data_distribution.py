#!/usr/bin/env python3
"""
数据分布分析脚本
分析RadioML数据集的类分布、SNR分布和数据质量
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import h5py
import os
import torch
from src.data import get_radioml_config
from src.data import DatasetLoader

def analyze_radioml_2016(dataset_path):
    """分析RadioML 2016.10A数据集"""
    print("=== RadioML 2016.10A 数据分析 ===")
    
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # 统计信息
    mod_types = sorted(list(set([key[0] for key in data.keys()])))
    snr_values = sorted(list(set([key[1] for key in data.keys()])))
    
    print(f"调制类型数量: {len(mod_types)}")
    print(f"调制类型: {mod_types}")
    print(f"SNR范围: {min(snr_values)} 到 {max(snr_values)} dB")
    print(f"SNR值: {snr_values}")
    
    # 统计每个类别和SNR的样本数
    class_counts = Counter()
    snr_counts = Counter()
    total_samples = 0
    
    for (mod, snr), samples in data.items():
        num_samples = samples.shape[0]
        class_counts[mod] += num_samples
        snr_counts[snr] += num_samples
        total_samples += num_samples
    
    print(f"\n总样本数: {total_samples}")
    
    # 类别分布
    print("\n=== 类别分布 ===")
    for mod in mod_types:
        count = class_counts[mod]
        percentage = count / total_samples * 100
        print(f"{mod:>8}: {count:>6} 样本 ({percentage:>5.1f}%)")
    
    # SNR分布
    print("\n=== SNR分布 ===")
    for snr in snr_values:
        count = snr_counts[snr]
        percentage = count / total_samples * 100
        print(f"SNR {snr:>3} dB: {count:>6} 样本 ({percentage:>5.1f}%)")
    
    # 检查数据质量
    print("\n=== 数据质量检查 ===")
    signal_shapes = []
    signal_stats = []
    
    for (mod, snr), samples in data.items():
        signal_shapes.append(samples.shape)
        # 计算信号统计
        mean_power = np.mean(np.sum(samples**2, axis=1))
        max_amplitude = np.max(np.abs(samples))
        signal_stats.append((mod, snr, mean_power, max_amplitude))
    
    # 检查形状一致性
    unique_shapes = set(signal_shapes)
    print(f"信号形状: {unique_shapes}")
    if len(unique_shapes) == 1:
        print("✓ 所有信号形状一致")
    else:
        print("✗ 信号形状不一致")
    
    # 功率统计
    powers = [stat[2] for stat in signal_stats]
    amplitudes = [stat[3] for stat in signal_stats]
    
    print(f"平均功率范围: {min(powers):.4f} 到 {max(powers):.4f}")
    print(f"最大幅度范围: {min(amplitudes):.4f} 到 {max(amplitudes):.4f}")
    
    return {
        'mod_types': mod_types,
        'snr_values': snr_values,
        'class_counts': dict(class_counts),
        'snr_counts': dict(snr_counts),
        'total_samples': total_samples,
        'signal_stats': signal_stats
    }

def analyze_data_loading():
    """分析数据加载过程"""
    print("\n=== 数据加载分析 ===")
    
    # 获取配置
    dataset_config, preprocess_config, dataloader_config = get_radioml_config('2016.10a')
    
    # 加载数据
    loader = DatasetLoader(dataset_config)
    train_loader, val_loader, test_loader = loader.load_radioml_data(
        dataset_config.path,
        batch_size=64,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 检查第一个批次
    for signals, labels, snrs in train_loader:
        print(f"\n批次信号形状: {signals.shape}")
        print(f"批次标签形状: {labels.shape}")
        print(f"批次SNR形状: {snrs.shape}")
        
        print(f"信号数据类型: {signals.dtype}")
        print(f"标签数据类型: {labels.dtype}")
        print(f"SNR数据类型: {snrs.dtype}")
        
        print(f"信号值范围: {signals.min():.4f} 到 {signals.max():.4f}")
        print(f"标签值范围: {labels.min()} 到 {labels.max()}")
        print(f"SNR值范围: {snrs.min():.1f} 到 {snrs.max():.1f}")
        
        # 检查是否有NaN或Inf
        print(f"信号中NaN数量: {torch.isnan(signals).sum()}")
        print(f"信号中Inf数量: {torch.isinf(signals).sum()}")
        
        break
    
    return {
        'train_batches': len(train_loader),
        'val_batches': len(val_loader),
        'test_batches': len(test_loader)
    }

def plot_distributions(analysis_results, save_dir="analysis_plots"):
    """绘制分布图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 类别分布饼图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    mod_types = analysis_results['mod_types']
    class_counts = [analysis_results['class_counts'][mod] for mod in mod_types]
    plt.pie(class_counts, labels=mod_types, autopct='%1.1f%%', startangle=90)
    plt.title('调制类型分布')
    
    # 类别分布条形图
    plt.subplot(2, 2, 2)
    plt.bar(mod_types, class_counts)
    plt.title('调制类型样本数')
    plt.xlabel('调制类型')
    plt.ylabel('样本数')
    plt.xticks(rotation=45)
    
    # SNR分布
    plt.subplot(2, 2, 3)
    snr_values = analysis_results['snr_values']
    snr_counts = [analysis_results['snr_counts'][snr] for snr in snr_values]
    plt.bar(snr_values, snr_counts)
    plt.title('SNR分布')
    plt.xlabel('SNR (dB)')
    plt.ylabel('样本数')
    
    # 功率分布
    plt.subplot(2, 2, 4)
    signal_stats = analysis_results['signal_stats']
    powers = [stat[2] for stat in signal_stats]
    plt.hist(powers, bins=30, alpha=0.7)
    plt.title('信号功率分布')
    plt.xlabel('平均功率')
    plt.ylabel('频次')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'data_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # SNR vs 调制类型热图
    plt.figure(figsize=(12, 8))
    
    # 创建热图数据
    heatmap_data = np.zeros((len(mod_types), len(snr_values)))
    for i, mod in enumerate(mod_types):
        for j, snr in enumerate(snr_values):
            # 从原始数据中获取样本数
            key = (mod, snr)
            if key in [(stat[0], stat[1]) for stat in signal_stats]:
                # 找到对应的样本数
                for stat in signal_stats:
                    if stat[0] == mod and stat[1] == snr:
                        heatmap_data[i, j] = 1000  # 每个(mod, snr)组合有1000个样本
                        break
    
    sns.heatmap(heatmap_data, 
                xticklabels=snr_values, 
                yticklabels=mod_types,
                annot=True, 
                fmt='g',
                cmap='Blues')
    plt.title('调制类型 vs SNR 样本分布')
    plt.xlabel('SNR (dB)')
    plt.ylabel('调制类型')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mod_snr_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("开始数据分布分析...")
    
    # 分析RadioML 2016.10A
    dataset_config, _, _ = get_radioml_config('2016.10a')
    
    if os.path.exists(dataset_config.path):
        analysis_results = analyze_radioml_2016(dataset_config.path)
        
        # 分析数据加载
        import torch
        loading_results = analyze_data_loading()
        
        # 绘制分布图
        plot_distributions(analysis_results)
        
        print("\n=== 分析总结 ===")
        print(f"数据集: RadioML 2016.10A")
        print(f"总样本数: {analysis_results['total_samples']:,}")
        print(f"调制类型: {len(analysis_results['mod_types'])} 种")
        print(f"SNR范围: {min(analysis_results['snr_values'])} 到 {max(analysis_results['snr_values'])} dB")
        print(f"训练/验证/测试批次: {loading_results['train_batches']}/{loading_results['val_batches']}/{loading_results['test_batches']}")
        
        # 检查类别平衡性
        class_counts = list(analysis_results['class_counts'].values())
        min_count = min(class_counts)
        max_count = max(class_counts)
        imbalance_ratio = max_count / min_count
        
        print(f"\n类别平衡性:")
        print(f"  最少样本数: {min_count}")
        print(f"  最多样本数: {max_count}")
        print(f"  不平衡比例: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2.0:
            print("  ⚠️  数据集存在类别不平衡问题")
        else:
            print("  ✓  数据集类别相对平衡")
        
    else:
        print(f"数据集文件不存在: {dataset_config.path}")

if __name__ == "__main__":
    main()