#!/usr/bin/env python3
"""
run_optimized_experiment.py - 基于策略分析结果的优化实验

使用最佳训练策略进行模型训练：
- 策略：大批量+低学习率
- 配置：optimizer=adam, lr=0.0003, batch_size=128, weight_decay=0.0001
- 预期性能：测试准确率 ~36.93%
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import time
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.improved_msac_t import ImprovedMSAC_T
from src.data.radioml_dataloader import RadioMLDataLoader
from src.training.optimized_trainer import OptimizedTrainer
from src.utils.config import TrainingConfig, ModelConfig
from src.evaluation.evaluation import evaluate_model


def load_optimized_data(data_subset_ratio: float = 0.2):
    """
    加载优化后的数据集
    
    Args:
        data_subset_ratio: 数据子集比例，用于快速验证
    """
    print(f"🔄 加载数据集 (使用 {data_subset_ratio*100:.1f}% 数据)...")
    
    # 数据集路径
    dataset_path = "dataset/RadioML 2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    # 加载数据
    dataloader = RadioMLDataLoader(dataset_path)
    
    # 获取数据子集
    train_data, train_labels, val_data, val_labels, test_data, test_labels = dataloader.load_data()
    
    # 如果使用数据子集
    if data_subset_ratio < 1.0:
        train_size = int(len(train_data) * data_subset_ratio)
        val_size = int(len(val_data) * data_subset_ratio)
        test_size = int(len(test_data) * data_subset_ratio)
        
        # 随机采样
        train_indices = np.random.choice(len(train_data), train_size, replace=False)
        val_indices = np.random.choice(len(val_data), val_size, replace=False)
        test_indices = np.random.choice(len(test_data), test_size, replace=False)
        
        train_data = train_data[train_indices]
        train_labels = train_labels[train_indices]
        val_data = val_data[val_indices]
        val_labels = val_labels[val_indices]
        test_data = test_data[test_indices]
        test_labels = test_labels[test_indices]
    
    print(f"✅ 数据加载完成:")
    print(f"   训练集: {len(train_data)} 样本")
    print(f"   验证集: {len(val_data)} 样本")
    print(f"   测试集: {len(test_data)} 样本")
    
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def create_optimized_dataloaders(train_data, train_labels, val_data, val_labels, test_data, test_labels):
    """创建优化的数据加载器"""
    
    # 使用最佳策略的批量大小
    batch_size = 128
    
    # 创建数据集
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_data),
        torch.LongTensor(train_labels)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_data),
        torch.LongTensor(val_labels)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_data),
        torch.LongTensor(test_labels)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ 数据加载器创建完成 (batch_size={batch_size})")
    
    return train_loader, val_loader, test_loader


def create_optimized_model():
    """创建优化的模型"""
    print("🔄 创建优化模型...")
    
    # 模型配置
    model_config = ModelConfig()
    
    # 基于策略分析结果优化模型参数
    model_config.feature_dim = 256
    model_config.num_heads = 8
    model_config.attention_dropout = 0.1
    model_config.dropout = 0.3  # 适度降低dropout以配合大批量训练
    
    # 创建模型
    model = ImprovedMSAC_T(
        num_classes=model_config.num_classes,
        feature_dim=model_config.feature_dim,
        num_heads=model_config.num_heads,
        dropout=model_config.dropout
    )
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ 模型创建完成:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    return model


def run_optimized_experiment(data_subset_ratio: float = 0.2, epochs: int = 50):
    """
    运行优化实验
    
    Args:
        data_subset_ratio: 数据子集比例
        epochs: 训练轮数
    """
    print("🚀 开始优化实验")
    print("=" * 60)
    print("基于策略分析结果的最佳配置:")
    print("- 策略: 大批量+低学习率")
    print("- 优化器: Adam")
    print("- 学习率: 0.0003")
    print("- 批量大小: 128")
    print("- 权重衰减: 0.0001")
    print("- 预期测试准确率: ~36.93%")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    try:
        # 1. 加载数据
        train_data, train_labels, val_data, val_labels, test_data, test_labels = load_optimized_data(data_subset_ratio)
        
        # 2. 创建数据加载器
        train_loader, val_loader, test_loader = create_optimized_dataloaders(
            train_data, train_labels, val_data, val_labels, test_data, test_labels
        )
        
        # 3. 创建模型
        model = create_optimized_model()
        
        # 4. 创建优化训练器
        print("🔄 创建优化训练器...")
        config = TrainingConfig()
        config.epochs = epochs
        
        trainer = OptimizedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            experiment_name='optimized_msac_t'
        )
        
        print("✅ 训练器创建完成")
        
        # 5. 开始训练
        print("\n🎯 开始优化训练...")
        start_time = time.time()
        
        results = trainer.train()
        
        training_time = time.time() - start_time
        
        # 6. 输出结果
        print("\n" + "=" * 60)
        print("🎉 优化实验完成!")
        print("=" * 60)
        print(f"📊 最终结果:")
        print(f"   最佳验证准确率: {results['best_val_acc']:.4f} ({results['best_val_acc']:.2%})")
        print(f"   最佳测试准确率: {results['best_test_acc']:.4f} ({results['best_test_acc']:.2%})")
        print(f"   训练轮数: {results['total_epochs']}")
        print(f"   训练时间: {training_time:.1f}秒 ({training_time/60:.1f}分钟)")
        
        # 7. 与策略分析结果对比
        expected_acc = 0.3693
        actual_acc = results['best_test_acc'] / 100.0
        improvement = (actual_acc - expected_acc) / expected_acc * 100
        
        print(f"\n📈 性能对比:")
        print(f"   策略分析预期: {expected_acc:.4f} ({expected_acc:.2%})")
        print(f"   实际测试结果: {actual_acc:.4f} ({actual_acc:.2%})")
        print(f"   性能变化: {improvement:+.1f}%")
        
        # 8. 保存详细结果
        detailed_results = {
            'experiment_type': 'optimized_training',
            'strategy': '大批量+低学习率',
            'config': {
                'optimizer': 'adam',
                'learning_rate': 0.0003,
                'batch_size': 128,
                'weight_decay': 0.0001,
                'epochs': epochs,
                'data_subset_ratio': data_subset_ratio
            },
            'results': results,
            'performance_comparison': {
                'expected_accuracy': expected_acc,
                'actual_accuracy': actual_acc,
                'improvement_percentage': improvement
            },
            'training_time_seconds': training_time,
            'device': str(device)
        }
        
        results_file = 'optimized_experiment_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 详细结果已保存到: {results_file}")
        
        return detailed_results
        
    except Exception as e:
        print(f"❌ 实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行基于策略分析的优化实验')
    parser.add_argument('--data-ratio', type=float, default=0.2,
                        help='数据子集比例 (默认: 0.2)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数 (默认: 50)')
    parser.add_argument('--full-data', action='store_true',
                        help='使用完整数据集')
    
    args = parser.parse_args()
    
    # 设置数据比例
    data_ratio = 1.0 if args.full_data else args.data_ratio
    
    print(f"参数设置:")
    print(f"- 数据比例: {data_ratio*100:.1f}%")
    print(f"- 训练轮数: {args.epochs}")
    print()
    
    # 运行实验
    results = run_optimized_experiment(
        data_subset_ratio=data_ratio,
        epochs=args.epochs
    )
    
    if results:
        print("\n✅ 实验成功完成!")
    else:
        print("\n❌ 实验失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()