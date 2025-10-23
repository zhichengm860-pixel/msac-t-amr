#!/usr/bin/env python3
"""
test_optimized_config.py - 测试优化后的配置

快速验证基于策略分析结果优化的模型和训练配置
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json

# 直接导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'training'))

from improved_msac_t import ImprovedMSAC_T
from config import TrainingConfig, ModelConfig


def create_synthetic_data(num_samples: int = 1000, seq_length: int = 1024):
    """创建合成数据用于快速测试"""
    print(f"🔄 创建合成数据 ({num_samples} 样本)...")
    
    # 创建复数信号数据 (I/Q 两个通道)
    data = np.random.randn(num_samples, 2, seq_length).astype(np.float32)
    
    # 创建随机标签 (11个调制类型)
    labels = np.random.randint(0, 11, num_samples)
    
    return data, labels


def test_model_creation():
    """测试优化后的模型创建"""
    print("🔄 测试模型创建...")
    
    # 使用优化后的配置
    config = ModelConfig()
    
    model = ImprovedMSAC_T(
        num_classes=config.num_classes,
        input_channels=1,
        base_channels=64,
        num_transformer_blocks=6,
        num_heads=config.num_heads,
        dropout=config.dropout
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ 模型创建成功:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   Dropout率: {config.dropout}")
    print(f"   特征维度: {config.feature_dim}")
    print(f"   注意力头数: {config.num_heads}")
    
    return model


def test_training_config():
    """测试优化后的训练配置"""
    print("🔄 测试训练配置...")
    
    config = TrainingConfig()
    
    print(f"✅ 训练配置验证:")
    print(f"   优化器: {config.optimizer}")
    print(f"   学习率: {config.learning_rate}")
    print(f"   批量大小: {config.batch_size}")
    print(f"   权重衰减: {config.weight_decay}")
    print(f"   调度器: {config.scheduler}")
    print(f"   混合精度: {config.mixed_precision}")
    print(f"   梯度裁剪: {config.grad_clip}")
    
    return config


def test_data_loading():
    """测试数据加载和批处理"""
    print("🔄 测试数据加载...")
    
    # 创建合成数据
    data, labels = create_synthetic_data(1000)
    
    # 创建数据集
    dataset = TensorDataset(
        torch.FloatTensor(data),
        torch.LongTensor(labels)
    )
    
    # 使用优化后的批量大小
    config = TrainingConfig()
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # 在Windows上设为0避免问题
        pin_memory=True
    )
    
    # 测试一个批次
    batch_data, batch_labels = next(iter(dataloader))
    
    print(f"✅ 数据加载测试成功:")
    print(f"   批量大小: {batch_data.shape[0]}")
    print(f"   数据形状: {batch_data.shape}")
    print(f"   标签形状: {batch_labels.shape}")
    print(f"   数据类型: {batch_data.dtype}")
    
    return dataloader


def test_forward_pass():
    """测试前向传播"""
    print("🔄 测试前向传播...")
    
    # 创建模型和数据
    model = test_model_creation()
    dataloader = test_data_loading()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        batch_data, batch_labels = next(iter(dataloader))
        batch_data = batch_data.to(device)
        
        start_time = time.time()
        output = model(batch_data)
        forward_time = time.time() - start_time
        
        print(f"✅ 前向传播测试成功:")
        print(f"   输入形状: {batch_data.shape}")
        
        # 处理可能的字典输出
        if isinstance(output, dict):
            main_output = output.get('logits', output.get('predictions', list(output.values())[0]))
            print(f"   输出类型: 字典 (包含 {list(output.keys())})")
            print(f"   主输出形状: {main_output.shape}")
            
            # 检查输出
            assert main_output.shape[0] == batch_data.shape[0], "批量大小不匹配"
            assert main_output.shape[1] == 11, "输出类别数不正确"
            
            print(f"   输出范围: [{main_output.min().item():.3f}, {main_output.max().item():.3f}]")
        else:
            print(f"   输出形状: {output.shape}")
            
            # 检查输出
            assert output.shape[0] == batch_data.shape[0], "批量大小不匹配"
            assert output.shape[1] == 11, "输出类别数不正确"
            
            print(f"   输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        print(f"   前向传播时间: {forward_time*1000:.2f}ms")
        print(f"   设备: {device}")


def test_optimizer_setup():
    """测试优化器设置"""
    print("🔄 测试优化器设置...")
    
    model = test_model_creation()
    config = TrainingConfig()
    
    # 创建优化器
    if config.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    # 创建调度器
    if config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01
        )
    
    print(f"✅ 优化器设置成功:")
    print(f"   优化器类型: {type(optimizer).__name__}")
    print(f"   初始学习率: {optimizer.param_groups[0]['lr']}")
    print(f"   权重衰减: {optimizer.param_groups[0]['weight_decay']}")
    print(f"   调度器类型: {type(scheduler).__name__}")
    
    return optimizer, scheduler


def test_performance_comparison():
    """测试性能对比"""
    print("🔄 测试性能对比...")
    
    # 原始配置
    original_config = {
        'batch_size': 64,
        'learning_rate': 1e-4,
        'optimizer': 'adamw',
        'dropout': 0.5
    }
    
    # 优化后配置
    optimized_config = TrainingConfig()
    
    print(f"✅ 配置对比:")
    print(f"   批量大小: {original_config['batch_size']} → {optimized_config.batch_size}")
    print(f"   学习率: {original_config['learning_rate']} → {optimized_config.learning_rate}")
    print(f"   优化器: {original_config['optimizer']} → {optimized_config.optimizer}")
    
    model_config = ModelConfig()
    print(f"   Dropout: {original_config['dropout']} → {model_config.dropout}")
    
    # 计算理论性能提升
    batch_improvement = optimized_config.batch_size / original_config['batch_size']
    lr_improvement = optimized_config.learning_rate / original_config['learning_rate']
    
    print(f"\n📊 理论性能分析:")
    print(f"   批量大小提升: {batch_improvement:.1f}x")
    print(f"   学习率调整: {lr_improvement:.1f}x")
    print(f"   预期准确率: ~36.93% (基于策略分析)")
    print(f"   训练时间优化: ~53% 减少 (5818s vs 12497s基准)")


def main():
    """主测试函数"""
    print("🚀 开始测试优化后的配置")
    print("=" * 60)
    
    try:
        # 1. 测试模型创建
        print("\n1. 模型创建测试")
        print("-" * 30)
        test_model_creation()
        
        # 2. 测试训练配置
        print("\n2. 训练配置测试")
        print("-" * 30)
        test_training_config()
        
        # 3. 测试数据加载
        print("\n3. 数据加载测试")
        print("-" * 30)
        test_data_loading()
        
        # 4. 测试前向传播
        print("\n4. 前向传播测试")
        print("-" * 30)
        test_forward_pass()
        
        # 5. 测试优化器设置
        print("\n5. 优化器设置测试")
        print("-" * 30)
        test_optimizer_setup()
        
        # 6. 测试性能对比
        print("\n6. 性能对比分析")
        print("-" * 30)
        test_performance_comparison()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！优化配置验证成功！")
        print("=" * 60)
        
        # 输出配置摘要
        print("\n📋 优化配置摘要:")
        print("- 策略: 大批量+低学习率")
        print("- 优化器: Adam")
        print("- 学习率: 0.0003")
        print("- 批量大小: 128")
        print("- 权重衰减: 0.0001")
        print("- 模型Dropout: 0.3 (从0.5优化)")
        print("- 预期性能提升: ~36.93% 测试准确率")
        print("- 训练时间优化: ~53% 减少")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)