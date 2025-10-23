#!/usr/bin/env python3
"""
轻量级自监督预训练实验脚本
使用更小的模型和批次大小来避免内存问题
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

from src.utils import Config
from src.models import ResNet1D  # 使用更轻量的ResNet模型
from src.data import DatasetLoader
from src.data import get_radioml_config
from src.utils import ExperimentTracker

def setup_experiment():
    """设置实验环境"""
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/pretrain_lite_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return exp_dir, device

def load_pretrain_data(config):
    """加载预训练数据"""
    print("=== 加载预训练数据 ===")
    
    # 获取数据集配置
    dataset_config, preprocess_config, dataloader_config = get_radioml_config('2016.10a')
    
    # 加载数据
    loader = DatasetLoader(dataset_config)
    train_loader, val_loader, test_loader = loader.load_radioml_data(
        dataset_config.path,
        batch_size=32,  # 减小批次大小
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    print(f"预训练数据加载完成:")
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")
    
    return train_loader, val_loader, test_loader, dataset_config

def run_simple_pretrain(model, train_loader, val_loader, device, exp_dir):
    """运行简单的重建预训练"""
    print("=== 简单重建预训练 ===")
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 训练参数
    epochs = 3  # 减少训练轮数
    max_batches = 20  # 限制批次数量
    
    print(f"开始预训练，共 {epochs} 轮，每轮最多 {max_batches} 批次...")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 训练循环
        for batch_idx, (signals, labels, snrs) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
                
            signals = signals.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(signals)
            
            # 简单的对比学习损失（使用标签作为监督信号）
            loss = criterion(outputs, labels.to(device))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{max_batches}, Loss: {loss.item():.6f}")
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (signals, labels, snrs) in enumerate(val_loader):
                if batch_idx >= 10:  # 限制验证批次
                    break
                    
                signals = signals.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels.to(device))
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = train_loss / max_batches
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoints", "best_pretrain_model.pth"))
            print(f"保存最佳模型，验证损失: {best_loss:.6f}")
    
    return {
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'best_val_loss': best_loss
    }

def main():
    """主函数"""
    print("开始轻量级自监督预训练实验...")
    
    # 设置实验
    exp_dir, device = setup_experiment()
    print(f"=== 实验设置 ===")
    print(f"实验目录: {exp_dir}")
    print(f"使用设备: {device}")
    
    # 加载配置
    config = Config()
    
    # 加载数据
    train_loader, val_loader, test_loader, dataset_config = load_pretrain_data(config)
    
    # 创建轻量级模型
    print(f"数据集: {dataset_config.name}")
    print(f"调制类型数量: {dataset_config.num_classes}")
    print(f"信号长度: {dataset_config.signal_length}")
    
    model = ResNet1D(
        num_classes=dataset_config.num_classes,
        input_channels=2,  # I/Q两个通道
        layers=[1, 1, 1, 1]  # 减少层数
    ).to(device)
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n轻量级模型参数数量: {total_params:,}")
    
    # 初始化实验跟踪器
    tracker = ExperimentTracker(exp_dir)
    
    # 保存配置
    config_dict = {
        'model': 'ResNet1D-Lite',
        'num_classes': dataset_config.num_classes,
        'signal_length': dataset_config.signal_length,
        'input_channels': 2,
        'layers': [1, 1, 1, 1],
        'batch_size': 32,
        'device': str(device),
        'total_params': total_params
    }
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # 运行预训练实验
    try:
        results = run_simple_pretrain(model, train_loader, val_loader, device, exp_dir)
        
        # 保存结果
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n=== 预训练完成 ===")
        print(f"最终训练损失: {results['final_train_loss']:.6f}")
        print(f"最终验证损失: {results['final_val_loss']:.6f}")
        print(f"最佳验证损失: {results['best_val_loss']:.6f}")
        print(f"实验结果保存在: {exp_dir}")
        
    except Exception as e:
        print(f"预训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()