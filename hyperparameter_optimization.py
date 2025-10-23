#!/usr/bin/env python3
"""
超参数优化实验 - 快速探索最佳训练配置
目标：在短时间内找到最优的训练策略和参数设置
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import h5py
import json
import itertools
from datetime import datetime

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class OptimizedComplexConv1d(nn.Module):
    """优化的复数卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=0.1):
        super().__init__()
        
        # 分离实部和虚部处理
        half_in = in_channels // 2 if in_channels > 1 else 1
        
        self.real_conv = nn.Conv1d(half_in, out_channels, kernel_size, stride, padding)
        self.imag_conv = nn.Conv1d(half_in, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 分离实部和虚部
        channels = x.size(1)
        half_channels = channels // 2
        
        real_part = x[:, :half_channels, :]
        imag_part = x[:, half_channels:, :]
        
        # 复数卷积
        real_out = self.real_conv(real_part) - self.imag_conv(imag_part)
        imag_out = self.real_conv(imag_part) + self.imag_conv(real_part)
        
        # 合并输出
        out = torch.cat([real_out, imag_out], dim=1)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        return out

class FlexibleMSAC(nn.Module):
    """灵活的多尺度注意力复数网络 - 支持参数调整"""
    def __init__(self, num_classes=24, base_channels=16, depth=3, dropout=0.1):
        super().__init__()
        
        # 输入层
        self.input_conv = OptimizedComplexConv1d(2, base_channels, kernel_size=7, padding=3, dropout=dropout)
        
        # 多尺度特征提取层
        self.scales = nn.ModuleList()
        current_channels = base_channels * 2  # 复数输出
        
        for i in range(depth):
            out_channels = base_channels * (2 ** (i + 1))
            
            scale_block = nn.Sequential(
                OptimizedComplexConv1d(current_channels, out_channels, kernel_size=3, padding=1, dropout=dropout),
                OptimizedComplexConv1d(out_channels * 2, out_channels, kernel_size=5, padding=2, dropout=dropout)
            )
            self.scales.append(scale_block)
            current_channels = out_channels * 2
        
        # 全局池化和分类器
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(current_channels, current_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(current_channels // 2, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入处理
        x = self.input_conv(x)
        
        # 多尺度特征提取
        for scale in self.scales:
            x = scale(x)
        
        # 全局池化和分类
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x

def load_radioml2018_data_fast():
    """快速加载数据子集用于参数优化"""
    print("正在加载RadioML 2018.01A数据集（优化版本）...")
    
    dataset_path = 'dataset/RadioML 2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5'
    
    with h5py.File(dataset_path, 'r') as f:
        X = f['X'][:]  # 信号数据
        Y = f['Y'][:]  # 调制类型标签
        Z = f['Z'][:]  # SNR标签
    
    print(f"原始数据形状: X={X.shape}, Y={Y.shape}, Z={Z.shape}")
    
    # 使用更小的子集进行快速优化（5%）
    subset_size = int(0.05 * len(X))
    indices = np.random.choice(len(X), subset_size, replace=False)
    
    X_subset = X[indices]
    Y_subset = Y[indices]
    Z_subset = Z[indices]
    
    print(f"优化子集形状: X={X_subset.shape}, Y={Y_subset.shape}, Z={Z_subset.shape}")
    
    return X_subset, Y_subset, Z_subset

def create_dataloaders_fast(X, Y, Z, batch_size=64):
    """创建快速数据加载器"""
    # 数据预处理
    X = torch.FloatTensor(X).permute(0, 2, 1)  # (batch, 2, 1024)
    Y = torch.LongTensor(np.argmax(Y, axis=1))
    
    # 简单的训练/验证分割
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # 创建数据集
    train_dataset = TensorDataset(X[train_indices], Y[train_indices])
    val_dataset = TensorDataset(X[val_indices], Y[val_indices])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_model_fast(model, train_loader, val_loader, device, config):
    """快速训练模型"""
    model.to(device)
    
    # 优化器设置
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    
    # 学习率调度器
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    elif config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['epochs']//3, gamma=0.1)
    else:
        scheduler = None
    
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_acc = val_correct / val_total
        avg_loss = epoch_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if scheduler:
            scheduler.step()
        
        if epoch % 2 == 0:  # 每2个epoch打印一次
            print(f"Epoch {epoch+1}/{config['epochs']}: Loss={avg_loss:.4f}, Val_Acc={val_acc:.4f}")
    
    return best_val_acc, train_losses, val_accuracies

def hyperparameter_search():
    """超参数搜索"""
    print("=" * 60)
    print("超参数优化实验")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    X, Y, Z = load_radioml2018_data_fast()
    
    # 定义搜索空间
    search_space = {
        'base_channels': [16, 32, 48],
        'depth': [2, 3, 4],
        'dropout': [0.1, 0.2, 0.3],
        'lr': [0.001, 0.003, 0.01],
        'batch_size': [32, 64, 128],
        'optimizer': ['adam', 'adamw'],
        'scheduler': ['cosine', 'step', None],
        'weight_decay': [1e-4, 1e-3, 1e-2],
        'epochs': [10]  # 快速测试
    }
    
    # 随机搜索（选择20个配置进行测试）
    n_trials = 20
    results = []
    
    for trial in range(n_trials):
        print(f"\n试验 {trial+1}/{n_trials}")
        print("-" * 40)
        
        # 随机选择配置
        config = {}
        for key, values in search_space.items():
            selected_value = np.random.choice(values)
            # 转换为Python原生类型
            if isinstance(selected_value, np.integer):
                config[key] = int(selected_value)
            elif isinstance(selected_value, np.floating):
                config[key] = float(selected_value)
            else:
                config[key] = selected_value
        
        print(f"配置: {config}")
        
        try:
            # 创建数据加载器
            train_loader, val_loader = create_dataloaders_fast(X, Y, Z, config['batch_size'])
            
            # 创建模型
            model = FlexibleMSAC(
                num_classes=24,
                base_channels=config['base_channels'],
                depth=config['depth'],
                dropout=config['dropout']
            )
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            
            # 训练模型
            start_time = time.time()
            best_val_acc, train_losses, val_accuracies = train_model_fast(
                model, train_loader, val_loader, device, config
            )
            training_time = time.time() - start_time
            
            # 记录结果
            result = {
                'trial': trial + 1,
                'config': config,
                'best_val_acc': best_val_acc,
                'total_params': total_params,
                'training_time': training_time,
                'final_loss': train_losses[-1] if train_losses else float('inf')
            }
            results.append(result)
            
            print(f"结果: 准确率={best_val_acc:.4f}, 参数量={total_params:,}, 时间={training_time:.1f}s")
            
        except Exception as e:
            print(f"试验失败: {e}")
            continue
    
    # 分析结果
    print("\n" + "=" * 60)
    print("优化结果分析")
    print("=" * 60)
    
    if results:
        # 按准确率排序
        results.sort(key=lambda x: x['best_val_acc'], reverse=True)
        
        print("前5个最佳配置:")
        for i, result in enumerate(results[:5]):
            print(f"\n第{i+1}名:")
            print(f"  准确率: {result['best_val_acc']:.4f}")
            print(f"  参数量: {result['total_params']:,}")
            print(f"  训练时间: {result['training_time']:.1f}s")
            print(f"  配置: {result['config']}")
        
        # 保存结果
        with open('hyperparameter_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n完整结果已保存到 hyperparameter_optimization_results.json")
        
        # 返回最佳配置
        return results[0]['config']
    else:
        print("没有成功的试验结果")
        return None

if __name__ == "__main__":
    best_config = hyperparameter_search()
    
    if best_config:
        print("\n" + "=" * 60)
        print("推荐的最佳配置:")
        print("=" * 60)
        for key, value in best_config.items():
            print(f"{key}: {value}")