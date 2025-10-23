#!/usr/bin/env python3
"""
训练策略分析实验
专门测试不同的训练技巧和策略对模型性能的影响
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
import h5py
import json
from datetime import datetime
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class BaselineModel(nn.Module):
    """基准模型 - 用于策略对比"""
    def __init__(self, num_classes=24):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x

def load_data_subset():
    """加载数据子集"""
    print("正在加载数据子集...")
    
    dataset_path = 'dataset/RadioML 2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5'
    
    with h5py.File(dataset_path, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
        Z = f['Z'][:]
    
    # 使用10%的数据进行策略测试
    subset_size = int(0.1 * len(X))
    indices = np.random.choice(len(X), subset_size, replace=False)
    
    X_subset = X[indices]
    Y_subset = Y[indices]
    Z_subset = Z[indices]
    
    # 数据预处理
    X_subset = torch.FloatTensor(X_subset).permute(0, 2, 1)
    Y_subset = torch.LongTensor(np.argmax(Y_subset, axis=1))
    
    return X_subset, Y_subset, Z_subset

def create_data_splits(X, Y, Z):
    """创建数据分割"""
    n_samples = len(X)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    return {
        'train': (X[train_indices], Y[train_indices]),
        'val': (X[val_indices], Y[val_indices]),
        'test': (X[test_indices], Y[test_indices])
    }

class TrainingStrategy:
    """训练策略类"""
    
    def __init__(self, name, config):
        self.name = name
        self.config = config
    
    def create_optimizer(self, model):
        """创建优化器"""
        if self.config['optimizer'] == 'adam':
            return optim.Adam(model.parameters(), lr=self.config['lr'], 
                            weight_decay=self.config.get('weight_decay', 0))
        elif self.config['optimizer'] == 'adamw':
            return optim.AdamW(model.parameters(), lr=self.config['lr'], 
                             weight_decay=self.config.get('weight_decay', 0))
        elif self.config['optimizer'] == 'sgd':
            return optim.SGD(model.parameters(), lr=self.config['lr'], 
                           momentum=self.config.get('momentum', 0.9),
                           weight_decay=self.config.get('weight_decay', 0))
    
    def create_scheduler(self, optimizer, epochs):
        """创建学习率调度器"""
        scheduler_type = self.config.get('scheduler', None)
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
        elif scheduler_type == 'reduce':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        elif scheduler_type == 'warmup':
            return optim.lr_scheduler.LambdaLR(optimizer, 
                                             lambda epoch: min(1.0, epoch / 5))
        return None
    
    def apply_regularization(self, model, loss):
        """应用正则化"""
        if self.config.get('l1_reg', 0) > 0:
            l1_loss = sum(p.abs().sum() for p in model.parameters())
            loss += self.config['l1_reg'] * l1_loss
        
        if self.config.get('l2_reg', 0) > 0:
            l2_loss = sum(p.pow(2).sum() for p in model.parameters())
            loss += self.config['l2_reg'] * l2_loss
        
        return loss
    
    def apply_data_augmentation(self, x):
        """应用数据增强"""
        if self.config.get('noise_aug', False):
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        if self.config.get('scale_aug', False):
            scale = torch.rand(x.size(0), 1, 1) * 0.2 + 0.9
            x = x * scale.to(x.device)
        
        return x

def train_with_strategy(model, data_splits, strategy, device, epochs=15):
    """使用特定策略训练模型"""
    print(f"\n训练策略: {strategy.name}")
    print("-" * 40)
    
    model.to(device)
    
    # 创建数据加载器
    train_loader = DataLoader(
        TensorDataset(*data_splits['train']), 
        batch_size=strategy.config.get('batch_size', 64), 
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(*data_splits['val']), 
        batch_size=64, 
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(*data_splits['test']), 
        batch_size=64, 
        shuffle=False
    )
    
    # 创建优化器和调度器
    optimizer = strategy.create_optimizer(model)
    scheduler = strategy.create_scheduler(optimizer, epochs)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 数据增强
            data = strategy.apply_data_augmentation(data)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 正则化
            loss = strategy.apply_regularization(model, loss)
            
            loss.backward()
            
            # 梯度裁剪
            if strategy.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), strategy.config['grad_clip'])
            
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
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # 更新学习率
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Val_Acc={val_acc:.4f}, LR={current_lr:.6f}")
    
    training_time = time.time() - start_time
    
    # 测试最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
    
    test_acc = test_correct / test_total
    
    return {
        'strategy_name': strategy.name,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'training_time': training_time,
        'history': history,
        'config': strategy.config
    }

def analyze_training_strategies():
    """分析不同训练策略"""
    print("=" * 60)
    print("训练策略分析实验")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    X, Y, Z = load_data_subset()
    data_splits = create_data_splits(X, Y, Z)
    
    # 定义不同的训练策略
    strategies = [
        TrainingStrategy("基准策略", {
            'optimizer': 'adam',
            'lr': 0.001,
            'batch_size': 64
        }),
        
        TrainingStrategy("高学习率+余弦退火", {
            'optimizer': 'adam',
            'lr': 0.01,
            'scheduler': 'cosine',
            'batch_size': 64
        }),
        
        TrainingStrategy("AdamW+权重衰减", {
            'optimizer': 'adamw',
            'lr': 0.003,
            'weight_decay': 0.01,
            'batch_size': 64
        }),
        
        TrainingStrategy("SGD+动量+预热", {
            'optimizer': 'sgd',
            'lr': 0.1,
            'momentum': 0.9,
            'scheduler': 'warmup',
            'batch_size': 64
        }),
        
        TrainingStrategy("数据增强+正则化", {
            'optimizer': 'adam',
            'lr': 0.003,
            'noise_aug': True,
            'scale_aug': True,
            'l2_reg': 0.001,
            'batch_size': 64
        }),
        
        TrainingStrategy("梯度裁剪+自适应LR", {
            'optimizer': 'adam',
            'lr': 0.005,
            'grad_clip': 1.0,
            'scheduler': 'reduce',
            'batch_size': 64
        }),
        
        TrainingStrategy("大批量+低学习率", {
            'optimizer': 'adam',
            'lr': 0.0003,
            'batch_size': 128,
            'weight_decay': 0.0001
        }),
        
        TrainingStrategy("小批量+高学习率", {
            'optimizer': 'adam',
            'lr': 0.01,
            'batch_size': 32,
            'scheduler': 'step'
        })
    ]
    
    results = []
    
    for i, strategy in enumerate(strategies):
        print(f"\n测试策略 {i+1}/{len(strategies)}")
        
        try:
            # 创建新模型
            model = BaselineModel(num_classes=24)
            
            # 训练模型
            result = train_with_strategy(model, data_splits, strategy, device)
            results.append(result)
            
            print(f"结果: Val_Acc={result['best_val_acc']:.4f}, Test_Acc={result['test_acc']:.4f}, Time={result['training_time']:.1f}s")
            
        except Exception as e:
            print(f"策略失败: {e}")
            continue
    
    # 分析结果
    print("\n" + "=" * 60)
    print("策略分析结果")
    print("=" * 60)
    
    if results:
        # 按测试准确率排序
        results.sort(key=lambda x: x['test_acc'], reverse=True)
        
        print("策略性能排名:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['strategy_name']}")
            print(f"   测试准确率: {result['test_acc']:.4f}")
            print(f"   验证准确率: {result['best_val_acc']:.4f}")
            print(f"   训练时间: {result['training_time']:.1f}s")
            print()
        
        # 保存结果
        with open('training_strategy_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("详细结果已保存到 training_strategy_analysis_results.json")
        
        return results
    else:
        print("没有成功的实验结果")
        return []

if __name__ == "__main__":
    results = analyze_training_strategies()
    
    if results:
        best_strategy = results[0]
        print("\n" + "=" * 60)
        print("推荐的最佳训练策略:")
        print("=" * 60)
        print(f"策略名称: {best_strategy['strategy_name']}")
        print(f"测试准确率: {best_strategy['test_acc']:.4f}")
        print(f"配置: {best_strategy['config']}")