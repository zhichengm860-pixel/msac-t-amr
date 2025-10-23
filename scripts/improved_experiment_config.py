#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的实验配置文件
解决当前模型性能问题：准确率8%和F1分数0.0119

主要改进：
1. 增加训练轮数 (3 → 50)
2. 优化学习率 (0.001 → 0.0001)
3. 添加学习率调度
4. 增加正则化技术
5. 添加早停机制
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
from pathlib import Path
import time
import json

class ImprovedExperimentConfig:
    """改进的实验配置类"""
    
    def __init__(self):
        # 基础配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 11
        self.input_channels = 2
        
        # 优化的训练参数
        self.training_config = {
            'epochs': 50,                    # 增加训练轮数
            'learning_rate': 0.0001,         # 降低学习率
            'batch_size': 64,                # 优化批次大小
            'weight_decay': 1e-4,            # 添加权重衰减
            'dropout_rate': 0.3,             # Dropout正则化
            'early_stopping_patience': 10,   # 早停耐心值
            'min_delta': 0.001,              # 早停最小改进
        }
        
        # 学习率调度配置
        self.scheduler_config = {
            'type': 'StepLR',               # 'StepLR' 或 'CosineAnnealingLR'
            'step_size': 15,                # StepLR步长
            'gamma': 0.5,                   # StepLR衰减因子
            'T_max': 50,                    # CosineAnnealingLR周期
        }
        
        # 数据增强配置
        self.augmentation_config = {
            'noise_std': 0.01,              # 噪声标准差
            'amplitude_scale_range': (0.8, 1.2),  # 幅度缩放范围
            'phase_shift_range': (-0.1, 0.1),     # 相位偏移范围
            'time_shift_range': (-5, 5),          # 时间偏移范围
        }
        
        # 验证配置
        self.validation_config = {
            'val_split': 0.2,               # 验证集比例
            'test_split': 0.1,              # 测试集比例
            'shuffle': True,                # 是否打乱数据
            'stratify': True,               # 是否分层采样
        }

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """保存最佳模型权重"""
        self.best_weights = model.state_dict().copy()

class ImprovedTrainer:
    """改进的训练器"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 设置优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.training_config['learning_rate'],
            weight_decay=config.training_config['weight_decay']
        )
        
        # 设置学习率调度器
        if config.scheduler_config['type'] == 'StepLR':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=config.scheduler_config['step_size'],
                gamma=config.scheduler_config['gamma']
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.scheduler_config['T_max']
            )
        
        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 设置早停
        self.early_stopping = EarlyStopping(
            patience=config.training_config['early_stopping_patience'],
            min_delta=config.training_config['min_delta']
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.config.device), target.to(self.config.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """完整训练流程"""
        print("开始改进的训练流程...")
        print(f"设备: {self.config.device}")
        print(f"训练轮数: {self.config.training_config['epochs']}")
        print(f"学习率: {self.config.training_config['learning_rate']}")
        print(f"批次大小: {self.config.training_config['batch_size']}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training_config['epochs']):
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{self.config.training_config['epochs']}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}")
            
            # 早停检查
            if self.early_stopping(val_loss, self.model):
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        training_time = time.time() - start_time
        print(f"训练完成，总时间: {training_time:.2f}秒")
        
        return self.history, training_time

def create_improved_experiment_runner():
    """创建改进的实验运行器"""
    config = ImprovedExperimentConfig()
    
    # 返回配置和训练器类
    return config, ImprovedTrainer

if __name__ == "__main__":
    # 示例使用
    config = ImprovedExperimentConfig()
    print("改进的实验配置已创建")
    print(f"训练配置: {config.training_config}")
    print(f"调度器配置: {config.scheduler_config}")