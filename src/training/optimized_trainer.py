#!/usr/bin/env python3
"""
optimized_trainer.py - 基于策略分析结果的优化训练器

基于训练策略分析结果，实现最佳训练配置：
- 大批量+低学习率策略（测试准确率: 36.93%）
- Adam优化器，lr=0.0003，batch_size=128，weight_decay=0.0001
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import json
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from ..utils.config import TrainingConfig, ModelConfig
from ..utils.experiment_tracker import ExperimentTracker


class OptimizedTrainer:
    """基于策略分析优化的训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        device: str = 'cuda',
        experiment_name: str = 'optimized_training'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.experiment_name = experiment_name
        
        # 使用优化后的配置
        self.config = config if config is not None else TrainingConfig()
        
        # 确保使用最佳策略配置
        self._apply_best_strategy_config()
        
        # 初始化优化器和调度器
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        
        # 实验跟踪
        self.tracker = ExperimentTracker(experiment_name)
        
        # 训练状态
        self.best_val_acc = 0.0
        self.best_test_acc = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_acc': [],
            'learning_rates': []
        }
        
        # 设置日志
        self._setup_logging()
        
    def _apply_best_strategy_config(self):
        """应用最佳策略配置"""
        # 基于策略分析结果的最佳配置
        self.config.optimizer = 'adam'
        self.config.learning_rate = 3e-4  # 0.0003
        self.config.batch_size = 128
        self.config.weight_decay = 1e-4  # 0.0001
        
        # 优化其他参数以配合最佳策略
        self.config.scheduler = 'cosine'  # 保持余弦退火
        self.config.mixed_precision = True  # 启用混合精度
        self.config.grad_clip = 1.0  # 适度梯度裁剪
        
        self.logger.info(f"应用最佳策略配置: {self._get_strategy_summary()}")
        
    def _get_strategy_summary(self) -> str:
        """获取策略配置摘要"""
        return (f"optimizer={self.config.optimizer}, "
                f"lr={self.config.learning_rate}, "
                f"batch_size={self.config.batch_size}, "
                f"weight_decay={self.config.weight_decay}")
    
    def _setup_optimizer(self):
        """设置优化器"""
        if self.config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")
            
    def _setup_scheduler(self):
        """设置学习率调度器"""
        if self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=self.config.lr_factor
            )
        elif self.config.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
                verbose=True
            )
        else:
            self.scheduler = None
            
    def _setup_loss_function(self):
        """设置损失函数"""
        if self.config.loss_function == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.loss_function == 'label_smoothing':
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
    def _setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger(f'OptimizedTrainer_{self.experiment_name}')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # 使用GradScaler进行反向传播
                if not hasattr(self, 'scaler'):
                    self.scaler = torch.cuda.amp.GradScaler()
                
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # 梯度裁剪
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 记录批次信息
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.6f}, '
                    f'Acc: {100. * correct / total:.2f}%'
                )
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, loader: DataLoader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, Any]:
        """执行完整训练过程"""
        self.logger.info(f"开始优化训练 - {self._get_strategy_summary()}")
        self.logger.info(f"训练集大小: {len(self.train_loader.dataset)}")
        self.logger.info(f"验证集大小: {len(self.val_loader.dataset)}")
        self.logger.info(f"测试集大小: {len(self.test_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate(self.val_loader)
            
            # 测试
            test_loss, test_acc = self.validate(self.test_loader)
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['test_acc'].append(test_acc)
            self.training_history['learning_rates'].append(current_lr)
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_test_acc = test_acc
                self.patience_counter = 0
                
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'config': self.config
                }, f'best_optimized_model_{self.experiment_name}.pth')
                
                self.logger.info(f"保存最佳模型 - Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            else:
                self.patience_counter += 1
            
            epoch_time = time.time() - epoch_start
            
            # 记录epoch信息
            self.logger.info(
                f'Epoch {epoch+1}/{self.config.epochs} '
                f'({epoch_time:.1f}s) - '
                f'Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}% - '
                f'Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}% - '
                f'Test Acc: {test_acc:.2f}% - '
                f'LR: {current_lr:.2e}'
            )
            
            # 早停检查
            if self.config.early_stopping and self.patience_counter >= self.config.patience:
                self.logger.info(f"早停触发 - 最佳验证准确率: {self.best_val_acc:.4f}")
                break
        
        total_time = time.time() - start_time
        
        # 训练结果
        results = {
            'best_val_acc': self.best_val_acc,
            'best_test_acc': self.best_test_acc,
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'strategy_config': {
                'optimizer': self.config.optimizer,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'weight_decay': self.config.weight_decay
            },
            'training_history': self.training_history
        }
        
        # 保存结果
        results_path = f'optimized_training_results_{self.experiment_name}.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"训练完成!")
        self.logger.info(f"最佳验证准确率: {self.best_val_acc:.4f}")
        self.logger.info(f"最佳测试准确率: {self.best_test_acc:.4f}")
        self.logger.info(f"总训练时间: {total_time:.1f}秒")
        self.logger.info(f"结果已保存到: {results_path}")
        
        return results


def create_optimized_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str = 'cuda',
    experiment_name: str = 'optimized_training'
) -> OptimizedTrainer:
    """创建优化训练器的便捷函数"""
    config = TrainingConfig()
    
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        experiment_name=experiment_name
    )
    
    return trainer


if __name__ == "__main__":
    # 示例用法
    print("OptimizedTrainer - 基于策略分析结果的优化训练器")
    print("最佳策略: 大批量+低学习率")
    print("配置: optimizer=adam, lr=0.0003, batch_size=128, weight_decay=0.0001")
    print("预期性能: 测试准确率 ~36.93%")