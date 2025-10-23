"""
trainer.py - 完整的训练器
包含：
1. 训练循环
2. 验证
3. 早停
4. 学习率调度
5. 混合精度训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np


# ==================== 损失函数 ====================

class FocalLoss(nn.Module):
    """Focal Loss"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy"""
    
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / num_classes
        
        loss = -(targets_smooth * log_probs).sum(dim=-1)
        return loss.mean()


def get_loss_function(config):
    """根据配置获取损失函数"""
    if config.training.loss_function == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif config.training.loss_function == 'focal':
        return FocalLoss(
            alpha=config.training.focal_alpha,
            gamma=config.training.focal_gamma
        )
    elif config.training.loss_function == 'label_smoothing':
        return LabelSmoothingCrossEntropy(epsilon=config.training.label_smoothing)
    else:
        raise ValueError(f"Unknown loss function: {config.training.loss_function}")


# ==================== 早停 ====================

class EarlyStopping:
    """早停"""
    
    def __init__(self, patience=10, min_delta=0, mode='max'):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善量
            mode: 'max' 或 'min'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


# ==================== 训练器 ====================

class Trainer:
    """完整的训练器"""
    
    def __init__(self, model, config, device='cuda', experiment_tracker=None):
        """
        Args:
            model: 模型
            config: 配置对象
            device: 设备
            experiment_tracker: 实验跟踪器
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.tracker = experiment_tracker
        
        # 优化器
        self.optimizer = self._get_optimizer()
        
        # 学习率调度器
        self.scheduler = self._get_scheduler()
        
        # 损失函数
        self.criterion = get_loss_function(config)
        
        # 混合精度训练
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            mode='max'
        ) if config.training.early_stopping else None
        
        # 历史记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.current_epoch = 0
    
    def _get_optimizer(self):
        """获取优化器"""
        if self.config.training.optimizer == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _get_scheduler(self):
        """获取学习率调度器"""
        if self.config.training.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=1e-6
            )
        elif self.config.training.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=self.config.training.lr_factor
            )
        elif self.config.training.scheduler == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.training.lr_factor,
                patience=self.config.training.lr_patience,
                verbose=True
            )
        else:
            return None
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1}')
        
        for batch_idx, (signals, labels, snrs) in enumerate(pbar):
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            snrs = snrs.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            if self.scaler:
                with autocast():
                    outputs = self.model(signals, snrs)
                    loss = self.criterion(outputs['logits'], labels)
                
                self.scaler.scale(loss).backward()
                
                if self.config.training.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(signals, snrs)
                loss = self.criterion(outputs['logits'], labels)
                
                loss.backward()
                
                if self.config.training.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip
                    )
                
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, labels, snrs in val_loader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                snrs = snrs.to(self.device)
                
                outputs = self.model(signals, snrs)
                loss = self.criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader=None):
        """完整的训练流程"""
        print(f"开始训练，共 {self.config.training.epochs} 个epoch")
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
            else:
                val_loss, val_acc = 0, 0
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # 打印结果
            print(f'Epoch {epoch+1}/{self.config.training.epochs}:')
            print(f'  Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}')
            if val_loader:
                print(f'  Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth')
            
            # 实验跟踪
            if self.tracker:
                self.tracker.log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': self.optimizer.param_groups[0]['lr']
                }, epoch)
            
            # 早停
            if self.early_stopping:
                if self.early_stopping(val_acc):
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        print(f'训练完成！最佳验证准确率: {self.best_val_acc:.4f}')
        return self.history
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']