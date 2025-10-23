"""
improved_trainer.py - 改进的训练器
包含先进的训练策略和技术：
1. 多种损失函数组合
2. 高级学习率调度
3. 数据增强策略
4. 混合精度训练
5. 梯度累积
6. 早停和模型保存
7. 训练监控和可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import time
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


# ==================== 高级损失函数 ====================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
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
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / num_classes
        
        loss = -(targets_smooth * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, 
                 classification_weight=1.0,
                 snr_weight=0.1,
                 focal_alpha=1.0,
                 focal_gamma=2.0,
                 label_smoothing=0.1):
        super().__init__()
        self.classification_weight = classification_weight
        self.snr_weight = snr_weight
        
        # 分类损失
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma)
        self.smooth_ce = LabelSmoothingCrossEntropy(label_smoothing)
        
        # SNR回归损失
        self.snr_loss = nn.MSELoss()
    
    def forward(self, outputs, targets, snr_targets=None):
        logits = outputs['logits']
        snr_pred = outputs.get('snr_pred', None)
        
        # 分类损失 (Focal + Label Smoothing的组合)
        focal_loss = self.focal_loss(logits, targets)
        smooth_loss = self.smooth_ce(logits, targets)
        classification_loss = 0.7 * focal_loss + 0.3 * smooth_loss
        
        total_loss = self.classification_weight * classification_loss
        
        # SNR回归损失
        if snr_pred is not None and snr_targets is not None:
            snr_loss = self.snr_loss(snr_pred, snr_targets)
            total_loss += self.snr_weight * snr_loss
        
        return total_loss, {
            'classification_loss': classification_loss.item(),
            'snr_loss': snr_loss.item() if snr_pred is not None else 0.0,
            'total_loss': total_loss.item()
        }


# ==================== 数据增强策略 ====================

class AdvancedSignalAugmentation:
    """高级信号增强策略"""
    
    @staticmethod
    def awgn_noise(signal, snr_db_range=(-5, 15)):
        """加性高斯白噪声"""
        snr_db = np.random.uniform(*snr_db_range)
        signal_power = torch.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        return signal + noise
    
    @staticmethod
    def frequency_shift(signal, max_shift=0.1):
        """频率偏移"""
        shift = np.random.uniform(-max_shift, max_shift)
        length = signal.size(-1)
        t = torch.arange(length, dtype=torch.float32, device=signal.device)
        
        # 对I/Q分别应用频率偏移
        real_shifted = signal[0] * torch.cos(2 * np.pi * shift * t) - signal[1] * torch.sin(2 * np.pi * shift * t)
        imag_shifted = signal[0] * torch.sin(2 * np.pi * shift * t) + signal[1] * torch.cos(2 * np.pi * shift * t)
        
        return torch.stack([real_shifted, imag_shifted])
    
    @staticmethod
    def time_shift(signal, max_shift=0.1):
        """时间偏移"""
        length = signal.size(-1)
        max_shift_samples = int(max_shift * length)
        if max_shift_samples == 0:
            return signal
            
        shift_samples = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        
        if shift_samples > 0:
            # 右移
            zeros = torch.zeros(signal.size(0), shift_samples, device=signal.device, dtype=signal.dtype)
            shifted = torch.cat([zeros, signal[:, :-shift_samples]], dim=1)
        elif shift_samples < 0:
            # 左移
            zeros = torch.zeros(signal.size(0), -shift_samples, device=signal.device, dtype=signal.dtype)
            shifted = torch.cat([signal[:, -shift_samples:], zeros], dim=1)
        else:
            shifted = signal
        
        return shifted
    
    @staticmethod
    def amplitude_scaling(signal, scale_range=(0.8, 1.2)):
        """幅度缩放"""
        scale = np.random.uniform(*scale_range)
        return signal * scale
    
    @staticmethod
    def phase_rotation(signal, max_rotation=np.pi/6):
        """相位旋转"""
        rotation = np.random.uniform(-max_rotation, max_rotation)
        
        real = signal[0] * np.cos(rotation) - signal[1] * np.sin(rotation)
        imag = signal[0] * np.sin(rotation) + signal[1] * np.cos(rotation)
        
        return torch.stack([real, imag])
    
    @classmethod
    def apply_augmentation(cls, signal, augment_prob=0.8):
        """应用随机增强"""
        if np.random.random() > augment_prob:
            return signal
        
        # 随机选择增强方法
        augmentations = [
            cls.awgn_noise,
            cls.frequency_shift,
            cls.time_shift,
            cls.amplitude_scaling,
            cls.phase_rotation
        ]
        
        # 随机应用1-3种增强
        num_augs = np.random.randint(1, 4)
        selected_augs = np.random.choice(augmentations, num_augs, replace=False)
        
        augmented_signal = signal
        for aug_func in selected_augs:
            augmented_signal = aug_func(augmented_signal)
        
        return augmented_signal


# ==================== 改进的训练器 ====================

class ImprovedTrainer:
    """改进的训练器"""
    
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader,
                 config,
                 device='cuda',
                 experiment_dir='experiments'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        
        # 创建实验目录
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 设置优化器
        self.optimizer = self._create_optimizer()
        
        # 设置学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 设置损失函数
        self.criterion = CombinedLoss(
            classification_weight=config.get('classification_weight', 1.0),
            snr_weight=config.get('snr_weight', 0.1),
            focal_alpha=config.get('focal_alpha', 1.0),
            focal_gamma=config.get('focal_gamma', 2.0),
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 梯度累积
        self.accumulation_steps = config.get('accumulation_steps', 1)
        
        # 早停
        self.patience = config.get('patience', 20)
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # 训练历史
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        
        # 数据增强
        self.use_augmentation = config.get('use_augmentation', True)
        self.augment_prob = config.get('augment_prob', 0.8)
    
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('T_0', 10),
                T_mult=self.config.get('T_mult', 2),
                eta_min=self.config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('max_lr', 1e-3),
                epochs=self.config.get('epochs', 200),
                steps_per_epoch=len(self.train_loader)
            )
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        loss_components = defaultdict(float)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            if len(batch) == 3:
                signals, labels, snr = batch
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                snr = snr.to(self.device)
            else:
                signals, labels = batch
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                snr = None
            
            # 数据增强
            if self.use_augmentation and self.model.training:
                batch_size = signals.size(0)
                for i in range(batch_size):
                    if np.random.random() < self.augment_prob:
                        signals[i] = AdvancedSignalAugmentation.apply_augmentation(
                            signals[i], self.augment_prob
                        )
            
            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(signals, snr)
                    loss, loss_dict = self.criterion(outputs, labels, snr)
                    loss = loss / self.accumulation_steps
            else:
                outputs = self.model(signals, snr)
                loss, loss_dict = self.criterion(outputs, labels, snr)
                loss = loss / self.accumulation_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # 学习率调度
                if isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
            
            # 统计
            total_loss += loss.item() * self.accumulation_steps
            for key, value in loss_dict.items():
                loss_components[key] += value
            
            predictions = torch.argmax(outputs['logits'], dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # 更新进度条
            current_acc = total_correct / total_samples
            progress_bar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples
        
        # 记录历史
        self.train_history['loss'].append(avg_loss)
        self.train_history['accuracy'].append(avg_acc)
        for key, value in loss_components.items():
            self.train_history[key].append(value / len(self.train_loader))
        
        return avg_loss, avg_acc
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                if len(batch) == 3:
                    signals, labels, snr = batch
                    signals = signals.to(self.device)
                    labels = labels.to(self.device)
                    snr = snr.to(self.device)
                else:
                    signals, labels = batch
                    signals = signals.to(self.device)
                    labels = labels.to(self.device)
                    snr = None
                
                outputs = self.model(signals, snr)
                loss, _ = self.criterion(outputs, labels, snr)
                
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs['logits'], dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_samples
        
        # 记录历史
        self.val_history['loss'].append(avg_loss)
        self.val_history['accuracy'].append(avg_acc)
        
        # 学习率调度
        if isinstance(self.scheduler, (CosineAnnealingWarmRestarts, ReduceLROnPlateau)):
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(avg_acc)
            else:
                self.scheduler.step()
        
        return avg_loss, avg_acc, all_predictions, all_labels
    
    def train(self, epochs):
        """完整训练流程"""
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(epoch)
            
            # 打印结果
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"  New best validation accuracy: {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # 保存训练历史
        self.save_training_history()
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': val_acc,
            'config': self.config,
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history)
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.experiment_dir, 'best_model.pth'))
        
        torch.save(checkpoint, os.path.join(self.experiment_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train': dict(self.train_history),
            'val': dict(self.val_history)
        }
        
        with open(os.path.join(self.experiment_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_history['loss'], label='Train Loss')
        axes[0, 0].plot(self.val_history['loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.train_history['accuracy'], label='Train Acc')
        axes[0, 1].plot(self.val_history['accuracy'], label='Val Acc')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        if 'lr' in self.train_history:
            axes[1, 0].plot(self.train_history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].grid(True)
        
        # 损失组件
        if 'classification_loss' in self.train_history:
            axes[1, 1].plot(self.train_history['classification_loss'], label='Classification')
            if 'snr_loss' in self.train_history:
                axes[1, 1].plot(self.train_history['snr_loss'], label='SNR')
            axes[1, 1].set_title('Loss Components')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()


# ==================== 使用示例 ====================

def create_improved_trainer(model, train_loader, val_loader, config, device='cuda'):
    """创建改进的训练器"""
    return ImprovedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )


if __name__ == "__main__":
    # 测试训练器
    from src.models.improved_msac_t import create_improved_msac_t
    
    # 创建模型
    model = create_improved_msac_t(num_classes=11)
    
    # 配置
    config = {
        'optimizer': 'adamw',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'use_amp': True,
        'accumulation_steps': 1,
        'patience': 20,
        'use_augmentation': True,
        'augment_prob': 0.8,
        'classification_weight': 1.0,
        'snr_weight': 0.1,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'label_smoothing': 0.1
    }
    
    print("Improved trainer created successfully!")