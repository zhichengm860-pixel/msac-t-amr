#!/usr/bin/env python3
"""
改进的RadioML 2018实验 - 针对性能瓶颈优化
基于前三个实验的分析结果，重点解决：
1. 模型复杂度与数据集匹配
2. 训练策略优化
3. 数据利用效率提升
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import h5py
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class ImprovedComplexConv1d(nn.Module):
    """改进的复数卷积层 - 简化但更有效"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=0.1):
        super().__init__()
        
        # 简化的复数卷积设计
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()  # 使用GELU激活函数
        self.dropout = nn.Dropout(dropout)
        
        # 简化的残差连接
        self.use_residual = (in_channels == out_channels and stride == 1)
        if not self.use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1, stride, bias=False)
        
    def forward(self, x):
        identity = x
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # 残差连接
        if self.use_residual:
            out = out + identity
        elif hasattr(self, 'residual_proj'):
            out = out + self.residual_proj(identity)
            
        return out

class ImprovedChannelAttention(nn.Module):
    """改进的通道注意力机制 - 更轻量但有效"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 简化的注意力网络
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _ = x.size()
        
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # 注意力权重
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1)
        return x * attention

class ImprovedMSAC(nn.Module):
    """改进的多尺度注意力卷积网络"""
    def __init__(self, num_classes=24):
        super().__init__()
        
        # 输入处理 - 直接处理2通道I/Q数据
        self.input_conv = ImprovedComplexConv1d(2, 64, 7, padding=3, dropout=0.1)
        
        # 多尺度特征提取 - 简化但保持有效性
        self.scale1 = nn.Sequential(
            ImprovedComplexConv1d(64, 64, 3, padding=1, dropout=0.1),
            ImprovedComplexConv1d(64, 64, 3, padding=1, dropout=0.1)
        )
        
        self.scale2 = nn.Sequential(
            ImprovedComplexConv1d(64, 64, 5, padding=2, dropout=0.1),
            ImprovedComplexConv1d(64, 64, 5, padding=2, dropout=0.1)
        )
        
        self.scale3 = nn.Sequential(
            ImprovedComplexConv1d(64, 64, 7, padding=3, dropout=0.1),
            ImprovedComplexConv1d(64, 64, 7, padding=3, dropout=0.1)
        )
        
        # 特征融合
        self.fusion_conv = ImprovedComplexConv1d(192, 128, 1, dropout=0.1)
        
        # 通道注意力
        self.channel_attention = ImprovedChannelAttention(128, reduction=8)
        
        # 下采样
        self.downsample = nn.Sequential(
            nn.Conv1d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 全局特征提取
        self.global_conv = nn.Sequential(
            ImprovedComplexConv1d(128, 256, 3, padding=1, dropout=0.2),
            ImprovedComplexConv1d(256, 256, 3, padding=1, dropout=0.2)
        )
        
        # 最终注意力
        self.final_attention = ImprovedChannelAttention(256, reduction=16)
        
        # 分类器 - 简化设计
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入处理
        x = self.input_conv(x)
        
        # 多尺度特征提取
        scale1_out = self.scale1(x)
        scale2_out = self.scale2(x)
        scale3_out = self.scale3(x)
        
        # 特征融合
        fused = torch.cat([scale1_out, scale2_out, scale3_out], dim=1)
        fused = self.fusion_conv(fused)
        
        # 通道注意力
        attended = self.channel_attention(fused)
        
        # 下采样
        downsampled = self.downsample(attended)
        
        # 全局特征提取
        global_features = self.global_conv(downsampled)
        
        # 最终注意力
        final_features = self.final_attention(global_features)
        
        # 全局池化和分类
        pooled = self.global_pool(final_features).squeeze(-1)
        output = self.classifier(pooled)
        
        return output

class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).to(inputs.device).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        loss = -(targets_smooth * log_probs).sum(dim=1).mean()
        return loss

def load_radioml2018_data():
    """加载RadioML 2018.01A数据集"""
    print("正在加载RadioML 2018.01A数据集...")
    
    data_path = "dataset/RadioML 2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件未找到: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        X = f['X'][:]  # 信号数据
        Y = f['Y'][:]  # 标签
        Z = f['Z'][:]  # SNR
    
    print(f"数据形状: X={X.shape}, Y={Y.shape}, Z={Z.shape}")
    
    # 转换为PyTorch张量
    X = torch.FloatTensor(X)
    Y = torch.LongTensor(np.argmax(Y, axis=1))
    Z = torch.FloatTensor(Z)
    
    # 转换数据维度：从(batch, length, channels)到(batch, channels, length)
    X = X.permute(0, 2, 1)  # (batch, 1024, 2) -> (batch, 2, 1024)
    
    # 数据预处理 - 改进的归一化
    X_mean = X.mean(dim=(0, 2), keepdim=True)
    X_std = X.std(dim=(0, 2), keepdim=True) + 1e-8
    X = (X - X_mean) / X_std
    
    return X, Y, Z

def create_improved_data_splits(X, Y, Z, train_ratio=0.6, val_ratio=0.2):
    """创建改进的数据分割 - 确保SNR分布平衡"""
    print("创建改进的数据分割...")
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    # 将Z转换为一维数组
    Z_flat = Z.squeeze() if Z.dim() > 1 else Z
    
    # 按SNR分层采样
    unique_snrs = np.unique(Z_flat)
    train_indices, val_indices, test_indices = [], [], []
    
    for snr in unique_snrs:
        snr_mask = (Z_flat == snr).numpy()
        snr_indices = indices[snr_mask]
        np.random.shuffle(snr_indices)
        
        n_snr = len(snr_indices)
        n_train = int(n_snr * train_ratio)
        n_val = int(n_snr * val_ratio)
        
        train_indices.extend(snr_indices[:n_train])
        val_indices.extend(snr_indices[n_train:n_train+n_val])
        test_indices.extend(snr_indices[n_train+n_val:])
    
    # 打乱索引
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    print(f"训练集: {len(train_indices)}, 验证集: {len(val_indices)}, 测试集: {len(test_indices)}")
    
    return train_indices, val_indices, test_indices

def create_improved_dataloaders(X, Y, Z, train_indices, val_indices, test_indices, batch_size=64):
    """创建改进的数据加载器"""
    
    # 创建数据集
    train_dataset = TensorDataset(X[train_indices], Y[train_indices], Z[train_indices])
    val_dataset = TensorDataset(X[val_indices], Y[val_indices], Z[val_indices])
    test_dataset = TensorDataset(X[test_indices], Y[test_indices], Z[test_indices])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def train_improved_model(model, train_loader, val_loader, device, epochs=50):
    """改进的训练函数"""
    print("开始改进的模型训练...")
    
    # 优化器 - 使用AdamW和余弦退火
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 损失函数 - 组合Focal Loss和Label Smoothing
    focal_loss = FocalLoss(alpha=1, gamma=2)
    smooth_loss = LabelSmoothingLoss(num_classes=24, smoothing=0.1)
    
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target, snr) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # 组合损失
            loss1 = focal_loss(output, target)
            loss2 = smooth_loss(output, target)
            loss = 0.7 * loss1 + 0.3 * loss2
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target, snr in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                loss1 = focal_loss(output, target)
                loss2 = smooth_loss(output, target)
                loss = 0.7 * loss1 + 0.3 * loss2
                val_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_improved_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"早停触发，最佳验证准确率: {best_val_acc:.4f}")
            break
    
    return train_losses, val_losses, val_accuracies, best_val_acc

def evaluate_improved_model(model, test_loader, device):
    """改进的模型评估"""
    print("评估改进模型...")
    
    model.eval()
    all_preds = []
    all_targets = []
    all_snrs = []
    
    with torch.no_grad():
        for data, target, snr in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_snrs.extend(snr.cpu().numpy())
    
    # 计算总体准确率
    overall_acc = accuracy_score(all_targets, all_preds)
    
    # 按SNR计算准确率
    snr_accuracies = {}
    unique_snrs = np.unique(all_snrs)
    
    for snr in unique_snrs:
        snr_mask = np.array(all_snrs) == snr
        snr_acc = accuracy_score(
            np.array(all_targets)[snr_mask], 
            np.array(all_preds)[snr_mask]
        )
        snr_accuracies[snr] = snr_acc
    
    return overall_acc, snr_accuracies, all_targets, all_preds

def main():
    print("=" * 60)
    print("改进的RadioML 2018实验 - 针对性能瓶颈优化")
    print("=" * 60)
    
    start_time = time.time()
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 加载数据
        X, Y, Z = load_radioml2018_data()
        
        # 创建数据分割
        train_indices, val_indices, test_indices = create_improved_data_splits(X, Y, Z)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_improved_dataloaders(
            X, Y, Z, train_indices, val_indices, test_indices, batch_size=128
        )
        
        # 创建模型
        model = ImprovedMSAC(num_classes=24).to(device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"模型参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 训练模型
        train_losses, val_losses, val_accuracies, best_val_acc = train_improved_model(
            model, train_loader, val_loader, device, epochs=60
        )
        
        # 加载最佳模型
        model.load_state_dict(torch.load('best_improved_model.pth'))
        
        # 评估模型
        test_acc, snr_accuracies, all_targets, all_preds = evaluate_improved_model(
            model, test_loader, device
        )
        
        # 计算训练时间
        training_time = time.time() - start_time
        
        # 打印结果
        print("\n" + "=" * 60)
        print("改进实验结果总结")
        print("=" * 60)
        print(f"模型参数量: {total_params:,}")
        print(f"训练时间: {training_time:.1f}秒")
        print(f"训练轮数: {len(val_accuracies)}")
        print(f"最佳验证准确率: {best_val_acc:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        print(f"平均每epoch时间: {training_time/len(val_accuracies):.1f}秒")
        
        # 按SNR显示结果
        print("\n按SNR的准确率:")
        for snr in sorted(snr_accuracies.keys()):
            print(f"SNR {snr:2.0f}dB: {snr_accuracies[snr]:.4f}")
        
        # 保存结果
        results = {
            'model_params': total_params,
            'training_time': training_time,
            'epochs': len(val_accuracies),
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'avg_epoch_time': training_time / len(val_accuracies),
            'snr_accuracies': {str(k): v for k, v in snr_accuracies.items()},
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
        
        with open('improved_experiment_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n结果已保存到 improved_experiment_results.json")
        
    except Exception as e:
        print(f"实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()