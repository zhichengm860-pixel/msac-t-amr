#!/usr/bin/env python3
"""
快速训练实验 - 用于论文初稿
减少参数和训练时间，快速获得初步结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import h5py
import os
import time
from datetime import datetime
import json

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class QuickComplexConv1d(nn.Module):
    """简化的复数卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # 实部和虚部分离后，每个部分的通道数是in_channels//2
        single_channel = in_channels // 2 if in_channels > 1 else 1
        self.real_conv = nn.Conv1d(single_channel, out_channels, kernel_size, stride, padding)
        self.imag_conv = nn.Conv1d(single_channel, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        # x shape: (batch, channels, length) - 前一半是实部，后一半是虚部
        channels = x.size(1)
        half_channels = channels // 2
        
        real_part = x[:, :half_channels, :]  # 前一半通道是实部
        imag_part = x[:, half_channels:, :]  # 后一半通道是虚部
        
        # 复数卷积
        real_out = self.real_conv(real_part) - self.imag_conv(imag_part)
        imag_out = self.real_conv(imag_part) + self.imag_conv(real_part)
        
        return torch.cat([real_out, imag_out], dim=1)

class QuickAttention(nn.Module):
    """简化的注意力机制"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class QuickMSACModel(nn.Module):
    """快速版本的多尺度注意力复数网络"""
    def __init__(self, num_classes=24):
        super().__init__()
        
        # 输入层 - 减少通道数
        self.input_conv = QuickComplexConv1d(2, 16, kernel_size=7, padding=3)  # 输入2通道(实部+虚部)
        self.input_bn = nn.BatchNorm1d(32)  # 16*2=32
        
        # 多尺度特征提取 - 简化版本
        self.scale1 = nn.Sequential(
            QuickComplexConv1d(32, 24, kernel_size=3, padding=1),  # 输入32通道(16*2)
            nn.BatchNorm1d(48),
            nn.GELU(),
            QuickAttention(48)
        )
        
        self.scale2 = nn.Sequential(
            QuickComplexConv1d(32, 24, kernel_size=5, padding=2),  # 输入32通道(16*2)
            nn.BatchNorm1d(48),
            nn.GELU(),
            QuickAttention(48)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=1),  # 48+48=96 -> 64
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # 全局池化和分类器
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),  # 减少隐藏层大小
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)
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
        x = self.input_bn(x)
        x = F.gelu(x)
        
        # 多尺度特征提取
        scale1_out = self.scale1(x)
        scale2_out = self.scale2(x)
        
        # 特征融合
        fused = torch.cat([scale1_out, scale2_out], dim=1)
        fused = self.fusion(fused)
        
        # 全局池化和分类
        pooled = self.global_pool(fused).squeeze(-1)
        output = self.classifier(pooled)
        
        return output

def load_quick_data():
    """快速数据加载 - 使用更小的数据子集"""
    print("正在加载RadioML 2018.01A数据集（快速版本）...")
    
    data_path = 'dataset/RadioML 2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5'
    
    with h5py.File(data_path, 'r') as f:
        X = f['X'][:]  # 信号数据
        Y = f['Y'][:]  # 标签
        Z = f['Z'][:]  # SNR
    
    print(f"原始数据形状: X={X.shape}, Y={Y.shape}, Z={Z.shape}")
    
    # 使用数据子集 - 只用20%的数据进行快速训练
    subset_size = len(X) // 5  # 使用1/5的数据
    indices = np.random.choice(len(X), subset_size, replace=False)
    
    X = X[indices]
    Y = Y[indices]
    Z = Z[indices]
    
    print(f"子集数据形状: X={X.shape}, Y={Y.shape}, Z={Z.shape}")
    
    # 转换为PyTorch张量
    X = torch.FloatTensor(X)
    Y = torch.LongTensor(np.argmax(Y, axis=1))
    Z = torch.FloatTensor(Z)
    
    # 转换数据维度：从(batch, length, channels)到(batch, channels, length)
    X = X.permute(0, 2, 1)  # (batch, 1024, 2) -> (batch, 2, 1024)
    
    # 简化的归一化
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    return X, Y, Z

def create_quick_data_splits(X, Y, Z, train_ratio=0.7, val_ratio=0.15):
    """快速数据分割"""
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    return {
        'train': (X[train_indices], Y[train_indices]),
        'val': (X[val_indices], Y[val_indices]),
        'test': (X[test_indices], Y[test_indices])
    }

def train_quick_model(model, train_loader, val_loader, device, epochs=10):  # 减少到10个epoch
    """快速训练函数"""
    print(f"\n开始快速训练 - {epochs} epochs")
    
    # 使用更大的学习率加速训练
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # 每50个batch打印一次进度
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'  验证准确率: {val_acc:.2f}%')
        print(f'  学习率: {scheduler.get_last_lr()[0]:.6f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_quick_model.pth')
            print(f'  ✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)')
        
        scheduler.step()
        print('-' * 60)
    
    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

def evaluate_quick_model(model, test_loader, device):
    """快速评估函数"""
    print("\n开始模型评估...")
    
    model.eval()
    test_correct = 0
    test_total = 0
    class_correct = np.zeros(24)
    class_total = np.zeros(24)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
            
            # 按类别统计
            for i in range(target.size(0)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    test_acc = 100. * test_correct / test_total
    
    print(f"测试准确率: {test_acc:.2f}%")
    
    # 按类别准确率
    print("\n各调制类型准确率:")
    modulation_types = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 
                       'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK',
                       '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC',
                       'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC',
                       'OOK', '16QAM']
    
    for i, mod_type in enumerate(modulation_types):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"  {mod_type}: {acc:.1f}%")
    
    return test_acc

def main():
    print("=" * 60)
    print("快速训练实验 - 用于论文初稿")
    print("=" * 60)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 数据加载
        X, Y, Z = load_quick_data()
        
        # 数据分割
        print("创建数据分割...")
        data_splits = create_quick_data_splits(X, Y, Z)
        
        # 创建数据加载器 - 使用更大的batch size
        batch_size = 256  # 增大batch size
        train_dataset = TensorDataset(data_splits['train'][0], data_splits['train'][1])
        val_dataset = TensorDataset(data_splits['val'][0], data_splits['val'][1])
        test_dataset = TensorDataset(data_splits['test'][0], data_splits['test'][1])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"数据加载器创建完成:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        # 创建模型
        print("创建快速模型...")
        model = QuickMSACModel(num_classes=24).to(device)
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
        
        # 训练模型
        start_time = time.time()
        training_results = train_quick_model(model, train_loader, val_loader, device, epochs=10)
        training_time = time.time() - start_time
        
        # 加载最佳模型进行测试
        print("加载最佳模型进行测试...")
        model.load_state_dict(torch.load('best_quick_model.pth'))
        test_acc = evaluate_quick_model(model, test_loader, device)
        
        # 保存结果
        results = {
            'experiment_type': 'quick_training',
            'timestamp': datetime.now().isoformat(),
            'model_params': total_params,
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'best_val_accuracy': training_results['best_val_acc'],
            'test_accuracy': test_acc,
            'train_losses': training_results['train_losses'],
            'val_accuracies': training_results['val_accuracies'],
            'data_subset_ratio': 0.2,  # 使用了20%的数据
            'epochs': 10,
            'batch_size': batch_size
        }
        
        # 保存结果到JSON文件
        with open('quick_experiment_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印最终结果
        print("\n" + "=" * 60)
        print("快速训练实验完成！")
        print("=" * 60)
        print(f"训练时间: {training_time/60:.1f} 分钟")
        print(f"最佳验证准确率: {training_results['best_val_acc']:.2f}%")
        print(f"测试准确率: {test_acc:.2f}%")
        print(f"模型参数量: {total_params:,}")
        print(f"数据使用量: 20% (快速训练)")
        print("\n论文初稿可用的关键数据:")
        print(f"- 模型架构: 简化多尺度注意力复数网络")
        print(f"- 参数量: {total_params:,}")
        print(f"- 准确率: {test_acc:.2f}%")
        print(f"- 训练效率: {training_time/60:.1f} 分钟")
        print("\n结果已保存到: quick_experiment_results.json")
        print("模型已保存到: best_quick_model.pth")
        
    except Exception as e:
        print(f"实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()