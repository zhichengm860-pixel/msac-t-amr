#!/usr/bin/env python3
"""
简化的训练脚本
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
from sklearn.metrics import accuracy_score, classification_report

from src.utils import Config
from src.models import ResNet1D  # 使用轻量的ResNet模型
from src.data import DatasetLoader
from src.data import get_radioml_config

def setup_experiment():
    """设置实验环境"""
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/simple_training_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    
    # 设置设备
    device = torch.device('cpu')  # 强制使用CPU
    
    return exp_dir, device

def load_training_data():
    """加载训练数据"""
    print("=== 加载训练数据 ===")
    
    # 获取数据集配置
    dataset_config, preprocess_config, dataloader_config = get_radioml_config('2016.10a')
    
    # 加载数据
    loader = DatasetLoader(dataset_config)
    train_loader, val_loader, test_loader = loader.load_radioml_data(
        dataset_config.path,
        batch_size=16,  # 进一步减小批次大小
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    print(f"训练数据加载完成:")
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")
    print(f"  测试批次: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, dataset_config

def train_model(model, train_loader, val_loader, device, exp_dir):
    """训练模型"""
    print("=== 开始训练 ===")
    
    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 训练参数
    epochs = 3  # 减少训练轮数
    max_batches_per_epoch = 50  # 限制每轮的批次数量
    
    print(f"开始训练，共 {epochs} 轮，每轮最多 {max_batches_per_epoch} 批次...")
    
    best_val_acc = 0.0
    train_history = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        for batch_idx, (signals, labels, snrs) in enumerate(train_loader):
            if batch_idx >= max_batches_per_epoch:
                break
                
            signals = signals.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx+1}/{max_batches_per_epoch}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {100.*train_correct/train_total:.2f}%")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (signals, labels, snrs) in enumerate(val_loader):
                if batch_idx >= 20:  # 限制验证批次
                    break
                    
                signals = signals.to(device)
                labels = labels.to(device)
                
                outputs = model(signals)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 计算平均指标
        avg_train_loss = train_loss / max_batches_per_epoch
        train_acc = 100. * train_correct / train_total
        avg_val_loss = val_loss / min(20, len(val_loader))
        val_acc = 100. * val_correct / val_total
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoints", "best_model.pth"))
            print(f"  保存最佳模型，验证准确率: {best_val_acc:.2f}%")
        
        # 记录训练历史
        train_history.append({
            'epoch': int(epoch + 1),
            'train_loss': float(avg_train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(avg_val_loss),
            'val_acc': float(val_acc)
        })
    
    return train_history, best_val_acc

def evaluate_model(model, test_loader, device):
    """评估模型"""
    print("\n=== 模型评估 ===")
    
    model.eval()
    all_predictions = []
    all_labels = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (signals, labels, snrs) in enumerate(test_loader):
            if batch_idx >= 30:  # 限制测试批次
                break
                
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    test_acc = accuracy_score(all_labels, all_predictions)
    avg_test_loss = test_loss / min(30, len(test_loader))
    
    print(f"测试结果:")
    print(f"  测试损失: {avg_test_loss:.4f}")
    print(f"  测试准确率: {test_acc*100:.2f}%")
    
    return {
        'test_loss': float(avg_test_loss),
        'test_accuracy': float(test_acc),
        'predictions': [int(p) for p in all_predictions],
        'labels': [int(l) for l in all_labels]
    }

def main():
    """主函数"""
    print("开始简化训练实验...")
    
    # 设置实验
    exp_dir, device = setup_experiment()
    print(f"=== 实验设置 ===")
    print(f"实验目录: {exp_dir}")
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, val_loader, test_loader, dataset_config = load_training_data()
    
    # 创建模型
    print(f"\n=== 模型设置 ===")
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
    print(f"模型参数数量: {total_params:,}")
    
    # 保存配置
    config_dict = {
        'model': 'ResNet1D-Simple',
        'num_classes': dataset_config.num_classes,
        'signal_length': dataset_config.signal_length,
        'input_channels': 2,
        'layers': [1, 1, 1, 1],
        'batch_size': 16,
        'device': str(device),
        'total_params': total_params
    }
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    try:
        # 训练模型
        train_history, best_val_acc = train_model(model, train_loader, val_loader, device, exp_dir)
        
        # 评估模型
        test_results = evaluate_model(model, test_loader, device)
        
        # 保存结果
        results = {
            'train_history': train_history,
            'best_val_accuracy': best_val_acc,
            'test_results': test_results
        }
        
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n=== 训练完成 ===")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"测试准确率: {test_results['test_accuracy']*100:.2f}%")
        print(f"实验结果保存在: {exp_dir}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()