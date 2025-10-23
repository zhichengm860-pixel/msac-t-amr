#!/usr/bin/env python3
"""
基线模型对比实验
比较不同基线模型的性能
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report

from src.utils import Config
from src.models import ResNet1D, CLDNN, MCformer
from src.data import DatasetLoader
from src.data import get_radioml_config

def setup_experiment():
    """设置实验环境"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/baseline_comparison_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    
    device = torch.device('cpu')
    return exp_dir, device

def load_data():
    """加载数据"""
    print("=== 加载数据 ===")
    dataset_config, preprocess_config, dataloader_config = get_radioml_config('2016.10a')
    
    loader = DatasetLoader(dataset_config)
    train_loader, val_loader, test_loader = loader.load_radioml_data(
        dataset_config.path,
        batch_size=32,  # 使用较小的批次大小
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    return train_loader, val_loader, test_loader, dataset_config

def create_models(num_classes, device):
    """创建不同的基线模型"""
    models = {}
    
    # 1. 轻量级ResNet
    models['ResNet1D_Light'] = ResNet1D(
        num_classes=num_classes,
        input_channels=2,
        layers=[1, 1, 1, 1]  # 轻量级配置
    ).to(device)
    
    # 2. CLDNN模型
    try:
        models['CLDNN'] = CLDNN(
            num_classes=num_classes,
            input_channels=2
        ).to(device)
    except Exception as e:
        print(f"CLDNN模型创建失败: {e}")
    
    # 3. MCformer模型
    try:
        models['MCformer'] = MCformer(
            num_classes=num_classes,
            input_channels=2
        ).to(device)
    except Exception as e:
        print(f"MCformer模型创建失败: {e}")
    
    return models

def train_model_quick(model, train_loader, val_loader, device, model_name, exp_dir):
    """快速训练模型"""
    print(f"\n=== 训练 {model_name} ===")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 2  # 快速训练，只训练2轮
    max_batches = 30  # 限制批次数量
    
    best_val_acc = 0.0
    train_history = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (signals, labels, snrs) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
                
            signals = signals.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{max_batches}, "
                      f"Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%")
        
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
        
        # 计算平均值
        avg_train_loss = train_loss / max_batches
        train_acc = 100. * train_correct / train_total
        avg_val_loss = val_loss / min(20, len(val_loader))
        val_acc = 100. * val_correct / val_total
        
        print(f"  Epoch {epoch+1} Results:")
        print(f"    Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(exp_dir, "checkpoints", f"{model_name}_best.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    保存最佳模型，验证准确率: {val_acc:.2f}%")
        
        # 记录历史
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': float(avg_train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(avg_val_loss),
            'val_acc': float(val_acc)
        })
    
    return train_history, best_val_acc

def evaluate_model_quick(model, test_loader, device, model_name):
    """快速评估模型"""
    print(f"\n=== 评估 {model_name} ===")
    
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
    
    print(f"  测试结果:")
    print(f"    测试损失: {avg_test_loss:.4f}")
    print(f"    测试准确率: {test_acc*100:.2f}%")
    
    return {
        'test_loss': float(avg_test_loss),
        'test_accuracy': float(test_acc),
        'predictions': [int(p) for p in all_predictions],
        'labels': [int(l) for l in all_labels]
    }

def main():
    """主函数"""
    try:
        print("=== 基线模型对比实验开始 ===")
        
        # 设置实验
        exp_dir, device = setup_experiment()
        print(f"实验目录: {exp_dir}")
        print(f"使用设备: {device}")
        
        # 加载数据
        train_loader, val_loader, test_loader, dataset_config = load_data()
        
        # 创建模型
        print(f"\n=== 创建模型 ===")
        models = create_models(dataset_config.num_classes, device)
        
        print(f"成功创建 {len(models)} 个模型:")
        for name, model in models.items():
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  {name}: {total_params:,} 参数")
        
        # 训练和评估所有模型
        all_results = {}
        
        for model_name, model in models.items():
            start_time = time.time()
            
            # 训练模型
            train_history, best_val_acc = train_model_quick(
                model, train_loader, val_loader, device, model_name, exp_dir
            )
            
            # 评估模型
            test_results = evaluate_model_quick(model, test_loader, device, model_name)
            
            training_time = time.time() - start_time
            
            # 保存结果
            model_results = {
                'model_name': model_name,
                'parameters': sum(p.numel() for p in model.parameters()),
                'training_time': float(training_time),
                'best_val_accuracy': float(best_val_acc),
                'test_results': test_results,
                'train_history': train_history
            }
            
            all_results[model_name] = model_results
            
            print(f"\n{model_name} 完成:")
            print(f"  训练时间: {training_time:.2f}秒")
            print(f"  最佳验证准确率: {best_val_acc:.2f}%")
            print(f"  测试准确率: {test_results['test_accuracy']*100:.2f}%")
        
        # 生成对比报告
        print(f"\n=== 模型对比结果 ===")
        print(f"{'模型名称':<15} {'参数数量':<10} {'训练时间(s)':<12} {'验证准确率(%)':<15} {'测试准确率(%)':<15}")
        print("-" * 80)
        
        for name, results in all_results.items():
            print(f"{name:<15} {results['parameters']:<10,} {results['training_time']:<12.2f} "
                  f"{results['best_val_accuracy']:<15.2f} {results['test_results']['test_accuracy']*100:<15.2f}")
        
        # 保存完整结果
        results_path = os.path.join(exp_dir, "comparison_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n=== 对比实验完成 ===")
        print(f"结果保存在: {exp_dir}")
        
    except Exception as e:
        print(f"实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()