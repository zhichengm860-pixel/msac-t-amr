#!/usr/bin/env python3
"""
自监督预训练实验脚本
执行重建、对比学习和掩码预测等预训练任务
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

from src.utils import Config
from src.models import MSAC_T
from src.data import DatasetLoader
from src.data import get_radioml_config
from src.utils import ExperimentTracker


def setup_experiment():
    """设置实验环境"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/pretrain_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return exp_dir, device


def load_pretrain_data(config):
    """加载预训练数据"""
    print("\n=== 加载预训练数据 ===")
    
    # 获取数据集配置
    dataset_config, preprocess_config, dataloader_config = get_radioml_config('2016.10a')
    
    # 加载数据
    loader = DatasetLoader(dataset_config)
    train_loader, val_loader, test_loader = loader.load_radioml_data(
        dataset_config.path,
        batch_size=config.training.batch_size,
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    print(f"预训练数据加载完成:")
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")
    
    return train_loader, val_loader, test_loader


def run_reconstruction_pretrain(model, train_loader, val_loader, device, exp_dir):
    """运行重建预训练"""
    print("\n=== 重建预训练 ===")
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # 训练参数
    epochs = 5  # 减少轮数用于测试
    best_loss = float('inf')
    
    print(f"开始重建预训练，共 {epochs} 轮...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 训练
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for signals, labels, snrs in train_loader:
            signals = signals.to(device)
            snrs = snrs.to(device)
            
            optimizer.zero_grad()
            
            # 重建损失
            outputs = model(signals, snrs)
            recon_loss = nn.MSELoss()(outputs['signal_recon'], signals.unsqueeze(1))
            
            recon_loss.backward()
            optimizer.step()
            
            train_loss += recon_loss.item()
            train_batches += 1
            
            if train_batches >= 50:  # 限制训练批次数
                break
        
        train_loss /= train_batches
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for signals, labels, snrs in val_loader:
                signals = signals.to(device)
                snrs = snrs.to(device)
                
                outputs = model(signals, snrs)
                recon_loss = nn.MSELoss()(outputs['signal_recon'], signals.unsqueeze(1))
                
                val_loss += recon_loss.item()
                val_batches += 1
                
                if val_batches >= 25:  # 限制验证批次数
                    break
        
        val_loss /= val_batches
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_reconstruction_model.pth'))
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")
    
    print(f"重建预训练完成，最佳验证损失: {best_loss:.4f}")
    return best_loss


def run_contrastive_pretrain(model, train_loader, val_loader, device, exp_dir):
    """运行对比学习预训练"""
    print("\n=== 对比学习预训练 ===")
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # 训练参数
    epochs = 5  # 减少轮数用于测试
    best_loss = float('inf')
    
    print(f"开始对比学习预训练，共 {epochs} 轮...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 训练
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for signals, labels, snrs in train_loader:
            signals = signals.to(device)
            snrs = snrs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 对比学习损失
            outputs = model(signals, snrs)
            projections = outputs['projection']
            
            # 简单的对比损失（InfoNCE）
            batch_size = projections.size(0)
            similarity_matrix = torch.mm(projections, projections.t())
            
            # 创建正样本掩码（同一批次内的样本）
            labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
            positive_mask = labels_matrix.float()
            
            # 计算对比损失
            exp_sim = torch.exp(similarity_matrix / 0.1)
            positive_sum = (exp_sim * positive_mask).sum(dim=1)
            total_sum = exp_sim.sum(dim=1)
            
            # 避免除零
            positive_sum = torch.clamp(positive_sum, min=1e-8)
            contrastive_loss = -torch.log(positive_sum / total_sum).mean()
            
            contrastive_loss.backward()
            optimizer.step()
            
            train_loss += contrastive_loss.item()
            train_batches += 1
            
            if train_batches >= 50:  # 限制训练批次数
                break
        
        train_loss /= train_batches
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for signals, labels, snrs in val_loader:
                signals = signals.to(device)
                snrs = snrs.to(device)
                labels = labels.to(device)
                
                outputs = model(signals, snrs)
                projections = outputs['projection']
                
                # 计算验证对比损失
                batch_size = projections.size(0)
                similarity_matrix = torch.mm(projections, projections.t())
                
                labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
                positive_mask = labels_matrix.float()
                
                exp_sim = torch.exp(similarity_matrix / 0.1)
                positive_sum = (exp_sim * positive_mask).sum(dim=1)
                total_sum = exp_sim.sum(dim=1)
                
                positive_sum = torch.clamp(positive_sum, min=1e-8)
                contrastive_loss = -torch.log(positive_sum / total_sum).mean()
                
                val_loss += contrastive_loss.item()
                val_batches += 1
                
                if val_batches >= 25:  # 限制验证批次数
                    break
        
        val_loss /= val_batches
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_contrastive_model.pth'))
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")
    
    print(f"对比学习预训练完成，最佳验证损失: {best_loss:.4f}")
    return best_loss


def main():
    """主函数"""
    print("开始自监督预训练实验...")
    
    # 设置实验
    exp_dir, device = setup_experiment()
    print(f"=== 自监督预训练实验设置 ===")
    print(f"实验目录: {exp_dir}")
    print(f"使用设备: {device}")
    
    # 加载配置
    config = Config()
    
    # 加载数据
    train_loader, val_loader, test_loader = load_pretrain_data(config)
    
    # 根据数据集配置创建模型
    dataset_config, _, _ = get_radioml_config('2016.10a')
    print(f"数据集: {dataset_config.name}")
    print(f"调制类型数量: {dataset_config.num_classes}")
    print(f"信号长度: {dataset_config.signal_length}")
    
    model = MSAC_T(
        num_classes=dataset_config.num_classes,
        input_channels=1,
        base_channels=64,
        num_transformer_blocks=4,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数数量: {total_params:,}")
    
    # 初始化实验跟踪器
    tracker = ExperimentTracker(exp_dir)
    
    # 保存配置
    config_dict = {
        'model': 'MSAC-T',
        'num_classes': dataset_config.num_classes,
        'signal_length': dataset_config.signal_length,
        'input_channels': 1,
        'base_channels': 64,
        'batch_size': config.training.batch_size,
        'device': str(device),
        'total_params': total_params
    }
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # 运行预训练实验
    results = {}
    
    try:
        # 1. 重建预训练
        results['reconstruction'] = run_reconstruction_pretrain(
            model, train_loader, val_loader, device, exp_dir
        )
        
        # 2. 对比学习预训练
        results['contrastive'] = run_contrastive_pretrain(
            model, train_loader, val_loader, device, exp_dir
        )
        
        print("\n=== 预训练实验完成 ===")
        print(f"重建预训练最佳损失: {results['reconstruction']:.4f}")
        print(f"对比学习预训练最佳损失: {results['contrastive']:.4f}")
        
        # 保存结果
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n实验结果已保存到: {exp_dir}")
        
    except Exception as e:
        print(f"预训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理资源
        if 'tracker' in locals():
            tracker.close()


if __name__ == "__main__":
    main()