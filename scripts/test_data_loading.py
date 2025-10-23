#!/usr/bin/env python3
"""
测试数据预处理和加载功能
"""

import torch
import numpy as np
from src.utils import Config
from src.data import DatasetLoader
from src.data import get_radioml_config
import time

def test_data_loading():
    print("=== 数据加载测试 ===")
    
    # 加载配置
    config = Config()
    
    # 测试RadioML 2016.10A数据集
    print("\n1. 测试RadioML 2016.10A数据集加载...")
    try:
        dataset_config, preprocess_config, dataloader_config = get_radioml_config("2016.10a", batch_size=32)
        
        # 创建数据加载器
        data_loader = DatasetLoader(config)
        
        # 加载数据
        start_time = time.time()
        train_loader, val_loader, test_loader = data_loader.load_radioml_data(
            dataset_path=dataset_config.path,
            batch_size=dataloader_config.batch_size,
            train_ratio=dataloader_config.train_ratio,
            val_ratio=dataloader_config.val_ratio
        )
        load_time = time.time() - start_time
        
        print(f"✓ 数据加载成功，耗时: {load_time:.2f}秒")
        print(f"  训练集批次数: {len(train_loader)}")
        print(f"  验证集批次数: {len(val_loader)}")
        print(f"  测试集批次数: {len(test_loader)}")
        
        # 测试一个批次
        print("\n2. 测试数据批次...")
        for signals, labels, snrs in train_loader:
            print(f"  信号形状: {signals.shape}")
            print(f"  标签形状: {labels.shape}")
            print(f"  SNR形状: {snrs.shape}")
            print(f"  信号数据类型: {signals.dtype}")
            print(f"  标签范围: {labels.min().item()} - {labels.max().item()}")
            print(f"  SNR范围: {snrs.min().item():.1f} - {snrs.max().item():.1f}")
            break
        
        # 测试数据统计
        print("\n3. 数据统计信息...")
        all_labels = []
        all_snrs = []
        
        for signals, labels, snrs in train_loader:
            all_labels.extend(labels.numpy())
            all_snrs.extend(snrs.numpy())
            if len(all_labels) > 1000:  # 只统计前1000个样本
                break
        
        all_labels = np.array(all_labels)
        all_snrs = np.array(all_snrs)
        
        print(f"  类别分布: {np.bincount(all_labels)}")
        print(f"  SNR分布: 最小={all_snrs.min():.1f}, 最大={all_snrs.max():.1f}, 平均={all_snrs.mean():.1f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_compatibility():
    print("\n=== 模型兼容性测试 ===")
    
    try:
        from src.models import AMRNet
        from src.utils import Config
        
        config = Config()
        
        # 创建模型
        model = AMRNet(
            num_classes=config.model.num_classes,
            input_channels=1,
            base_channels=64,
            num_transformer_blocks=4,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout
        )
        print(f"✓ 模型创建成功")
        print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        batch_size = 8
        signal_length = 128
        
        # 创建测试数据
        test_signals = torch.randn(batch_size, 2, signal_length)  # I/Q两个通道
        test_snrs = torch.randn(batch_size, 1)
        
        print(f"\n测试输入形状:")
        print(f"  信号: {test_signals.shape}")
        print(f"  SNR: {test_snrs.shape}")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(test_signals, test_snrs)
        
        print(f"\n测试输出:")
        print(f"  输出类型: {type(outputs)}")
        
        if isinstance(outputs, dict):
            print(f"  输出键: {list(outputs.keys())}")
            logits = outputs['logits']
            print(f"  分类输出形状: {logits.shape}")
            print(f"  分类输出范围: {logits.min().item():.4f} - {logits.max().item():.4f}")
            
            # 测试概率输出
            probs = torch.softmax(logits, dim=1)
            print(f"  概率和: {probs.sum(dim=1).mean().item():.4f} (应该接近1.0)")
            
            # 检查其他输出
            if 'snr_pred' in outputs:
                snr_pred = outputs['snr_pred']
                print(f"  SNR预测形状: {snr_pred.shape}")
                print(f"  SNR预测范围: {snr_pred.min().item():.2f} - {snr_pred.max().item():.2f}")
        else:
            print(f"  输出形状: {outputs.shape}")
            print(f"  输出范围: {outputs.min().item():.4f} - {outputs.max().item():.4f}")
            
            # 测试概率输出
            probs = torch.softmax(outputs, dim=1)
            print(f"  概率和: {probs.sum(dim=1).mean().item():.4f} (应该接近1.0)")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("开始数据预处理和加载测试...")
    
    # 测试数据加载
    data_success = test_data_loading()
    
    # 测试模型兼容性
    model_success = test_model_compatibility()
    
    print(f"\n=== 测试总结 ===")
    print(f"数据加载测试: {'✓ 通过' if data_success else '✗ 失败'}")
    print(f"模型兼容性测试: {'✓ 通过' if model_success else '✗ 失败'}")
    
    if data_success and model_success:
        print("\n🎉 所有测试通过！可以开始训练实验。")
        return True
    else:
        print("\n❌ 存在问题，请检查配置。")
        return False

if __name__ == "__main__":
    main()