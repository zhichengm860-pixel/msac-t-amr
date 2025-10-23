#!/usr/bin/env python3
"""
RadioML数据加载器测试模块

作者: Assistant
日期: 2025-01-16
"""

import os
import sys
import numpy as np
import torch
from radioml_dataloader import RadioMLDataLoader, RadioMLDataset
import matplotlib.pyplot as plt
import seaborn as sns

def test_radioml_dataset():
    """测试RadioMLDataset类"""
    print("测试RadioMLDataset类...")
    
    # 创建测试数据
    test_data = np.random.randn(100, 2, 128)
    test_labels = np.random.randint(0, 11, 100)
    
    # 创建数据集
    dataset = RadioMLDataset(test_data, test_labels)
    
    # 测试基本功能
    assert len(dataset) == 100, f"数据集长度错误: {len(dataset)}"
    
    sample, label = dataset[0]
    assert sample.shape == (2, 128), f"样本形状错误: {sample.shape}"
    assert isinstance(label, torch.Tensor), f"标签类型错误: {type(label)}"
    
    print("  ✓ RadioMLDataset基础功能测试通过")
    return True

def test_radioml_dataloader_2016():
    """测试RadioML 2016数据加载器"""
    print("测试RadioML 2016数据加载器...")
    
    # 创建数据加载器
    loader = RadioMLDataLoader("dummy_2016.pkl", "2016.10A")
    
    # 加载数据（会生成模拟数据）
    data, labels, mod_classes, snr_levels = loader.load_data()
    
    # 验证数据
    assert data.shape[1] == 2, f"I/Q通道数错误: {data.shape[1]}"
    assert data.shape[2] == 128, f"信号长度错误: {data.shape[2]}"
    assert len(mod_classes) == 11, f"调制类型数量错误: {len(mod_classes)}"
    
    # 获取数据集信息
    info = loader.get_dataset_info()
    assert info['num_classes'] == 11, f"类别数量错误: {info['num_classes']}"
    assert info['signal_length'] == 128, f"信号长度错误: {info['signal_length']}"
    
    print(f"  ✓ 数据形状: {data.shape}")
    print(f"  ✓ 调制类型: {mod_classes}")
    print(f"  ✓ SNR级别数量: {len(snr_levels)}")
    
    return True

def test_radioml_dataloader_2018():
    """测试RadioML 2018数据加载器"""
    print("测试RadioML 2018数据加载器...")
    
    # 创建数据加载器
    loader = RadioMLDataLoader("dummy_2018.h5", "2018.01A")
    
    # 加载数据（会生成模拟数据）
    data, labels, mod_classes, snr_levels = loader.load_data()
    
    # 验证数据
    assert data.shape[1] == 2, f"I/Q通道数错误: {data.shape[1]}"
    assert data.shape[2] == 1024, f"信号长度错误: {data.shape[2]}"
    assert len(mod_classes) == 24, f"调制类型数量错误: {len(mod_classes)}"
    
    print(f"  ✓ 数据形状: {data.shape}")
    print(f"  ✓ 调制类型数量: {len(mod_classes)}")
    print(f"  ✓ SNR级别数量: {len(snr_levels)}")
    
    return True

def test_dataloader_creation():
    """测试数据加载器创建"""
    print("测试数据加载器创建...")
    
    # 创建RadioML数据加载器
    loader = RadioMLDataLoader("dummy.pkl", "2016.10A")
    data, labels, mod_classes, snr_levels = loader.load_data()
    
    # 创建PyTorch数据加载器
    train_loader, val_loader, test_loader = loader.create_dataloaders(
        batch_size=32, test_size=0.2, val_size=0.1
    )
    
    # 测试数据加载
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))
    
    # 验证批次
    assert train_batch[0].shape[0] <= 32, f"训练批次大小错误: {train_batch[0].shape[0]}"
    assert train_batch[0].shape[1:] == (2, 128), f"训练数据形状错误: {train_batch[0].shape[1:]}"
    
    print(f"  ✓ 训练批次形状: {train_batch[0].shape}")
    print(f"  ✓ 验证批次形状: {val_batch[0].shape}")
    print(f"  ✓ 测试批次形状: {test_batch[0].shape}")
    
    return True

def test_class_distribution():
    """测试类别分布"""
    print("测试类别分布...")
    
    loader = RadioMLDataLoader("dummy.pkl", "2016.10A")
    data, labels, mod_classes, snr_levels = loader.load_data()
    
    # 获取类别分布
    distribution = loader.get_class_distribution()
    
    assert len(distribution) == len(mod_classes), f"分布类别数量错误: {len(distribution)}"
    
    print("  ✓ 类别分布:")
    for class_name, count in list(distribution.items())[:5]:  # 只显示前5个
        print(f"    {class_name}: {count} 样本")
    
    return True

def test_signal_visualization():
    """测试信号可视化"""
    print("测试信号可视化...")
    
    loader = RadioMLDataLoader("dummy.pkl", "2016.10A")
    data, labels, mod_classes, snr_levels = loader.load_data()
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # 为每个调制类型绘制一个样本
    for i, mod_class in enumerate(mod_classes[:6]):
        # 找到该调制类型的样本
        class_idx = mod_classes.index(mod_class)
        mask = labels == class_idx
        if mask.any():
            sample_idx = np.where(mask)[0][0]
            sample = data[sample_idx]
            
            # 绘制I和Q通道
            axes[i].plot(sample[0], label='I', alpha=0.7)
            axes[i].plot(sample[1], label='Q', alpha=0.7)
            axes[i].set_title(f'{mod_class}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('radioml_signal_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ 信号可视化保存为: radioml_signal_samples.png")
    return True

def test_comprehensive_functionality():
    """综合功能测试"""
    print("进行综合功能测试...")
    
    # 测试多个数据集类型
    dataset_types = ["2016.10A", "2016.10B", "2018.01A"]
    results = {}
    
    for dataset_type in dataset_types:
        try:
            loader = RadioMLDataLoader(f"dummy_{dataset_type}.pkl", dataset_type)
            data, labels, mod_classes, snr_levels = loader.load_data()
            
            # 创建数据加载器
            train_loader, val_loader, test_loader = loader.create_dataloaders(
                batch_size=16, test_size=0.15, val_size=0.15
            )
            
            # 测试几个批次
            train_batches = []
            for i, batch in enumerate(train_loader):
                train_batches.append(batch)
                if i >= 2:  # 只测试3个批次
                    break
            
            results[dataset_type] = {
                'data_shape': data.shape,
                'num_classes': len(mod_classes),
                'num_snr_levels': len(snr_levels),
                'train_batches': len(train_batches),
                'batch_shape': train_batches[0][0].shape if train_batches else None
            }
            
            print(f"  ✓ {dataset_type}: {results[dataset_type]}")
            
        except Exception as e:
            print(f"  ✗ {dataset_type}: 错误 - {e}")
            results[dataset_type] = {'error': str(e)}
    
    return results

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("RadioML数据加载器测试套件")
    print("=" * 60)
    
    tests = [
        ("RadioMLDataset类", test_radioml_dataset),
        ("RadioML 2016数据加载器", test_radioml_dataloader_2016),
        ("RadioML 2018数据加载器", test_radioml_dataloader_2018),
        ("数据加载器创建", test_dataloader_creation),
        ("类别分布", test_class_distribution),
        ("信号可视化", test_signal_visualization),
        ("综合功能", test_comprehensive_functionality)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n正在测试: {test_name}")
        print("=" * 50)
        
        try:
            result = test_func()
            if result:
                print(f"{test_name}: ✓ 通过")
                passed += 1
            else:
                print(f"{test_name}: ✗ 失败")
                failed += 1
        except Exception as e:
            print(f"{test_name}: ✗ 错误 - {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"✓ 通过: {passed}")
    print(f"✗ 失败: {failed}")
    print(f"总计: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 所有测试通过! RadioML数据加载器功能正常")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查相关功能")
    
    return passed, failed

if __name__ == "__main__":
    run_all_tests()