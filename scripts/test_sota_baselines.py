#!/usr/bin/env python3
"""
SOTA基线模型快速测试
"""

import os
import sys
import torch
import time
from typing import Dict, List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sota_baselines import create_sota_model


def test_model_creation():
    """测试模型创建"""
    print("="*60)
    print("测试SOTA基线模型创建")
    print("="*60)
    
    # 测试配置
    config = {
        'input_channels': 2,
        'num_classes': 11,
        'signal_length': 1024
    }
    
    # 要测试的模型
    models_to_test = [
        'resnet18',
        'resnet34',
        'densenet121',
        'densenet169',
        'efficientnet_b0',
        'efficientnet_b1',
        'vit_small',
        'vit_base'
    ]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n测试 {model_name}:")
        try:
            # 创建模型
            model = create_sota_model(model_name, **config)
            
            # 计算参数数量
            num_params = sum(p.numel() for p in model.parameters())
            model_size_mb = num_params * 4 / (1024 * 1024)
            
            print(f"  ✓ 模型创建成功")
            print(f"  ✓ 参数数量: {num_params:,}")
            print(f"  ✓ 模型大小: {model_size_mb:.2f} MB")
            
            results[model_name] = {
                'status': 'success',
                'num_parameters': num_params,
                'model_size_mb': model_size_mb
            }
            
        except Exception as e:
            print(f"  ✗ 模型创建失败: {str(e)}")
            results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results


def test_model_forward():
    """测试模型前向传播"""
    print("\n" + "="*60)
    print("测试模型前向传播")
    print("="*60)
    
    # 测试配置
    config = {
        'input_channels': 2,
        'num_classes': 11,
        'signal_length': 1024
    }
    
    batch_size = 4
    input_tensor = torch.randn(batch_size, config['input_channels'], config['signal_length'])
    
    # 轻量级模型测试（避免内存问题）
    lightweight_models = [
        'resnet18',
        'densenet121', 
        'efficientnet_b0'
    ]
    
    results = {}
    
    for model_name in lightweight_models:
        print(f"\n测试 {model_name} 前向传播:")
        try:
            # 创建模型
            model = create_sota_model(model_name, **config)
            model.eval()
            
            # 前向传播
            with torch.no_grad():
                start_time = time.time()
                output = model(input_tensor)
                end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # ms
            
            print(f"  ✓ 输入形状: {input_tensor.shape}")
            print(f"  ✓ 输出形状: {output.shape}")
            print(f"  ✓ 推理时间: {inference_time:.2f} ms")
            print(f"  ✓ 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            # 验证输出形状
            expected_shape = (batch_size, config['num_classes'])
            if output.shape == expected_shape:
                print(f"  ✓ 输出形状正确")
            else:
                print(f"  ✗ 输出形状错误，期望 {expected_shape}，得到 {output.shape}")
            
            results[model_name] = {
                'status': 'success',
                'input_shape': list(input_tensor.shape),
                'output_shape': list(output.shape),
                'inference_time_ms': inference_time
            }
            
        except Exception as e:
            print(f"  ✗ 前向传播失败: {str(e)}")
            results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results


def test_model_training():
    """测试模型训练（简单测试）"""
    print("\n" + "="*60)
    print("测试模型训练")
    print("="*60)
    
    # 测试配置
    config = {
        'input_channels': 2,
        'num_classes': 11,
        'signal_length': 1024
    }
    
    # 创建简单数据
    batch_size = 8
    num_batches = 3
    
    # 只测试一个轻量级模型
    model_name = 'resnet18'
    
    print(f"\n测试 {model_name} 训练:")
    try:
        # 创建模型
        model = create_sota_model(model_name, **config)
        
        # 创建优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        total_loss = 0.0
        
        for batch_idx in range(num_batches):
            # 生成随机数据
            inputs = torch.randn(batch_size, config['input_channels'], config['signal_length'])
            targets = torch.randint(0, config['num_classes'], (batch_size,))
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print(f"  批次 {batch_idx+1}/{num_batches}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"  ✓ 平均损失: {avg_loss:.4f}")
        print(f"  ✓ 训练测试成功")
        
        return {
            'status': 'success',
            'average_loss': avg_loss,
            'num_batches': num_batches
        }
        
    except Exception as e:
        print(f"  ✗ 训练测试失败: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e)
        }


def test_model_comparison():
    """测试模型对比"""
    print("\n" + "="*60)
    print("模型对比分析")
    print("="*60)
    
    # 测试配置
    config = {
        'input_channels': 2,
        'num_classes': 11,
        'signal_length': 1024
    }
    
    # 轻量级模型对比
    models_to_compare = [
        'resnet18',
        'densenet121',
        'efficientnet_b0'
    ]
    
    comparison_results = []
    
    for model_name in models_to_compare:
        try:
            model = create_sota_model(model_name, **config)
            num_params = sum(p.numel() for p in model.parameters())
            model_size_mb = num_params * 4 / (1024 * 1024)
            
            comparison_results.append({
                'model': model_name,
                'parameters': num_params,
                'size_mb': model_size_mb
            })
            
        except Exception as e:
            print(f"模型 {model_name} 对比失败: {e}")
    
    if comparison_results:
        print("\n模型对比结果:")
        print(f"{'模型':<15} {'参数数量':<12} {'大小(MB)':<10}")
        print("-" * 40)
        
        for result in comparison_results:
            print(f"{result['model']:<15} {result['parameters']:<12,} {result['size_mb']:<10.2f}")
        
        # 找出最小和最大的模型
        min_params = min(comparison_results, key=lambda x: x['parameters'])
        max_params = max(comparison_results, key=lambda x: x['parameters'])
        
        print(f"\n分析:")
        print(f"  最小模型: {min_params['model']} ({min_params['parameters']:,} 参数)")
        print(f"  最大模型: {max_params['model']} ({max_params['parameters']:,} 参数)")
        
        return comparison_results
    
    return []


def main():
    """主函数"""
    print("SOTA基线模型测试")
    print("="*60)
    
    # 1. 测试模型创建
    creation_results = test_model_creation()
    
    # 2. 测试前向传播
    forward_results = test_model_forward()
    
    # 3. 测试训练
    training_results = test_model_training()
    
    # 4. 模型对比
    comparison_results = test_model_comparison()
    
    # 5. 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    # 统计成功的模型
    successful_creations = sum(1 for r in creation_results.values() if r['status'] == 'success')
    total_creations = len(creation_results)
    
    successful_forwards = sum(1 for r in forward_results.values() if r['status'] == 'success')
    total_forwards = len(forward_results)
    
    print(f"模型创建: {successful_creations}/{total_creations} 成功")
    print(f"前向传播: {successful_forwards}/{total_forwards} 成功")
    print(f"训练测试: {'成功' if training_results['status'] == 'success' else '失败'}")
    print(f"模型对比: {len(comparison_results)} 个模型")
    
    if successful_creations > 0 and successful_forwards > 0:
        print("\n🎉 SOTA基线模型测试通过!")
        print("可以进行完整的基准测试")
        print("\n下一步:")
        print("1. 运行完整基准测试: python model_benchmark.py")
        print("2. 查看模型性能对比")
        print("3. 选择最适合的基线模型")
    else:
        print("\n❌ 部分测试失败")
        print("请检查错误信息并修复问题")


if __name__ == "__main__":
    main()