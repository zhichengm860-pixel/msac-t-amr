#!/usr/bin/env python3
"""
鲁棒性测试验证脚本

作者: Assistant
日期: 2025-01-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from robustness_testing import RobustnessEvaluator, NoiseGenerator, DatasetVariationGenerator

def create_test_model():
    """创建测试模型"""
    class TestModel(nn.Module):
        def __init__(self, input_channels=2, num_classes=11):
            super().__init__()
            self.conv1 = nn.Conv1d(input_channels, 32, 7, padding=3)
            self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
            self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, num_classes)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.pool(x).squeeze(-1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)
    
    return TestModel()

def test_noise_generators():
    """测试噪声生成器"""
    print("=" * 60)
    print("测试噪声生成器")
    print("=" * 60)
    
    # 创建测试数据
    test_signal = torch.randn(4, 2, 1024)
    noise_gen = NoiseGenerator()
    
    # 测试AWGN
    print("测试AWGN噪声:")
    awgn_signal = noise_gen.awgn(test_signal, 10)
    print(f"  原始信号形状: {test_signal.shape}")
    print(f"  AWGN信号形状: {awgn_signal.shape}")
    print(f"  信号功率变化: {torch.mean(test_signal**2):.4f} -> {torch.mean(awgn_signal**2):.4f}")
    
    # 测试脉冲噪声
    print("\n测试脉冲噪声:")
    impulse_signal = noise_gen.impulse_noise(test_signal, 0.05, 3.0)
    print(f"  脉冲噪声信号形状: {impulse_signal.shape}")
    print(f"  最大值变化: {torch.max(test_signal):.4f} -> {torch.max(impulse_signal):.4f}")
    
    # 测试相位噪声
    print("\n测试相位噪声:")
    phase_signal = noise_gen.phase_noise(test_signal, 0.1)
    print(f"  相位噪声信号形状: {phase_signal.shape}")
    
    # 测试频率偏移
    print("\n测试频率偏移:")
    freq_signal = noise_gen.frequency_offset(test_signal, 1000)
    print(f"  频率偏移信号形状: {freq_signal.shape}")
    
    print("✓ 噪声生成器测试通过")

def test_dataset_generators():
    """测试数据集变化生成器"""
    print("\n" + "=" * 60)
    print("测试数据集变化生成器")
    print("=" * 60)
    
    # 创建测试数据
    test_signal = torch.randn(4, 2, 1024)
    dataset_gen = DatasetVariationGenerator()
    
    # 测试幅度缩放
    print("测试幅度缩放:")
    scaled_signal = dataset_gen.amplitude_scaling(test_signal, (0.5, 2.0))
    print(f"  原始信号RMS: {torch.sqrt(torch.mean(test_signal**2)):.4f}")
    print(f"  缩放信号RMS: {torch.sqrt(torch.mean(scaled_signal**2)):.4f}")
    
    # 测试时间偏移
    print("\n测试时间偏移:")
    shifted_signal = dataset_gen.time_shift(test_signal, 50)
    print(f"  时间偏移信号形状: {shifted_signal.shape}")
    
    # 测试通道不平衡
    print("\n测试通道不平衡:")
    imbalanced_signal = dataset_gen.channel_imbalance(test_signal, 2.0)
    print(f"  I通道RMS: {torch.sqrt(torch.mean(test_signal[:, 0, :]**2)):.4f} -> {torch.sqrt(torch.mean(imbalanced_signal[:, 0, :]**2)):.4f}")
    print(f"  Q通道RMS: {torch.sqrt(torch.mean(test_signal[:, 1, :]**2)):.4f} -> {torch.sqrt(torch.mean(imbalanced_signal[:, 1, :]**2)):.4f}")
    
    print("✓ 数据集变化生成器测试通过")

def test_robustness_evaluator():
    """测试鲁棒性评估器"""
    print("\n" + "=" * 60)
    print("测试鲁棒性评估器")
    print("=" * 60)
    
    # 创建模型和评估器
    model = create_test_model()
    evaluator = RobustnessEvaluator(model)
    
    # 创建测试数据
    print("创建测试数据...")
    test_data, test_labels = evaluator.create_test_data(batch_size=50)
    print(f"  测试数据形状: {test_data.shape}")
    print(f"  测试标签形状: {test_labels.shape}")
    
    # 测试SNR鲁棒性（简化版本）
    print("\n测试SNR鲁棒性:")
    snr_results = evaluator.evaluate_snr_robustness(test_data, test_labels, [-5, 0, 5, 10])
    print(f"  SNR测试结果: {len(snr_results)} 个SNR点")
    
    # 测试噪声鲁棒性（简化版本）
    print("\n测试噪声鲁棒性:")
    # 创建简化的噪声配置
    evaluator_copy = RobustnessEvaluator(model)
    
    # 手动测试几种噪声
    with torch.no_grad():
        # 清洁数据
        clean_outputs = model(test_data)
        clean_acc = (torch.argmax(clean_outputs, dim=1) == test_labels).float().mean().item()
        print(f"  清洁数据准确率: {clean_acc:.4f}")
        
        # AWGN噪声
        awgn_data = evaluator.noise_generator.awgn(test_data, 10)
        awgn_outputs = model(awgn_data)
        awgn_acc = (torch.argmax(awgn_outputs, dim=1) == test_labels).float().mean().item()
        print(f"  AWGN(10dB)准确率: {awgn_acc:.4f}")
        
        # 脉冲噪声
        impulse_data = evaluator.noise_generator.impulse_noise(test_data, 0.01, 3.0)
        impulse_outputs = model(impulse_data)
        impulse_acc = (torch.argmax(impulse_outputs, dim=1) == test_labels).float().mean().item()
        print(f"  脉冲噪声准确率: {impulse_acc:.4f}")
    
    print("✓ 鲁棒性评估器测试通过")

def test_comprehensive_evaluation():
    """测试综合评估（快速版本）"""
    print("\n" + "=" * 60)
    print("测试综合评估（快速版本）")
    print("=" * 60)
    
    # 创建模型和评估器
    model = create_test_model()
    evaluator = RobustnessEvaluator(model)
    
    # 创建小规模测试数据
    test_data, test_labels = evaluator.create_test_data(batch_size=20)
    
    # 快速SNR测试
    print("快速SNR测试:")
    snr_results = evaluator.evaluate_snr_robustness(test_data, test_labels, [0, 10, 20])
    
    # 快速噪声测试
    print("\n快速噪声测试:")
    noise_results = {}
    with torch.no_grad():
        # 只测试几种主要噪声
        test_configs = {
            'clean': lambda x: x,
            'awgn_10db': lambda x: evaluator.noise_generator.awgn(x, 10),
            'impulse_1%': lambda x: evaluator.noise_generator.impulse_noise(x, 0.01, 3.0),
        }
        
        for noise_type, noise_func in test_configs.items():
            noisy_data = noise_func(test_data)
            outputs = model(noisy_data)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == test_labels).float().mean().item()
            noise_results[noise_type] = accuracy
            print(f"  {noise_type}: {accuracy:.4f}")
    
    # 快速数据集变化测试
    print("\n快速数据集变化测试:")
    dataset_results = {}
    with torch.no_grad():
        test_configs = {
            'original': lambda x: x,
            'amplitude_scaling': lambda x: evaluator.dataset_generator.amplitude_scaling(x, (0.5, 2.0)),
            'time_shift': lambda x: evaluator.dataset_generator.time_shift(x, 25),
        }
        
        for variation_type, variation_func in test_configs.items():
            varied_data = variation_func(test_data)
            outputs = model(varied_data)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == test_labels).float().mean().item()
            dataset_results[variation_type] = accuracy
            print(f"  {variation_type}: {accuracy:.4f}")
    
    # 组合结果
    results = {
        'snr_robustness': snr_results,
        'noise_robustness': noise_results,
        'dataset_robustness': dataset_results
    }
    
    # 计算统计信息
    results['statistics'] = evaluator._calculate_statistics(results)
    
    print("\n统计信息:")
    stats = results['statistics']
    print(f"  SNR平均准确率: {stats['snr_mean']:.4f} ± {stats['snr_std']:.4f}")
    print(f"  噪声平均准确率: {stats['noise_mean']:.4f} ± {stats['noise_std']:.4f}")
    print(f"  数据集平均准确率: {stats['dataset_mean']:.4f} ± {stats['dataset_std']:.4f}")
    
    # 保存结果
    print("\n保存测试结果...")
    evaluator.visualize_results(results, "test_robustness_results")
    evaluator.save_results(results, "test_robustness_results")
    
    print("✓ 综合评估测试通过")

def main():
    """主测试函数"""
    print("🚀 开始鲁棒性测试验证")
    
    try:
        # 测试各个组件
        test_noise_generators()
        test_dataset_generators()
        test_robustness_evaluator()
        test_comprehensive_evaluation()
        
        print("\n" + "=" * 60)
        print("🎉 所有鲁棒性测试验证通过!")
        print("=" * 60)
        print("\n下一步:")
        print("1. 运行完整鲁棒性测试: python robustness_testing.py")
        print("2. 查看生成的分析报告和可视化结果")
        print("3. 根据结果优化模型鲁棒性")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()