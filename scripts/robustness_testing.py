#!/usr/bin/env python3
"""
鲁棒性测试模块
测试模型在不同SNR条件、噪声类型和数据集间的泛化能力

作者: Assistant
日期: 2025-01-16
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class NoiseGenerator:
    """噪声生成器 - 生成不同类型的噪声"""
    
    @staticmethod
    def awgn(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
        """添加高斯白噪声"""
        signal_power = torch.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.sqrt(noise_power) * torch.randn_like(signal)
        return signal + noise
    
    @staticmethod
    def impulse_noise(signal: torch.Tensor, probability: float = 0.01, amplitude: float = 5.0) -> torch.Tensor:
        """添加脉冲噪声"""
        noise_mask = torch.rand_like(signal) < probability
        impulse = amplitude * torch.randn_like(signal) * noise_mask
        return signal + impulse
    
    @staticmethod
    def phase_noise(signal: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        """添加相位噪声"""
        if signal.shape[1] == 2:  # I/Q数据
            i_channel = signal[:, 0:1, :]
            q_channel = signal[:, 1:2, :]
            
            # 计算幅度和相位
            magnitude = torch.sqrt(i_channel**2 + q_channel**2)
            phase = torch.atan2(q_channel, i_channel)
            
            # 添加相位噪声
            phase_noise = std * torch.randn_like(phase)
            new_phase = phase + phase_noise
            
            # 转换回I/Q
            new_i = magnitude * torch.cos(new_phase)
            new_q = magnitude * torch.sin(new_phase)
            
            return torch.cat([new_i, new_q], dim=1)
        else:
            return signal
    
    @staticmethod
    def frequency_offset(signal: torch.Tensor, offset_hz: float, sample_rate: float = 1e6) -> torch.Tensor:
        """添加频率偏移"""
        if signal.shape[1] == 2:  # I/Q数据
            batch_size, _, signal_length = signal.shape
            t = torch.arange(signal_length, dtype=torch.float32) / sample_rate
            t = t.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
            
            # 频率偏移
            offset_phase = 2 * np.pi * offset_hz * t
            cos_offset = torch.cos(offset_phase)
            sin_offset = torch.sin(offset_phase)
            
            i_channel = signal[:, 0:1, :]
            q_channel = signal[:, 1:2, :]
            
            # 应用频率偏移
            new_i = i_channel * cos_offset - q_channel * sin_offset
            new_q = i_channel * sin_offset + q_channel * cos_offset
            
            return torch.cat([new_i, new_q], dim=1)
        else:
            return signal

class DatasetVariationGenerator:
    """数据集变化生成器 - 模拟不同数据集的特征"""
    
    @staticmethod
    def amplitude_scaling(signal: torch.Tensor, scale_range: Tuple[float, float] = (0.5, 2.0)) -> torch.Tensor:
        """幅度缩放变化"""
        # 生成随机缩放因子
        scale = torch.rand(signal.shape[0], 1, 1) * (scale_range[1] - scale_range[0]) + scale_range[0]
        return signal * scale
    
    @staticmethod
    def time_shift(signal: torch.Tensor, max_shift: int = 50) -> torch.Tensor:
        """时间偏移变化"""
        batch_size, channels, length = signal.shape
        shifted_signals = torch.zeros_like(signal)
        
        for i in range(batch_size):
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            if shift > 0:
                shifted_signals[i, :, shift:] = signal[i, :, :-shift]
            elif shift < 0:
                shifted_signals[i, :, :shift] = signal[i, :, -shift:]
            else:
                shifted_signals[i] = signal[i]
        
        return shifted_signals
    
    @staticmethod
    def channel_imbalance(signal: torch.Tensor, imbalance_db: float = 1.0) -> torch.Tensor:
        """通道不平衡"""
        if signal.shape[1] == 2:  # I/Q数据
            imbalance_linear = 10 ** (imbalance_db / 20)
            signal[:, 0:1, :] *= imbalance_linear  # I通道
            return signal
        else:
            return signal

class RobustnessEvaluator:
    """鲁棒性评估器"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.noise_generator = NoiseGenerator()
        self.dataset_generator = DatasetVariationGenerator()
        
        # 结果存储
        self.results = {}
        
    def create_test_data(self, batch_size: int = 100, signal_length: int = 1024, 
                        num_classes: int = 11) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建测试数据"""
        # 生成模拟的调制信号
        data = torch.randn(batch_size, 2, signal_length)
        labels = torch.randint(0, num_classes, (batch_size,))
        return data, labels
    
    def evaluate_snr_robustness(self, test_data: torch.Tensor, test_labels: torch.Tensor,
                               snr_range: List[float] = None) -> Dict[str, float]:
        """评估SNR鲁棒性"""
        if snr_range is None:
            snr_range = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
        
        snr_results = {}
        
        with torch.no_grad():
            for snr_db in snr_range:
                # 添加噪声
                noisy_data = self.noise_generator.awgn(test_data, snr_db)
                noisy_data = noisy_data.to(self.device)
                test_labels_device = test_labels.to(self.device)
                
                # 预测
                outputs = self.model(noisy_data)
                predictions = torch.argmax(outputs, dim=1)
                
                # 计算准确率
                accuracy = (predictions == test_labels_device).float().mean().item()
                snr_results[f"SNR_{snr_db}dB"] = accuracy
                
                print(f"SNR {snr_db:2d}dB: 准确率 = {accuracy:.4f}")
        
        return snr_results
    
    def evaluate_noise_robustness(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
        """评估不同噪声类型的鲁棒性"""
        noise_results = {}
        
        # 测试配置
        noise_configs = {
            'clean': lambda x: x,
            'awgn_10db': lambda x: self.noise_generator.awgn(x, 10),
            'awgn_0db': lambda x: self.noise_generator.awgn(x, 0),
            'impulse_1%': lambda x: self.noise_generator.impulse_noise(x, 0.01, 3.0),
            'impulse_5%': lambda x: self.noise_generator.impulse_noise(x, 0.05, 3.0),
            'phase_noise_0.1': lambda x: self.noise_generator.phase_noise(x, 0.1),
            'phase_noise_0.2': lambda x: self.noise_generator.phase_noise(x, 0.2),
            'freq_offset_1k': lambda x: self.noise_generator.frequency_offset(x, 1000),
            'freq_offset_5k': lambda x: self.noise_generator.frequency_offset(x, 5000),
        }
        
        with torch.no_grad():
            for noise_type, noise_func in noise_configs.items():
                # 应用噪声
                noisy_data = noise_func(test_data)
                noisy_data = noisy_data.to(self.device)
                test_labels_device = test_labels.to(self.device)
                
                # 预测
                outputs = self.model(noisy_data)
                predictions = torch.argmax(outputs, dim=1)
                
                # 计算准确率
                accuracy = (predictions == test_labels_device).float().mean().item()
                noise_results[noise_type] = accuracy
                
                print(f"{noise_type:15s}: 准确率 = {accuracy:.4f}")
        
        return noise_results
    
    def evaluate_dataset_robustness(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
        """评估数据集变化的鲁棒性"""
        dataset_results = {}
        
        # 测试配置
        dataset_configs = {
            'original': lambda x: x,
            'amplitude_0.5-2.0': lambda x: self.dataset_generator.amplitude_scaling(x, (0.5, 2.0)),
            'amplitude_0.3-3.0': lambda x: self.dataset_generator.amplitude_scaling(x, (0.3, 3.0)),
            'time_shift_25': lambda x: self.dataset_generator.time_shift(x, 25),
            'time_shift_50': lambda x: self.dataset_generator.time_shift(x, 50),
            'channel_imbalance_1db': lambda x: self.dataset_generator.channel_imbalance(x, 1.0),
            'channel_imbalance_3db': lambda x: self.dataset_generator.channel_imbalance(x, 3.0),
        }
        
        with torch.no_grad():
            for variation_type, variation_func in dataset_configs.items():
                # 应用变化
                varied_data = variation_func(test_data)
                varied_data = varied_data.to(self.device)
                test_labels_device = test_labels.to(self.device)
                
                # 预测
                outputs = self.model(varied_data)
                predictions = torch.argmax(outputs, dim=1)
                
                # 计算准确率
                accuracy = (predictions == test_labels_device).float().mean().item()
                dataset_results[variation_type] = accuracy
                
                print(f"{variation_type:20s}: 准确率 = {accuracy:.4f}")
        
        return dataset_results
    
    def comprehensive_evaluation(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, Any]:
        """综合鲁棒性评估"""
        print("=" * 60)
        print("开始综合鲁棒性评估")
        print("=" * 60)
        
        results = {}
        
        # SNR鲁棒性
        print("\n1. SNR鲁棒性测试:")
        print("-" * 40)
        results['snr_robustness'] = self.evaluate_snr_robustness(test_data, test_labels)
        
        # 噪声鲁棒性
        print("\n2. 噪声类型鲁棒性测试:")
        print("-" * 40)
        results['noise_robustness'] = self.evaluate_noise_robustness(test_data, test_labels)
        
        # 数据集变化鲁棒性
        print("\n3. 数据集变化鲁棒性测试:")
        print("-" * 40)
        results['dataset_robustness'] = self.evaluate_dataset_robustness(test_data, test_labels)
        
        # 计算统计信息
        results['statistics'] = self._calculate_statistics(results)
        
        return results
    
    def _calculate_statistics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """计算统计信息"""
        stats = {}
        
        # SNR统计
        snr_accuracies = list(results['snr_robustness'].values())
        stats['snr_mean'] = np.mean(snr_accuracies)
        stats['snr_std'] = np.std(snr_accuracies)
        stats['snr_min'] = np.min(snr_accuracies)
        stats['snr_max'] = np.max(snr_accuracies)
        
        # 噪声统计
        noise_accuracies = list(results['noise_robustness'].values())
        stats['noise_mean'] = np.mean(noise_accuracies)
        stats['noise_std'] = np.std(noise_accuracies)
        stats['noise_min'] = np.min(noise_accuracies)
        stats['noise_max'] = np.max(noise_accuracies)
        
        # 数据集变化统计
        dataset_accuracies = list(results['dataset_robustness'].values())
        stats['dataset_mean'] = np.mean(dataset_accuracies)
        stats['dataset_std'] = np.std(dataset_accuracies)
        stats['dataset_min'] = np.min(dataset_accuracies)
        stats['dataset_max'] = np.max(dataset_accuracies)
        
        return stats
    
    def visualize_results(self, results: Dict[str, Any], save_dir: str = "robustness_results"):
        """可视化结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. SNR vs 准确率
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        snr_data = results['snr_robustness']
        snr_values = [float(k.split('_')[1].replace('dB', '')) for k in snr_data.keys()]
        accuracies = list(snr_data.values())
        
        plt.plot(snr_values, accuracies, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('SNR (dB)')
        plt.ylabel('准确率')
        plt.title('SNR vs 准确率')
        plt.grid(True, alpha=0.3)
        
        # 2. 噪声类型对比
        plt.subplot(1, 3, 2)
        noise_data = results['noise_robustness']
        noise_types = list(noise_data.keys())
        noise_accuracies = list(noise_data.values())
        
        bars = plt.bar(range(len(noise_types)), noise_accuracies, alpha=0.7)
        plt.xlabel('噪声类型')
        plt.ylabel('准确率')
        plt.title('不同噪声类型的鲁棒性')
        plt.xticks(range(len(noise_types)), noise_types, rotation=45, ha='right')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. 数据集变化对比
        plt.subplot(1, 3, 3)
        dataset_data = results['dataset_robustness']
        dataset_types = list(dataset_data.keys())
        dataset_accuracies = list(dataset_data.values())
        
        bars = plt.bar(range(len(dataset_types)), dataset_accuracies, alpha=0.7, color='green')
        plt.xlabel('数据集变化')
        plt.ylabel('准确率')
        plt.title('数据集变化的鲁棒性')
        plt.xticks(range(len(dataset_types)), dataset_types, rotation=45, ha='right')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'robustness_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 统计热图
        plt.figure(figsize=(10, 6))
        
        # 准备热图数据
        categories = ['SNR', 'Noise', 'Dataset']
        metrics = ['Mean', 'Std', 'Min', 'Max']
        
        heatmap_data = np.array([
            [results['statistics']['snr_mean'], results['statistics']['snr_std'], 
             results['statistics']['snr_min'], results['statistics']['snr_max']],
            [results['statistics']['noise_mean'], results['statistics']['noise_std'], 
             results['statistics']['noise_min'], results['statistics']['noise_max']],
            [results['statistics']['dataset_mean'], results['statistics']['dataset_std'], 
             results['statistics']['dataset_min'], results['statistics']['dataset_max']]
        ])
        
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd',
                   xticklabels=metrics, yticklabels=categories)
        plt.title('鲁棒性统计热图')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'robustness_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n可视化结果已保存到: {save_dir}/")
    
    def save_results(self, results: Dict[str, Any], save_dir: str = "robustness_results"):
        """保存结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(save_dir, f"robustness_results_{timestamp}.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成报告
        report_file = os.path.join(save_dir, f"robustness_report_{timestamp}.md")
        self._generate_report(results, report_file)
        
        print(f"\n结果已保存:")
        print(f"  详细结果: {results_file}")
        print(f"  分析报告: {report_file}")
    
    def _generate_report(self, results: Dict[str, Any], report_file: str):
        """生成分析报告"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 模型鲁棒性测试报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 统计摘要
            f.write("## 统计摘要\n\n")
            stats = results['statistics']
            f.write("| 测试类型 | 平均准确率 | 标准差 | 最小值 | 最大值 |\n")
            f.write("|----------|------------|--------|--------|--------|\n")
            f.write(f"| SNR鲁棒性 | {stats['snr_mean']:.4f} | {stats['snr_std']:.4f} | {stats['snr_min']:.4f} | {stats['snr_max']:.4f} |\n")
            f.write(f"| 噪声鲁棒性 | {stats['noise_mean']:.4f} | {stats['noise_std']:.4f} | {stats['noise_min']:.4f} | {stats['noise_max']:.4f} |\n")
            f.write(f"| 数据集鲁棒性 | {stats['dataset_mean']:.4f} | {stats['dataset_std']:.4f} | {stats['dataset_min']:.4f} | {stats['dataset_max']:.4f} |\n\n")
            
            # SNR详细结果
            f.write("## SNR鲁棒性详细结果\n\n")
            f.write("| SNR (dB) | 准确率 |\n")
            f.write("|----------|--------|\n")
            for snr, acc in results['snr_robustness'].items():
                snr_val = snr.replace('SNR_', '').replace('dB', '')
                f.write(f"| {snr_val} | {acc:.4f} |\n")
            f.write("\n")
            
            # 噪声详细结果
            f.write("## 噪声鲁棒性详细结果\n\n")
            f.write("| 噪声类型 | 准确率 |\n")
            f.write("|----------|--------|\n")
            for noise, acc in results['noise_robustness'].items():
                f.write(f"| {noise} | {acc:.4f} |\n")
            f.write("\n")
            
            # 数据集变化详细结果
            f.write("## 数据集变化鲁棒性详细结果\n\n")
            f.write("| 变化类型 | 准确率 |\n")
            f.write("|----------|--------|\n")
            for variation, acc in results['dataset_robustness'].items():
                f.write(f"| {variation} | {acc:.4f} |\n")
            f.write("\n")
            
            # 分析结论
            f.write("## 分析结论\n\n")
            f.write("### 主要发现\n\n")
            
            # SNR分析
            snr_best = max(results['snr_robustness'], key=results['snr_robustness'].get)
            snr_worst = min(results['snr_robustness'], key=results['snr_robustness'].get)
            f.write(f"- **SNR鲁棒性**: 在{snr_best}条件下表现最佳({results['snr_robustness'][snr_best]:.4f})，在{snr_worst}条件下表现最差({results['snr_robustness'][snr_worst]:.4f})\n")
            
            # 噪声分析
            noise_best = max(results['noise_robustness'], key=results['noise_robustness'].get)
            noise_worst = min(results['noise_robustness'], key=results['noise_robustness'].get)
            f.write(f"- **噪声鲁棒性**: 对{noise_best}最鲁棒({results['noise_robustness'][noise_best]:.4f})，对{noise_worst}最敏感({results['noise_robustness'][noise_worst]:.4f})\n")
            
            # 数据集分析
            dataset_best = max(results['dataset_robustness'], key=results['dataset_robustness'].get)
            dataset_worst = min(results['dataset_robustness'], key=results['dataset_robustness'].get)
            f.write(f"- **数据集鲁棒性**: 对{dataset_best}最鲁棒({results['dataset_robustness'][dataset_best]:.4f})，对{dataset_worst}最敏感({results['dataset_robustness'][dataset_worst]:.4f})\n\n")
            
            f.write("### 改进建议\n\n")
            f.write("1. **数据增强**: 在训练时加入更多噪声和变化类型\n")
            f.write("2. **正则化**: 增强模型的泛化能力\n")
            f.write("3. **集成学习**: 使用多个模型提高鲁棒性\n")
            f.write("4. **对抗训练**: 针对特定噪声类型进行对抗训练\n\n")
            
            f.write("## 可视化图表\n\n")
            f.write("- 鲁棒性分析图: `robustness_analysis.png`\n")
            f.write("- 统计热图: `robustness_heatmap.png`\n")

# 测试代码
if __name__ == "__main__":
    # 创建一个简单的测试模型
    class SimpleTestModel(nn.Module):
        def __init__(self, input_channels=2, num_classes=11):
            super().__init__()
            self.conv1 = nn.Conv1d(input_channels, 32, 7, padding=3)
            self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(64, num_classes)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)
    
    # 创建模型和评估器
    model = SimpleTestModel()
    evaluator = RobustnessEvaluator(model)
    
    # 创建测试数据
    test_data, test_labels = evaluator.create_test_data(batch_size=200)
    
    print("开始鲁棒性测试...")
    
    # 进行综合评估
    results = evaluator.comprehensive_evaluation(test_data, test_labels)
    
    # 可视化和保存结果
    evaluator.visualize_results(results)
    evaluator.save_results(results)
    
    print("\n鲁棒性测试完成!")
    print("结果已保存到 robustness_results/ 目录")