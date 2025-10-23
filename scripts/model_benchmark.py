#!/usr/bin/env python3
"""
模型基准测试和比较模块
系统性地比较不同SOTA模型的性能
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sota_baselines import create_sota_model
from src.models.improved_msac_t import ImprovedMSAC_T


class ModelBenchmark:
    """模型基准测试类"""
    
    def __init__(self, device: str = 'cpu', save_dir: str = 'benchmark_results'):
        self.device = torch.device(device)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 基准测试配置
        self.config = {
            'input_channels': 2,
            'num_classes': 11,
            'signal_length': 1024,
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
        }
        
        # 要测试的模型列表
        self.models_to_test = [
            'resnet18',
            'resnet34', 
            'densenet121',
            'efficientnet_b0',
            'vit_small',
            'improved_msac_t'  # 我们的改进模型
        ]
        
        # 结果存储
        self.results = {}
        
    def create_mock_data(self, num_samples: int = 1000) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建模拟数据"""
        print("创建模拟数据...")
        
        # 生成模拟信号数据
        signals = torch.randn(num_samples, self.config['input_channels'], self.config['signal_length'])
        labels = torch.randint(0, self.config['num_classes'], (num_samples,))
        
        # 数据集分割
        train_size = int(0.7 * num_samples)
        val_size = int(0.15 * num_samples)
        test_size = num_samples - train_size - val_size
        
        # 训练集
        train_dataset = TensorDataset(signals[:train_size], labels[:train_size])
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        # 验证集
        val_dataset = TensorDataset(signals[train_size:train_size+val_size], 
                                   labels[train_size:train_size+val_size])
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # 测试集
        test_dataset = TensorDataset(signals[train_size+val_size:], 
                                    labels[train_size+val_size:])
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        print(f"数据集大小 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def create_model(self, model_name: str) -> nn.Module:
        """创建模型"""
        if model_name == 'improved_msac_t':
            # 创建我们的改进模型
            model = ImprovedMSAC_T(
                input_channels=self.config['input_channels'],
                num_classes=self.config['num_classes'],
                base_channels=32,  # 使用较小的配置以适应CPU
                num_transformer_blocks=2,
                num_heads=4
            )
        else:
            # 创建SOTA基线模型
            model = create_sota_model(
                model_name,
                input_channels=self.config['input_channels'],
                num_classes=self.config['num_classes'],
                signal_length=self.config['signal_length']
            )
        
        return model.to(self.device)
    
    def count_parameters(self, model: nn.Module) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, 
                             num_runs: int = 100) -> float:
        """测量推理时间"""
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # 测量时间
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        return avg_time
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader) -> Dict:
        """训练模型"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), 
                              lr=self.config['learning_rate'],
                              weight_decay=self.config['weight_decay'])
        
        train_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        
        for epoch in range(self.config['num_epochs']):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            val_acc = self.evaluate_model(model, val_loader)
            val_accuracies.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if epoch % 2 == 0:  # 每2个epoch打印一次
                print(f"  Epoch {epoch+1}/{self.config['num_epochs']}: "
                      f"Loss={avg_train_loss:.4f}, Val_Acc={val_acc:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_acc,
            'final_val_accuracy': val_accuracies[-1]
        }
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> float:
        """评估模型"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total
        return accuracy
    
    def benchmark_single_model(self, model_name: str, train_loader: DataLoader,
                              val_loader: DataLoader, test_loader: DataLoader) -> Dict:
        """对单个模型进行基准测试"""
        print(f"\n{'='*60}")
        print(f"基准测试: {model_name}")
        print(f"{'='*60}")
        
        try:
            # 创建模型
            model = self.create_model(model_name)
            
            # 模型信息
            num_params = self.count_parameters(model)
            model_size_mb = num_params * 4 / (1024 * 1024)  # 假设float32
            
            print(f"模型参数数量: {num_params:,}")
            print(f"模型大小: {model_size_mb:.2f} MB")
            
            # 测量推理时间
            sample_input = torch.randn(1, self.config['input_channels'], 
                                     self.config['signal_length']).to(self.device)
            inference_time = self.measure_inference_time(model, sample_input)
            
            print(f"平均推理时间: {inference_time:.2f} ms")
            
            # 训练模型
            print("开始训练...")
            training_results = self.train_model(model, train_loader, val_loader)
            
            # 测试模型
            test_accuracy = self.evaluate_model(model, test_loader)
            print(f"测试准确率: {test_accuracy:.2f}%")
            
            # 汇总结果
            results = {
                'model_name': model_name,
                'num_parameters': num_params,
                'model_size_mb': model_size_mb,
                'inference_time_ms': inference_time,
                'best_val_accuracy': training_results['best_val_accuracy'],
                'final_val_accuracy': training_results['final_val_accuracy'],
                'test_accuracy': test_accuracy,
                'train_losses': training_results['train_losses'],
                'val_accuracies': training_results['val_accuracies'],
                'training_time': self.config['num_epochs'],  # 简化
                'status': 'success'
            }
            
            return results
            
        except Exception as e:
            print(f"❌ 模型 {model_name} 测试失败: {str(e)}")
            return {
                'model_name': model_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_benchmark(self) -> Dict:
        """运行完整基准测试"""
        print("开始模型基准测试")
        print(f"设备: {self.device}")
        print(f"配置: {self.config}")
        
        # 创建数据
        train_loader, val_loader, test_loader = self.create_mock_data()
        
        # 测试所有模型
        all_results = {}
        
        for model_name in self.models_to_test:
            try:
                results = self.benchmark_single_model(
                    model_name, train_loader, val_loader, test_loader
                )
                all_results[model_name] = results
                
            except Exception as e:
                print(f"模型 {model_name} 基准测试失败: {e}")
                all_results[model_name] = {
                    'model_name': model_name,
                    'status': 'failed',
                    'error': str(e)
                }
        
        # 保存结果
        self.results = all_results
        self.save_results()
        
        # 生成报告
        self.generate_report()
        
        return all_results
    
    def save_results(self):
        """保存结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细结果
        results_file = os.path.join(self.save_dir, f'benchmark_results_{timestamp}.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存汇总表格
        summary_data = []
        for model_name, results in self.results.items():
            if results.get('status') == 'success':
                summary_data.append({
                    'Model': model_name,
                    'Parameters': results['num_parameters'],
                    'Size (MB)': results['model_size_mb'],
                    'Inference (ms)': results['inference_time_ms'],
                    'Best Val Acc (%)': results['best_val_accuracy'],
                    'Test Acc (%)': results['test_accuracy']
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_file = os.path.join(self.save_dir, f'benchmark_summary_{timestamp}.csv')
            df.to_csv(summary_file, index=False)
            
            print(f"\n结果已保存:")
            print(f"  详细结果: {results_file}")
            print(f"  汇总表格: {summary_file}")
    
    def generate_report(self):
        """生成基准测试报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建可视化
        self.create_visualizations()
        
        # 生成Markdown报告
        report_file = os.path.join(self.save_dir, f'benchmark_report_{timestamp}.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 模型基准测试报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 测试配置\n\n")
            for key, value in self.config.items():
                f.write(f"- {key}: {value}\n")
            f.write(f"- 设备: {self.device}\n\n")
            
            f.write("## 模型性能对比\n\n")
            
            # 成功的模型
            successful_models = {k: v for k, v in self.results.items() 
                               if v.get('status') == 'success'}
            
            if successful_models:
                f.write("| 模型 | 参数数量 | 模型大小(MB) | 推理时间(ms) | 最佳验证准确率(%) | 测试准确率(%) |\n")
                f.write("|------|----------|--------------|--------------|-------------------|---------------|\n")
                
                for model_name, results in successful_models.items():
                    f.write(f"| {model_name} | {results['num_parameters']:,} | "
                           f"{results['model_size_mb']:.2f} | {results['inference_time_ms']:.2f} | "
                           f"{results['best_val_accuracy']:.2f} | {results['test_accuracy']:.2f} |\n")
                
                f.write("\n## 性能分析\n\n")
                
                # 找出最佳模型
                best_accuracy = max(successful_models.values(), 
                                  key=lambda x: x['test_accuracy'])
                fastest_model = min(successful_models.values(), 
                                  key=lambda x: x['inference_time_ms'])
                smallest_model = min(successful_models.values(), 
                                   key=lambda x: x['num_parameters'])
                
                f.write(f"- **最高准确率**: {best_accuracy['model_name']} ({best_accuracy['test_accuracy']:.2f}%)\n")
                f.write(f"- **最快推理**: {fastest_model['model_name']} ({fastest_model['inference_time_ms']:.2f}ms)\n")
                f.write(f"- **最小模型**: {smallest_model['model_name']} ({smallest_model['num_parameters']:,} 参数)\n\n")
            
            # 失败的模型
            failed_models = {k: v for k, v in self.results.items() 
                           if v.get('status') == 'failed'}
            
            if failed_models:
                f.write("## 失败的模型\n\n")
                for model_name, results in failed_models.items():
                    f.write(f"- **{model_name}**: {results.get('error', '未知错误')}\n")
                f.write("\n")
            
            f.write("## 可视化图表\n\n")
            f.write("- 模型性能对比图: `performance_comparison.png`\n")
            f.write("- 参数数量对比图: `parameter_comparison.png`\n")
            f.write("- 推理时间对比图: `inference_time_comparison.png`\n")
        
        print(f"  基准测试报告: {report_file}")
    
    def create_visualizations(self):
        """创建可视化图表"""
        successful_models = {k: v for k, v in self.results.items() 
                           if v.get('status') == 'success'}
        
        if not successful_models:
            return
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. 性能对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(successful_models.keys())
        test_accs = [successful_models[m]['test_accuracy'] for m in models]
        params = [successful_models[m]['num_parameters'] for m in models]
        inference_times = [successful_models[m]['inference_time_ms'] for m in models]
        model_sizes = [successful_models[m]['model_size_mb'] for m in models]
        
        # 测试准确率
        axes[0, 0].bar(models, test_accs, color='skyblue')
        axes[0, 0].set_title('测试准确率对比')
        axes[0, 0].set_ylabel('准确率 (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 参数数量
        axes[0, 1].bar(models, params, color='lightgreen')
        axes[0, 1].set_title('参数数量对比')
        axes[0, 1].set_ylabel('参数数量')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 推理时间
        axes[1, 0].bar(models, inference_times, color='salmon')
        axes[1, 0].set_title('推理时间对比')
        axes[1, 0].set_ylabel('推理时间 (ms)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 模型大小
        axes[1, 1].bar(models, model_sizes, color='gold')
        axes[1, 1].set_title('模型大小对比')
        axes[1, 1].set_ylabel('模型大小 (MB)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 效率分析图（准确率 vs 参数数量）
        plt.figure(figsize=(10, 6))
        plt.scatter(params, test_accs, s=100, alpha=0.7)
        
        for i, model in enumerate(models):
            plt.annotate(model, (params[i], test_accs[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('参数数量')
        plt.ylabel('测试准确率 (%)')
        plt.title('模型效率分析：准确率 vs 参数数量')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'efficiency_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主函数"""
    print("模型基准测试")
    print("="*60)
    
    # 创建基准测试器
    benchmark = ModelBenchmark(device='cpu')  # 使用CPU以确保兼容性
    
    # 运行基准测试
    results = benchmark.run_benchmark()
    
    # 打印汇总
    print("\n" + "="*60)
    print("基准测试完成")
    print("="*60)
    
    successful_models = {k: v for k, v in results.items() 
                        if v.get('status') == 'success'}
    
    if successful_models:
        print(f"成功测试 {len(successful_models)} 个模型:")
        for model_name in successful_models:
            acc = successful_models[model_name]['test_accuracy']
            params = successful_models[model_name]['num_parameters']
            print(f"  - {model_name}: {acc:.2f}% 准确率, {params:,} 参数")
    
    failed_models = {k: v for k, v in results.items() 
                    if v.get('status') == 'failed'}
    
    if failed_models:
        print(f"\n失败 {len(failed_models)} 个模型:")
        for model_name in failed_models:
            print(f"  - {model_name}: {failed_models[model_name].get('error', '未知错误')}")


if __name__ == "__main__":
    main()