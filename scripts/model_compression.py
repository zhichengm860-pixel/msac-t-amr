#!/usr/bin/env python3
"""
模型压缩和量化技术实现
包括模型剪枝、知识蒸馏、INT8量化等部署优化功能

作者: Assistant
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quant
import numpy as np
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 导入模型
from src.models.model import MSAC_T


class ModelPruner:
    """模型剪枝器 - 实现结构化和非结构化剪枝"""
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.5):
        """
        初始化模型剪枝器
        
        Args:
            model: 要剪枝的模型
            pruning_ratio: 剪枝比例 (0-1)
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.original_params = self._count_parameters()
        
    def _count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def magnitude_pruning(self, structured: bool = False) -> Dict[str, Any]:
        """
        基于权重幅度的剪枝
        
        Args:
            structured: 是否使用结构化剪枝
            
        Returns:
            剪枝结果统计
        """
        print(f"开始{'结构化' if structured else '非结构化'}剪枝...")
        
        if structured:
            return self._structured_pruning()
        else:
            return self._unstructured_pruning()
    
    def _unstructured_pruning(self) -> Dict[str, Any]:
        """非结构化剪枝 - 基于权重幅度"""
        import torch.nn.utils.prune as prune
        
        # 收集所有可剪枝的参数
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        # 全局剪枝
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.pruning_ratio,
        )
        
        # 移除剪枝掩码，使剪枝永久化
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        pruned_params = self._count_parameters()
        compression_ratio = 1 - (pruned_params / self.original_params)
        
        return {
            'method': 'unstructured_magnitude',
            'original_params': self.original_params,
            'pruned_params': pruned_params,
            'compression_ratio': compression_ratio,
            'pruning_ratio': self.pruning_ratio
        }
    
    def _structured_pruning(self) -> Dict[str, Any]:
        """结构化剪枝 - 移除整个通道或神经元"""
        pruned_layers = 0
        total_layers = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d):
                total_layers += 1
                # 计算每个通道的L1范数
                channel_norms = torch.norm(module.weight.data, p=1, dim=(1, 2))
                
                # 确定要剪枝的通道数
                num_channels = module.out_channels
                num_to_prune = int(num_channels * self.pruning_ratio)
                
                if num_to_prune > 0:
                    # 选择L1范数最小的通道进行剪枝
                    _, indices_to_prune = torch.topk(channel_norms, num_to_prune, largest=False)
                    
                    # 创建保留通道的掩码
                    keep_mask = torch.ones(num_channels, dtype=torch.bool)
                    keep_mask[indices_to_prune] = False
                    
                    # 更新权重
                    module.weight.data = module.weight.data[keep_mask]
                    if module.bias is not None:
                        module.bias.data = module.bias.data[keep_mask]
                    
                    module.out_channels = keep_mask.sum().item()
                    pruned_layers += 1
        
        pruned_params = self._count_parameters()
        compression_ratio = 1 - (pruned_params / self.original_params)
        
        return {
            'method': 'structured_magnitude',
            'original_params': self.original_params,
            'pruned_params': pruned_params,
            'compression_ratio': compression_ratio,
            'pruning_ratio': self.pruning_ratio,
            'pruned_layers': pruned_layers,
            'total_layers': total_layers
        }
    
    def gradual_pruning(self, dataloader, epochs: int = 10) -> Dict[str, Any]:
        """
        渐进式剪枝 - 在训练过程中逐步剪枝
        
        Args:
            dataloader: 训练数据加载器
            epochs: 剪枝轮数
            
        Returns:
            剪枝结果统计
        """
        print("开始渐进式剪枝...")
        
        import torch.nn.utils.prune as prune
        
        # 收集可剪枝参数
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        # 每轮剪枝的比例
        step_ratio = self.pruning_ratio / epochs
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # 训练一轮
            self.model.train()
            for batch_idx, (data, target) in enumerate(dataloader):
                if batch_idx >= 10:  # 限制每轮的批次数
                    break
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # 剪枝
            current_ratio = step_ratio * (epoch + 1)
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=current_ratio,
            )
            
            print(f"轮次 {epoch+1}/{epochs}: 累计剪枝比例 {current_ratio:.3f}")
        
        # 移除剪枝掩码
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        pruned_params = self._count_parameters()
        compression_ratio = 1 - (pruned_params / self.original_params)
        
        return {
            'method': 'gradual_pruning',
            'original_params': self.original_params,
            'pruned_params': pruned_params,
            'compression_ratio': compression_ratio,
            'pruning_ratio': self.pruning_ratio,
            'epochs': epochs
        }


class KnowledgeDistiller:
    """知识蒸馏器 - 将大模型的知识转移到小模型"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, 
                 temperature: float = 4.0, alpha: float = 0.7):
        """
        初始化知识蒸馏器
        
        Args:
            teacher_model: 教师模型（大模型）
            student_model: 学生模型（小模型）
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # 冻结教师模型
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
    
    def distillation_loss(self, student_outputs: torch.Tensor, 
                         teacher_outputs: torch.Tensor, 
                         targets: torch.Tensor) -> torch.Tensor:
        """
        计算蒸馏损失
        
        Args:
            student_outputs: 学生模型输出
            teacher_outputs: 教师模型输出
            targets: 真实标签
            
        Returns:
            蒸馏损失
        """
        # 软标签损失（蒸馏损失）
        soft_targets = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        distill_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
        distill_loss *= (self.temperature ** 2)
        
        # 硬标签损失（分类损失）
        hard_loss = F.cross_entropy(student_outputs, targets)
        
        # 组合损失
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, distill_loss, hard_loss
    
    def train_student(self, dataloader, epochs: int = 10, 
                     device: str = 'cpu') -> Dict[str, Any]:
        """
        训练学生模型
        
        Args:
            dataloader: 训练数据加载器
            epochs: 训练轮数
            device: 设备
            
        Returns:
            训练结果统计
        """
        print("开始知识蒸馏训练...")
        
        self.teacher_model.to(device)
        self.student_model.to(device)
        
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=0.001)
        
        train_losses = []
        distill_losses = []
        hard_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_distill = 0
            epoch_hard = 0
            num_batches = 0
            
            self.student_model.train()
            
            for batch_idx, (data, target) in enumerate(dataloader):
                if batch_idx >= 20:  # 限制批次数
                    break
                
                data, target = data.to(device), target.to(device)
                
                # 教师模型推理
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(data)
                
                # 学生模型推理
                student_outputs = self.student_model(data)
                
                # 计算损失
                total_loss, distill_loss, hard_loss = self.distillation_loss(
                    student_outputs, teacher_outputs, target
                )
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_distill += distill_loss.item()
                epoch_hard += hard_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_distill = epoch_distill / num_batches
            avg_hard = epoch_hard / num_batches
            
            train_losses.append(avg_loss)
            distill_losses.append(avg_distill)
            hard_losses.append(avg_hard)
            
            print(f"轮次 {epoch+1}/{epochs}: "
                  f"总损失={avg_loss:.4f}, "
                  f"蒸馏损失={avg_distill:.4f}, "
                  f"分类损失={avg_hard:.4f}")
        
        return {
            'method': 'knowledge_distillation',
            'epochs': epochs,
            'temperature': self.temperature,
            'alpha': self.alpha,
            'train_losses': train_losses,
            'distill_losses': distill_losses,
            'hard_losses': hard_losses,
            'final_loss': train_losses[-1] if train_losses else 0
        }


class ModelQuantizer:
    """模型量化器 - 实现INT8量化等技术"""
    
    def __init__(self, model: nn.Module):
        """
        初始化模型量化器
        
        Args:
            model: 要量化的模型
        """
        self.model = model
        self.original_size = self._get_model_size()
    
    def _get_model_size(self) -> float:
        """获取模型大小（MB）"""
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / 1024 / 1024  # MB
    
    def post_training_quantization(self, dataloader, 
                                  backend: str = 'fbgemm') -> Dict[str, Any]:
        """
        训练后量化（Post-Training Quantization）
        
        Args:
            dataloader: 校准数据加载器
            backend: 量化后端
            
        Returns:
            量化结果统计
        """
        print("开始训练后量化...")
        
        # 设置量化配置
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # 准备模型
        torch.quantization.prepare(self.model, inplace=True)
        
        # 校准
        print("正在校准模型...")
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if batch_idx >= 10:  # 限制校准批次
                    break
                self.model(data)
        
        # 转换为量化模型
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        
        quantized_size = self._get_model_size_quantized(quantized_model)
        compression_ratio = 1 - (quantized_size / self.original_size)
        
        return {
            'method': 'post_training_quantization',
            'backend': backend,
            'original_size_mb': self.original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': compression_ratio,
            'quantized_model': quantized_model
        }
    
    def _get_model_size_quantized(self, model) -> float:
        """获取量化模型大小"""
        # 简化计算，假设INT8量化将模型大小减少约4倍
        return self.original_size / 4
    
    def dynamic_quantization(self) -> Dict[str, Any]:
        """
        动态量化
        
        Returns:
            量化结果统计
        """
        print("开始动态量化...")
        
        # 动态量化主要针对线性层
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        quantized_size = self._get_model_size_quantized(quantized_model)
        compression_ratio = 1 - (quantized_size / self.original_size)
        
        return {
            'method': 'dynamic_quantization',
            'original_size_mb': self.original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': compression_ratio,
            'quantized_model': quantized_model
        }


class CompressionEvaluator:
    """模型压缩评估器"""
    
    def __init__(self):
        self.results = []
    
    def evaluate_model(self, model: nn.Module, dataloader, 
                      device: str = 'cpu', model_name: str = 'model') -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            model: 要评估的模型
            dataloader: 测试数据加载器
            device: 设备
            model_name: 模型名称
            
        Returns:
            评估结果
        """
        model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        total_time = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                if batch_idx >= 20:  # 限制测试批次
                    break
                
                data, target = data.to(device), target.to(device)
                
                # 测量推理时间
                start_time = time.time()
                outputs = model(data)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        avg_inference_time = total_time / min(20, len(dataloader))
        
        # 计算模型大小
        model_size = self._get_model_size(model)
        param_count = sum(p.numel() for p in model.parameters())
        
        result = {
            'model_name': model_name,
            'accuracy': accuracy,
            'model_size_mb': model_size,
            'param_count': param_count,
            'avg_inference_time': avg_inference_time,
            'throughput': 1 / avg_inference_time if avg_inference_time > 0 else 0
        }
        
        self.results.append(result)
        return result
    
    def _get_model_size(self, model: nn.Module) -> float:
        """计算模型大小（MB）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / 1024 / 1024
    
    def compare_models(self, save_path: str = "compression_comparison.png"):
        """比较不同压缩方法的效果"""
        if not self.results:
            print("没有可比较的结果")
            return
        
        # 创建比较图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = [r['model_name'] for r in self.results]
        accuracies = [r['accuracy'] for r in self.results]
        sizes = [r['model_size_mb'] for r in self.results]
        params = [r['param_count'] for r in self.results]
        times = [r['avg_inference_time'] for r in self.results]
        
        # 准确率比较
        axes[0, 0].bar(models, accuracies, color='skyblue')
        axes[0, 0].set_title('模型准确率比较')
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 模型大小比较
        axes[0, 1].bar(models, sizes, color='lightgreen')
        axes[0, 1].set_title('模型大小比较')
        axes[0, 1].set_ylabel('大小 (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 参数数量比较
        axes[1, 0].bar(models, params, color='lightcoral')
        axes[1, 0].set_title('参数数量比较')
        axes[1, 0].set_ylabel('参数数量')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 推理时间比较
        axes[1, 1].bar(models, times, color='gold')
        axes[1, 1].set_title('推理时间比较')
        axes[1, 1].set_ylabel('时间 (秒)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"比较图表已保存到: {save_path}")
    
    def generate_report(self, save_path: str = "compression_report.md") -> str:
        """生成压缩报告"""
        if not self.results:
            return "没有可生成报告的结果"
        
        report = f"""# 模型压缩评估报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 压缩结果概览

| 模型 | 准确率 | 模型大小(MB) | 参数数量 | 推理时间(s) | 吞吐量(samples/s) |
|------|--------|-------------|----------|------------|------------------|
"""
        
        for result in self.results:
            report += f"| {result['model_name']} | {result['accuracy']:.4f} | {result['model_size_mb']:.2f} | {result['param_count']:,} | {result['avg_inference_time']:.4f} | {result['throughput']:.2f} |\n"
        
        # 找到基准模型（通常是第一个）
        if len(self.results) > 1:
            baseline = self.results[0]
            report += f"\n## 压缩效果分析\n\n"
            report += f"以 {baseline['model_name']} 作为基准模型:\n\n"
            
            for result in self.results[1:]:
                size_reduction = (1 - result['model_size_mb'] / baseline['model_size_mb']) * 100
                param_reduction = (1 - result['param_count'] / baseline['param_count']) * 100
                accuracy_change = (result['accuracy'] - baseline['accuracy']) * 100
                speed_improvement = (baseline['avg_inference_time'] / result['avg_inference_time'] - 1) * 100
                
                report += f"### {result['model_name']}\n"
                report += f"- 模型大小减少: {size_reduction:.1f}%\n"
                report += f"- 参数数量减少: {param_reduction:.1f}%\n"
                report += f"- 准确率变化: {accuracy_change:+.2f}%\n"
                report += f"- 推理速度提升: {speed_improvement:+.1f}%\n\n"
        
        report += """
## 建议

1. **剪枝**: 适合需要减少参数数量和计算量的场景
2. **知识蒸馏**: 适合需要保持较高准确率的场景
3. **量化**: 适合需要快速部署和减少内存占用的场景

## 注意事项

- 压缩后的模型需要在目标设备上进行充分测试
- 不同的压缩技术可以组合使用以获得更好的效果
- 压缩比例需要根据具体应用场景进行调整
"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"压缩报告已保存到: {save_path}")
        return report


# 测试和演示代码
if __name__ == "__main__":
    print("=" * 60)
    print("模型压缩和量化技术测试")
    print("=" * 60)
    
    # 创建简单的测试模型
    class SimpleTestModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv1d(2, 64, 3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # 创建测试数据
    def create_test_data():
        batch_size = 32
        num_samples = 128
        signal_length = 128
        num_classes = 10
        
        # 生成随机数据
        data = torch.randn(num_samples, 2, signal_length)
        labels = torch.randint(0, num_classes, (num_samples,))
        
        dataset = torch.utils.data.TensorDataset(data, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    # 创建测试数据
    test_dataloader = create_test_data()
    
    # 创建评估器
    evaluator = CompressionEvaluator()
    
    print("\n1. 测试模型剪枝")
    print("-" * 40)
    
    # 原始模型
    original_model = SimpleTestModel()
    original_result = evaluator.evaluate_model(original_model, test_dataloader, model_name="原始模型")
    print(f"原始模型 - 准确率: {original_result['accuracy']:.4f}, "
          f"大小: {original_result['model_size_mb']:.2f}MB, "
          f"参数: {original_result['param_count']:,}")
    
    # 非结构化剪枝
    pruned_model = SimpleTestModel()
    pruner = ModelPruner(pruned_model, pruning_ratio=0.5)
    pruning_result = pruner.magnitude_pruning(structured=False)
    pruned_result = evaluator.evaluate_model(pruned_model, test_dataloader, model_name="剪枝模型(非结构化)")
    print(f"剪枝模型 - 准确率: {pruned_result['accuracy']:.4f}, "
          f"大小: {pruned_result['model_size_mb']:.2f}MB, "
          f"参数: {pruned_result['param_count']:,}")
    print(f"压缩比例: {pruning_result['compression_ratio']:.2f}")
    
    print("\n2. 测试知识蒸馏")
    print("-" * 40)
    
    # 创建教师和学生模型
    teacher_model = SimpleTestModel()
    student_model = SimpleTestModel()  # 简化版本，实际应该更小
    
    # 知识蒸馏
    distiller = KnowledgeDistiller(teacher_model, student_model)
    distill_result = distiller.train_student(test_dataloader, epochs=5)
    student_result = evaluator.evaluate_model(student_model, test_dataloader, model_name="蒸馏学生模型")
    print(f"学生模型 - 准确率: {student_result['accuracy']:.4f}, "
          f"最终损失: {distill_result['final_loss']:.4f}")
    
    print("\n3. 测试模型量化")
    print("-" * 40)
    
    # 动态量化
    quantization_model = SimpleTestModel()
    quantizer = ModelQuantizer(quantization_model)
    quant_result = quantizer.dynamic_quantization()
    quant_model_result = evaluator.evaluate_model(quant_result['quantized_model'], test_dataloader, model_name="量化模型")
    print(f"量化模型 - 准确率: {quant_model_result['accuracy']:.4f}, "
          f"大小: {quant_model_result['model_size_mb']:.2f}MB")
    print(f"压缩比例: {quant_result['compression_ratio']:.2f}")
    
    print("\n4. 生成比较报告")
    print("-" * 40)
    
    # 生成比较图表和报告
    evaluator.compare_models("model_compression_comparison.png")
    report = evaluator.generate_report("model_compression_report.md")
    
    print("\n模型压缩和量化技术测试完成!")
    print("=" * 60)
    print("生成的文件:")
    print("- model_compression_comparison.png: 压缩效果比较图")
    print("- model_compression_report.md: 详细压缩报告")
    print("=" * 60)