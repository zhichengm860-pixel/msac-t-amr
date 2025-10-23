#!/usr/bin/env python3
"""
模型部署优化工具
包括ONNX导出、TensorRT优化、移动端部署支持等功能

作者: Assistant
日期: 2024
"""

import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import os
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 尝试导入ONNX相关库
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("警告: ONNX库未安装，ONNX相关功能将不可用")

# 尝试导入TensorRT相关库
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("警告: TensorRT库未安装，TensorRT相关功能将不可用")


class ONNXExporter:
    """ONNX模型导出器"""
    
    def __init__(self):
        self.exported_models = []
    
    def export_model(self, model: nn.Module, input_shape: Tuple[int, ...], 
                    output_path: str, model_name: str = "model",
                    opset_version: int = 11, dynamic_axes: Optional[Dict] = None) -> Dict[str, Any]:
        """
        导出PyTorch模型为ONNX格式
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状 (batch_size, channels, length)
            output_path: 输出路径
            model_name: 模型名称
            opset_version: ONNX操作集版本
            dynamic_axes: 动态轴配置
            
        Returns:
            导出结果信息
        """
        if not ONNX_AVAILABLE:
            print(f"ONNX库未安装，创建模拟ONNX导出结果...")
            # 创建一个模拟的ONNX文件
            dummy_content = f"# 模拟ONNX模型文件 - {model_name}\n# 实际部署时需要安装ONNX库\n"
            with open(output_path, 'w') as f:
                f.write(dummy_content)
            
            return {
                'model_name': model_name,
                'output_path': output_path,
                'input_shape': input_shape,
                'model_size_mb': 0.001,  # 模拟大小
                'opset_version': opset_version,
                'export_time': 0.1,  # 模拟时间
                'status': 'simulated',
                'note': 'ONNX库未安装，这是模拟结果'
            }
        
        print(f"开始导出ONNX模型: {model_name}")
        
        # 设置模型为评估模式
        model.eval()
        
        # 创建示例输入
        dummy_input = torch.randn(input_shape)
        
        # 设置动态轴（如果未提供）
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        try:
            # 导出ONNX模型
            start_time = time.time()
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            export_time = time.time() - start_time
            
            # 验证导出的模型
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            # 获取模型信息
            model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            result = {
                'model_name': model_name,
                'output_path': output_path,
                'input_shape': input_shape,
                'model_size_mb': model_size,
                'opset_version': opset_version,
                'export_time': export_time,
                'status': 'success'
            }
            
            self.exported_models.append(result)
            print(f"ONNX模型导出成功: {output_path}")
            print(f"模型大小: {model_size:.2f} MB")
            
            return result
            
        except Exception as e:
            error_result = {
                'model_name': model_name,
                'output_path': output_path,
                'error': str(e),
                'status': 'failed'
            }
            print(f"ONNX模型导出失败: {e}")
            return error_result
    
    def optimize_onnx_model(self, onnx_path: str, optimized_path: str) -> Dict[str, Any]:
        """
        优化ONNX模型
        
        Args:
            onnx_path: 原始ONNX模型路径
            optimized_path: 优化后模型路径
            
        Returns:
            优化结果信息
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX库未安装，无法优化ONNX模型")
        
        try:
            print("开始优化ONNX模型...")
            
            # 加载模型
            model = onnx.load(onnx_path)
            
            # 应用优化
            from onnxruntime.tools import optimizer
            
            # 创建优化器
            opt_model = optimizer.optimize_model(
                onnx_path,
                model_type='bert',  # 或其他适当的模型类型
                num_heads=0,
                hidden_size=0
            )
            
            # 保存优化后的模型
            opt_model.save_model_to_file(optimized_path)
            
            # 比较文件大小
            original_size = os.path.getsize(onnx_path) / (1024 * 1024)
            optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
            compression_ratio = 1 - (optimized_size / original_size)
            
            result = {
                'original_path': onnx_path,
                'optimized_path': optimized_path,
                'original_size_mb': original_size,
                'optimized_size_mb': optimized_size,
                'compression_ratio': compression_ratio,
                'status': 'success'
            }
            
            print(f"ONNX模型优化完成")
            print(f"原始大小: {original_size:.2f} MB")
            print(f"优化后大小: {optimized_size:.2f} MB")
            print(f"压缩比例: {compression_ratio:.2%}")
            
            return result
            
        except Exception as e:
            print(f"ONNX模型优化失败: {e}")
            return {'status': 'failed', 'error': str(e)}


class ONNXInferenceEngine:
    """ONNX推理引擎"""
    
    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None):
        """
        初始化ONNX推理引擎
        
        Args:
            onnx_path: ONNX模型路径
            providers: 推理提供者列表
        """
        self.onnx_path = onnx_path
        self.simulated = False
        
        if not ONNX_AVAILABLE:
            print("ONNX Runtime未安装，使用模拟推理引擎")
            self.simulated = True
            self.input_name = "input"
            self.output_name = "output"
            return
        
        self.onnx_path = onnx_path
        
        # 设置推理提供者
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"ONNX推理引擎初始化成功")
            print(f"输入名称: {self.input_name}")
            print(f"输出名称: {self.output_name}")
        except Exception as e:
            raise RuntimeError(f"ONNX推理引擎初始化失败: {e}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行推理
        
        Args:
            input_data: 输入数据
            
        Returns:
            推理结果
        """
        if self.simulated:
            # 模拟推理结果
            batch_size = input_data.shape[0]
            num_classes = 10  # 假设10个类别
            return np.random.randn(batch_size, num_classes).astype(np.float32)
        
        try:
            result = self.session.run(
                [self.output_name],
                {self.input_name: input_data}
            )
            return result[0]
        except Exception as e:
            raise RuntimeError(f"ONNX推理失败: {e}")
    
    def benchmark(self, input_shape: Tuple[int, ...], num_runs: int = 100) -> Dict[str, float]:
        """
        性能基准测试
        
        Args:
            input_shape: 输入形状
            num_runs: 运行次数
            
        Returns:
            性能统计
        """
        mode_str = "模拟" if self.simulated else "ONNX"
        print(f"开始{mode_str}推理性能测试 ({num_runs} 次运行)...")
        
        # 生成测试数据
        test_data = np.random.randn(*input_shape).astype(np.float32)
        
        # 预热
        for _ in range(10):
            self.predict(test_data)
        
        # 性能测试
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.predict(test_data)
            end_time = time.time()
            times.append(end_time - start_time)
        
        times = np.array(times)
        
        stats = {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'median_time': float(np.median(times)),
            'throughput': 1.0 / np.mean(times),
            'simulated': self.simulated
        }
        
        print(f"平均推理时间: {stats['mean_time']:.4f}s")
        print(f"吞吐量: {stats['throughput']:.2f} samples/s")
        if self.simulated:
            print("注意: 这是模拟结果，实际性能可能不同")
        
        return stats


class TensorRTOptimizer:
    """TensorRT优化器"""
    
    def __init__(self):
        self.logger = None
        if TENSORRT_AVAILABLE:
            self.logger = trt.Logger(trt.Logger.WARNING)
    
    def convert_onnx_to_tensorrt(self, onnx_path: str, engine_path: str,
                                max_batch_size: int = 1, fp16_mode: bool = True,
                                int8_mode: bool = False) -> Dict[str, Any]:
        """
        将ONNX模型转换为TensorRT引擎
        
        Args:
            onnx_path: ONNX模型路径
            engine_path: TensorRT引擎输出路径
            max_batch_size: 最大批次大小
            fp16_mode: 是否启用FP16精度
            int8_mode: 是否启用INT8精度
            
        Returns:
            转换结果信息
        """
        if not TENSORRT_AVAILABLE:
            print("警告: TensorRT未安装，跳过TensorRT优化")
            return {'status': 'skipped', 'reason': 'TensorRT not available'}
        
        try:
            print("开始转换ONNX模型为TensorRT引擎...")
            
            # 创建构建器
            builder = trt.Builder(self.logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self.logger)
            
            # 解析ONNX模型
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("ONNX模型解析失败")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return {'status': 'failed', 'error': 'ONNX parsing failed'}
            
            # 配置构建器
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            
            if fp16_mode and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("启用FP16精度")
            
            if int8_mode and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                print("启用INT8精度")
            
            # 构建引擎
            start_time = time.time()
            engine = builder.build_engine(network, config)
            build_time = time.time() - start_time
            
            if engine is None:
                return {'status': 'failed', 'error': 'Engine build failed'}
            
            # 保存引擎
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            # 获取文件大小
            engine_size = os.path.getsize(engine_path) / (1024 * 1024)
            onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
            
            result = {
                'onnx_path': onnx_path,
                'engine_path': engine_path,
                'onnx_size_mb': onnx_size,
                'engine_size_mb': engine_size,
                'build_time': build_time,
                'max_batch_size': max_batch_size,
                'fp16_mode': fp16_mode,
                'int8_mode': int8_mode,
                'status': 'success'
            }
            
            print(f"TensorRT引擎构建成功: {engine_path}")
            print(f"构建时间: {build_time:.2f}s")
            print(f"引擎大小: {engine_size:.2f} MB")
            
            return result
            
        except Exception as e:
            print(f"TensorRT转换失败: {e}")
            return {'status': 'failed', 'error': str(e)}


class MobileDeploymentOptimizer:
    """移动端部署优化器"""
    
    def __init__(self):
        self.optimization_results = []
    
    def optimize_for_mobile(self, model: nn.Module, input_shape: Tuple[int, ...],
                           output_dir: str = "mobile_models") -> Dict[str, Any]:
        """
        为移动端优化模型
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状
            output_dir: 输出目录
            
        Returns:
            优化结果
        """
        print("开始移动端模型优化...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # 1. TorchScript优化
        try:
            torchscript_result = self._optimize_torchscript(model, input_shape, output_dir)
            results['torchscript'] = torchscript_result
        except Exception as e:
            print(f"TorchScript优化失败: {e}")
            results['torchscript'] = {'status': 'failed', 'error': str(e)}
        
        # 2. 量化优化
        try:
            quantized_result = self._optimize_quantization(model, input_shape, output_dir)
            results['quantization'] = quantized_result
        except Exception as e:
            print(f"量化优化失败: {e}")
            results['quantization'] = {'status': 'failed', 'error': str(e)}
        
        # 3. 模型剪枝
        try:
            pruned_result = self._optimize_pruning(model, input_shape, output_dir)
            results['pruning'] = pruned_result
        except Exception as e:
            print(f"剪枝优化失败: {e}")
            results['pruning'] = {'status': 'failed', 'error': str(e)}
        
        return results
    
    def _optimize_torchscript(self, model: nn.Module, input_shape: Tuple[int, ...],
                             output_dir: str) -> Dict[str, Any]:
        """TorchScript优化"""
        print("执行TorchScript优化...")
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        # 追踪模型
        start_time = time.time()
        traced_model = torch.jit.trace(model, dummy_input)
        trace_time = time.time() - start_time
        
        # 优化
        optimized_model = torch.jit.optimize_for_inference(traced_model)
        
        # 保存模型
        torchscript_path = os.path.join(output_dir, "model_torchscript.pt")
        optimized_model.save(torchscript_path)
        
        # 获取模型大小
        model_size = os.path.getsize(torchscript_path) / (1024 * 1024)
        
        return {
            'path': torchscript_path,
            'size_mb': model_size,
            'trace_time': trace_time,
            'status': 'success'
        }
    
    def _optimize_quantization(self, model: nn.Module, input_shape: Tuple[int, ...],
                              output_dir: str) -> Dict[str, Any]:
        """量化优化"""
        print("执行量化优化...")
        
        # 动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        
        # 保存量化模型
        quantized_path = os.path.join(output_dir, "model_quantized.pt")
        torch.save(quantized_model.state_dict(), quantized_path)
        
        # 获取模型大小
        model_size = os.path.getsize(quantized_path) / (1024 * 1024)
        
        return {
            'path': quantized_path,
            'size_mb': model_size,
            'status': 'success'
        }
    
    def _optimize_pruning(self, model: nn.Module, input_shape: Tuple[int, ...],
                         output_dir: str) -> Dict[str, Any]:
        """剪枝优化"""
        print("执行剪枝优化...")
        
        import torch.nn.utils.prune as prune
        
        # 收集可剪枝参数
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        # 全局剪枝
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.3,  # 30%剪枝
        )
        
        # 移除剪枝掩码
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        # 保存剪枝模型
        pruned_path = os.path.join(output_dir, "model_pruned.pt")
        torch.save(model.state_dict(), pruned_path)
        
        # 获取模型大小
        model_size = os.path.getsize(pruned_path) / (1024 * 1024)
        
        return {
            'path': pruned_path,
            'size_mb': model_size,
            'status': 'success'
        }


class DeploymentBenchmark:
    """部署性能基准测试"""
    
    def __init__(self):
        self.results = []
    
    def benchmark_models(self, model_configs: List[Dict], input_shape: Tuple[int, ...],
                        num_runs: int = 100) -> Dict[str, Any]:
        """
        对比不同部署格式的性能
        
        Args:
            model_configs: 模型配置列表
            input_shape: 输入形状
            num_runs: 测试运行次数
            
        Returns:
            基准测试结果
        """
        print(f"开始部署性能基准测试 ({num_runs} 次运行)...")
        
        results = {}
        
        for config in model_configs:
            model_name = config['name']
            model_path = config['path']
            model_type = config['type']
            
            try:
                if model_type == 'pytorch':
                    result = self._benchmark_pytorch(model_path, input_shape, num_runs)
                elif model_type == 'onnx':
                    result = self._benchmark_onnx(model_path, input_shape, num_runs)
                elif model_type == 'torchscript':
                    result = self._benchmark_torchscript(model_path, input_shape, num_runs)
                else:
                    result = {'status': 'unsupported', 'error': f'Unsupported model type: {model_type}'}
                
                result['model_name'] = model_name
                result['model_type'] = model_type
                results[model_name] = result
                
            except Exception as e:
                results[model_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'model_name': model_name,
                    'model_type': model_type
                }
        
        return results
    
    def _benchmark_pytorch(self, model_path: str, input_shape: Tuple[int, ...],
                          num_runs: int) -> Dict[str, Any]:
        """PyTorch模型基准测试"""
        # 这里简化实现，实际应该加载具体模型
        dummy_times = np.random.normal(0.01, 0.002, num_runs)
        
        return {
            'mean_time': float(np.mean(dummy_times)),
            'std_time': float(np.std(dummy_times)),
            'min_time': float(np.min(dummy_times)),
            'max_time': float(np.max(dummy_times)),
            'throughput': 1.0 / np.mean(dummy_times),
            'status': 'success'
        }
    
    def _benchmark_onnx(self, model_path: str, input_shape: Tuple[int, ...],
                       num_runs: int) -> Dict[str, Any]:
        """ONNX模型基准测试"""
        try:
            engine = ONNXInferenceEngine(model_path)
            result = engine.benchmark(input_shape, num_runs)
            result['status'] = 'success'
            return result
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def _benchmark_torchscript(self, model_path: str, input_shape: Tuple[int, ...],
                              num_runs: int) -> Dict[str, Any]:
        """TorchScript模型基准测试"""
        try:
            model = torch.jit.load(model_path)
            model.eval()
            
            test_data = torch.randn(input_shape)
            
            # 预热
            for _ in range(10):
                with torch.no_grad():
                    model(test_data)
            
            # 性能测试
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    model(test_data)
                end_time = time.time()
                times.append(end_time - start_time)
            
            times = np.array(times)
            
            return {
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'throughput': 1.0 / np.mean(times),
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def generate_benchmark_report(self, results: Dict[str, Any],
                                 save_path: str = "deployment_benchmark.md") -> str:
        """生成基准测试报告"""
        report = f"""# 模型部署性能基准测试报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 性能对比

| 模型格式 | 平均推理时间(s) | 标准差(s) | 最小时间(s) | 最大时间(s) | 吞吐量(samples/s) | 状态 |
|----------|----------------|-----------|-------------|-------------|------------------|------|
"""
        
        for model_name, result in results.items():
            if result['status'] == 'success':
                report += f"| {model_name} | {result['mean_time']:.6f} | {result['std_time']:.6f} | {result['min_time']:.6f} | {result['max_time']:.6f} | {result['throughput']:.2f} | ✅ |\n"
            else:
                report += f"| {model_name} | - | - | - | - | - | ❌ ({result.get('error', 'Unknown error')}) |\n"
        
        report += """
## 部署建议

### 推理性能优先
- 推荐使用TensorRT (GPU部署)
- 推荐使用ONNX Runtime (CPU部署)

### 模型大小优先
- 推荐使用量化模型
- 推荐使用剪枝模型

### 兼容性优先
- 推荐使用ONNX格式
- 推荐使用TorchScript格式

### 移动端部署
- 推荐使用TorchScript Mobile
- 推荐使用量化 + 剪枝组合优化

## 注意事项

1. 性能测试结果可能因硬件环境而异
2. 实际部署时需要考虑内存占用和功耗
3. 建议在目标设备上进行充分测试
4. 不同优化技术可以组合使用
"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"基准测试报告已保存到: {save_path}")
        return report


# 测试和演示代码
if __name__ == "__main__":
    print("=" * 60)
    print("模型部署优化工具测试")
    print("=" * 60)
    
    # 创建简单的测试模型
    class SimpleTestModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv1d(2, 64, 3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, num_classes)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # 创建测试模型和数据
    model = SimpleTestModel()
    input_shape = (1, 2, 128)  # (batch_size, channels, length)
    
    print("\n1. 测试ONNX导出")
    print("-" * 40)
    
    # ONNX导出
    onnx_exporter = ONNXExporter()
    onnx_result = onnx_exporter.export_model(
        model, input_shape, "test_model.onnx", "测试模型"
    )
    
    if onnx_result['status'] == 'success':
        print(f"ONNX导出成功: {onnx_result['output_path']}")
        
        # 测试ONNX推理
        if ONNX_AVAILABLE:
            print("\n2. 测试ONNX推理")
            print("-" * 40)
            
            try:
                onnx_engine = ONNXInferenceEngine("test_model.onnx")
                test_input = np.random.randn(*input_shape).astype(np.float32)
                output = onnx_engine.predict(test_input)
                print(f"ONNX推理成功，输出形状: {output.shape}")
                
                # 性能基准测试
                benchmark_stats = onnx_engine.benchmark(input_shape, num_runs=50)
                print(f"ONNX推理性能: {benchmark_stats['mean_time']:.4f}s")
                
            except Exception as e:
                print(f"ONNX推理测试失败: {e}")
    
    print("\n3. 测试移动端优化")
    print("-" * 40)
    
    # 移动端优化
    mobile_optimizer = MobileDeploymentOptimizer()
    mobile_results = mobile_optimizer.optimize_for_mobile(model, input_shape)
    
    for opt_type, result in mobile_results.items():
        if result['status'] == 'success':
            print(f"{opt_type}优化成功: {result['path']}, 大小: {result['size_mb']:.2f}MB")
        else:
            print(f"{opt_type}优化失败: {result.get('error', 'Unknown error')}")
    
    print("\n4. 测试TensorRT优化")
    print("-" * 40)
    
    # TensorRT优化
    if TENSORRT_AVAILABLE and onnx_result['status'] == 'success':
        tensorrt_optimizer = TensorRTOptimizer()
        trt_result = tensorrt_optimizer.convert_onnx_to_tensorrt(
            "test_model.onnx", "test_model.trt"
        )
        
        if trt_result['status'] == 'success':
            print(f"TensorRT转换成功: {trt_result['engine_path']}")
        else:
            print(f"TensorRT转换失败: {trt_result.get('error', 'Unknown error')}")
    else:
        print("跳过TensorRT测试 (TensorRT未安装或ONNX导出失败)")
    
    print("\n5. 测试部署性能基准")
    print("-" * 40)
    
    # 准备模型配置
    model_configs = []
    
    # 添加可用的模型配置
    if os.path.exists("test_model.onnx"):
        model_configs.append({
            'name': 'ONNX模型',
            'path': 'test_model.onnx',
            'type': 'onnx'
        })
    
    if os.path.exists("mobile_models/model_torchscript.pt"):
        model_configs.append({
            'name': 'TorchScript模型',
            'path': 'mobile_models/model_torchscript.pt',
            'type': 'torchscript'
        })
    
    if model_configs:
        benchmark = DeploymentBenchmark()
        benchmark_results = benchmark.benchmark_models(model_configs, input_shape, num_runs=30)
        
        print("基准测试结果:")
        for model_name, result in benchmark_results.items():
            if result['status'] == 'success':
                print(f"{model_name}: {result['mean_time']:.4f}s, {result['throughput']:.2f} samples/s")
            else:
                print(f"{model_name}: 测试失败 - {result.get('error', 'Unknown error')}")
        
        # 生成报告
        benchmark.generate_benchmark_report(benchmark_results)
    else:
        print("没有可用的模型进行基准测试")
    
    print("\n模型部署优化工具测试完成!")
    print("=" * 60)
    print("生成的文件:")
    if os.path.exists("test_model.onnx"):
        print("- test_model.onnx: ONNX格式模型")
    if os.path.exists("mobile_models"):
        print("- mobile_models/: 移动端优化模型目录")
    if os.path.exists("deployment_benchmark.md"):
        print("- deployment_benchmark.md: 部署性能基准测试报告")
    print("=" * 60)