#!/usr/bin/env python3
"""
高级可视化工具测试文件

作者: Assistant
日期: 2025-01-16
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from advanced_visualization import AdvancedVisualizer, FeatureExtractor, AttentionVisualizer

def test_feature_extractor():
    """测试特征提取器"""
    print("=" * 50)
    print("测试特征提取器")
    print("=" * 50)
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(2, 16, 7)
            self.conv2 = nn.Conv1d(16, 32, 5)
            self.fc = nn.Linear(32, 11)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
            return self.fc(x)
    
    model = SimpleModel()
    extractor = FeatureExtractor(model)
    
    # 注册钩子
    layer_names = extractor.register_hooks()
    print(f"注册的层: {layer_names}")
    
    # 测试特征提取
    test_data = torch.randn(5, 2, 100)
    features = extractor.extract_features(test_data)
    
    print(f"提取的特征层数: {len(features)}")
    for name, feature in features.items():
        print(f"  {name}: {feature.shape}")
    
    # 清理
    extractor.remove_hooks()
    print("特征提取器测试完成!")
    return True

def test_attention_visualizer():
    """测试注意力可视化器"""
    print("=" * 50)
    print("测试注意力可视化器")
    print("=" * 50)
    
    # 创建带注意力的模型
    class AttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(2, 32, 7)
            self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
            self.fc = nn.Linear(32, 11)
            
        def forward(self, x):
            x = F.relu(self.conv(x))  # [B, 32, L]
            x = x.transpose(1, 2)     # [B, L, 32]
            
            # 注意力机制
            attn_output, attn_weights = self.attention(x, x, x)
            self.attention_weights = attn_weights  # 保存注意力权重
            
            x = attn_output.mean(dim=1)  # 全局平均池化
            return self.fc(x)
    
    model = AttentionModel()
    visualizer = AttentionVisualizer(model)
    
    # 测试注意力权重提取
    test_data = torch.randn(3, 2, 50)
    attention_weights = visualizer.extract_attention_weights(test_data)
    
    print(f"提取的注意力权重: {len(attention_weights)}")
    for name, weights in attention_weights.items():
        print(f"  {name}: {weights.shape}")
    
    # 测试注意力热图
    if attention_weights:
        for name, weights in attention_weights.items():
            fig = visualizer.visualize_attention_heatmap(weights, sample_idx=0)
            print(f"  生成注意力热图: {name}")
            break
    
    print("注意力可视化器测试完成!")
    return True

def test_advanced_visualizer():
    """测试高级可视化器"""
    print("=" * 50)
    print("测试高级可视化器")
    print("=" * 50)
    
    # 创建测试模型
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
    
    model = TestModel()
    visualizer = AdvancedVisualizer(model)
    
    # 创建测试数据
    num_samples = 50
    test_data = torch.randn(num_samples, 2, 1024)
    test_labels = torch.randint(0, 11, (num_samples,))
    
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试标签形状: {test_labels.shape}")
    
    # 1. 测试混淆矩阵
    print("\\n1. 测试混淆矩阵...")
    with torch.no_grad():
        outputs = model(test_data)
        predictions = torch.argmax(outputs, dim=1)
    
    cm_fig = visualizer.create_confusion_matrix(
        test_labels.numpy(), predictions.numpy()
    )
    print("   混淆矩阵创建成功!")
    
    # 2. 测试特征提取和t-SNE
    print("\\n2. 测试t-SNE可视化...")
    layer_names = visualizer.feature_extractor.register_hooks()
    features = visualizer.feature_extractor.extract_features(test_data)
    
    if features:
        last_layer = list(features.keys())[-1]
        last_features = features[last_layer]
        
        tsne_fig = visualizer.create_tsne_visualization(last_features, test_labels)
        print("   t-SNE可视化创建成功!")
        
        # 3. 测试交互式可视化
        print("\\n3. 测试交互式3D可视化...")
        interactive_fig = visualizer.create_interactive_feature_plot(last_features, test_labels)
        print("   交互式3D可视化创建成功!")
        
        # 4. 测试特征分布
        print("\\n4. 测试特征分布图...")
        dist_fig = visualizer.create_feature_distribution_plot(features, test_labels)
        print("   特征分布图创建成功!")
    
    # 5. 测试信号可视化
    print("\\n5. 测试信号可视化...")
    signal_fig = visualizer.create_signal_visualization(test_data, test_labels)
    print("   信号可视化创建成功!")
    
    # 6. 测试模型架构图
    print("\\n6. 测试模型架构图...")
    arch_fig = visualizer.create_model_architecture_plot()
    print("   模型架构图创建成功!")
    
    # 清理
    visualizer.feature_extractor.remove_hooks()
    
    print("\\n高级可视化器基础测试完成!")
    return True

def test_comprehensive_analysis():
    """测试综合分析功能"""
    print("=" * 50)
    print("测试综合分析功能")
    print("=" * 50)
    
    # 创建测试模型
    class TestModel(nn.Module):
        def __init__(self, input_channels=2, num_classes=11):
            super().__init__()
            self.conv1 = nn.Conv1d(input_channels, 16, 7, padding=3)
            self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(32, num_classes)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)
    
    model = TestModel()
    visualizer = AdvancedVisualizer(model)
    
    # 创建测试数据
    num_samples = 30  # 减少样本数以加快测试
    test_data = torch.randn(num_samples, 2, 512)  # 减少信号长度
    test_labels = torch.randint(0, 11, (num_samples,))
    
    print(f"测试数据: {test_data.shape}")
    print(f"测试标签: {test_labels.shape}")
    
    # 进行综合分析
    save_dir = "test_visualization_results"
    results = visualizer.comprehensive_analysis(test_data, test_labels, save_dir)
    
    print(f"\\n综合分析完成! 生成了 {len(results)} 个可视化结果")
    
    # 检查生成的文件
    if os.path.exists(save_dir):
        files = os.listdir(save_dir)
        print(f"\\n生成的文件 ({len(files)} 个):")
        for file in sorted(files):
            file_path = os.path.join(save_dir, file)
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {file} ({size:.1f} KB)")
    
    return True

def run_all_tests():
    """运行所有测试"""
    print("开始高级可视化工具测试...")
    print("=" * 60)
    
    tests = [
        ("特征提取器", test_feature_extractor),
        ("注意力可视化器", test_attention_visualizer),
        ("高级可视化器", test_advanced_visualizer),
        ("综合分析功能", test_comprehensive_analysis)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\\n正在测试: {test_name}")
            success = test_func()
            results[test_name] = "通过" if success else "失败"
            print(f"{test_name}: {'✓ 通过' if success else '✗ 失败'}")
        except Exception as e:
            results[test_name] = f"错误: {str(e)}"
            print(f"{test_name}: ✗ 错误 - {str(e)}")
    
    # 测试总结
    print("\\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✓" if result == "通过" else "✗"
        print(f"{status} {test_name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "通过")
    total = len(results)
    
    print(f"\\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\\n🎉 所有测试通过! 高级可视化工具功能正常")
    else:
        print(f"\\n⚠️  有 {total - passed} 个测试失败，请检查相关功能")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)