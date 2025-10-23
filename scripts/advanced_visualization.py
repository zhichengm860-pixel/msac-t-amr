#!/usr/bin/env python3
"""
高级可视化和分析工具模块
包括混淆矩阵、t-SNE、注意力热图、特征可视化等

作者: Assistant
日期: 2025-01-16
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FeatureExtractor:
    """特征提取器 - 从模型中提取中间特征"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 存储特征的钩子
        self.features = {}
        self.hooks = []
        
    def register_hooks(self, layer_names: List[str] = None):
        """注册钩子函数来提取特征"""
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output.detach().cpu()
            return hook
        
        # 如果没有指定层名，自动注册所有卷积层和线性层
        if layer_names is None:
            layer_names = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                    layer_names.append(name)
        
        # 注册钩子
        for name in layer_names:
            module = dict(self.model.named_modules())[name]
            hook = module.register_forward_hook(hook_fn(name))
            self.hooks.append(hook)
        
        return layer_names
    
    def extract_features(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取特征"""
        self.features.clear()
        data = data.to(self.device)
        
        with torch.no_grad():
            _ = self.model(data)
        
        return self.features.copy()
    
    def remove_hooks(self):
        """移除钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class AttentionVisualizer:
    """注意力可视化器"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def extract_attention_weights(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取注意力权重"""
        attention_weights = {}
        
        def attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'attention_weights'):
                    attention_weights[name] = module.attention_weights.detach().cpu()
            return hook
        
        # 注册注意力层的钩子
        hooks = []
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or hasattr(module, 'attention_weights'):
                hook = module.register_forward_hook(attention_hook(name))
                hooks.append(hook)
        
        # 前向传播
        data = data.to(self.device)
        with torch.no_grad():
            _ = self.model(data)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def visualize_attention_heatmap(self, attention_weights: torch.Tensor, 
                                   sample_idx: int = 0, save_path: str = None) -> plt.Figure:
        """可视化注意力热图"""
        if len(attention_weights.shape) == 4:  # [batch, heads, seq, seq]
            attention = attention_weights[sample_idx].mean(dim=0)  # 平均所有头
        elif len(attention_weights.shape) == 3:  # [batch, seq, seq]
            attention = attention_weights[sample_idx]
        else:
            attention = attention_weights
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建热图
        im = ax.imshow(attention.numpy(), cmap='Blues', aspect='auto')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax)
        
        # 设置标签
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'Attention Heatmap (Sample {sample_idx})')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class AdvancedVisualizer:
    """高级可视化工具"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.feature_extractor = FeatureExtractor(model, device)
        self.attention_visualizer = AttentionVisualizer(model, device)
        
        # 调制类型标签
        self.modulation_labels = [
            'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM',
            'BFSK', 'CPFSK', 'PAM4', 'WBFM', 'AM-SSB', 'AM-DSB'
        ]
    
    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               save_path: str = None) -> plt.Figure:
        """创建混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 绝对数量混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.modulation_labels,
                   yticklabels=self.modulation_labels, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # 百分比混淆矩阵
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=self.modulation_labels,
                   yticklabels=self.modulation_labels, ax=ax2)
        ax2.set_title('Confusion Matrix (Percentages)')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_tsne_visualization(self, features: torch.Tensor, labels: torch.Tensor,
                                 perplexity: int = 30, save_path: str = None) -> plt.Figure:
        """创建t-SNE可视化"""
        # 如果特征是多维的，先展平
        if len(features.shape) > 2:
            features_flat = features.view(features.shape[0], -1)
        else:
            features_flat = features
        
        # 如果特征维度太高，先用PCA降维
        if features_flat.shape[1] > 50:
            pca = PCA(n_components=50)
            features_flat = torch.tensor(pca.fit_transform(features_flat.numpy()))
        
        # 调整perplexity参数，确保小于样本数
        n_samples = features_flat.shape[0]
        adjusted_perplexity = min(perplexity, max(5, n_samples - 1))
        
        # t-SNE降维
        tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=42)
        features_2d = tsne.fit_transform(features_flat.numpy())
        
        # 创建可视化
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 为每个类别使用不同颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.modulation_labels)))
        
        for i, label in enumerate(self.modulation_labels):
            mask = labels.numpy() == i
            if mask.any():
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          c=[colors[i]], label=label, alpha=0.7, s=50)
        
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('t-SNE Visualization of Features')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_feature_distribution_plot(self, features: Dict[str, torch.Tensor],
                                        labels: torch.Tensor, save_path: str = None) -> plt.Figure:
        """创建特征分布图"""
        n_layers = len(features)
        fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(5 * n_layers, 10))
        if n_layers == 1:
            axes = [axes]
        axes = axes.flatten()
        
        for idx, (layer_name, feature_tensor) in enumerate(features.items()):
            if idx >= len(axes):
                break
                
            # 计算每个类别的特征统计
            feature_flat = feature_tensor.view(feature_tensor.shape[0], -1)
            feature_mean = feature_flat.mean(dim=1).numpy()
            
            # 为每个类别创建分布图
            for class_idx in range(len(self.modulation_labels)):
                mask = labels.numpy() == class_idx
                if mask.any():
                    axes[idx].hist(feature_mean[mask], alpha=0.6, 
                                 label=self.modulation_labels[class_idx], bins=20)
            
            axes[idx].set_title(f'Feature Distribution - {layer_name}')
            axes[idx].set_xlabel('Mean Feature Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_signal_visualization(self, signals: torch.Tensor, labels: torch.Tensor,
                                   num_samples: int = 5, save_path: str = None) -> plt.Figure:
        """创建信号可视化"""
        fig, axes = plt.subplots(len(self.modulation_labels), 2, 
                                figsize=(15, 3 * len(self.modulation_labels)))
        
        for class_idx, label in enumerate(self.modulation_labels):
            mask = labels == class_idx
            if mask.any():
                # 选择该类别的样本
                class_signals = signals[mask][:num_samples]
                
                if class_signals.shape[0] > 0:
                    # I通道
                    for i in range(min(num_samples, class_signals.shape[0])):
                        axes[class_idx, 0].plot(class_signals[i, 0, :].numpy(), 
                                              alpha=0.7, linewidth=0.8)
                    axes[class_idx, 0].set_title(f'{label} - I Channel')
                    axes[class_idx, 0].set_xlabel('Sample')
                    axes[class_idx, 0].set_ylabel('Amplitude')
                    axes[class_idx, 0].grid(True, alpha=0.3)
                    
                    # Q通道
                    for i in range(min(num_samples, class_signals.shape[0])):
                        axes[class_idx, 1].plot(class_signals[i, 1, :].numpy(), 
                                              alpha=0.7, linewidth=0.8)
                    axes[class_idx, 1].set_title(f'{label} - Q Channel')
                    axes[class_idx, 1].set_xlabel('Sample')
                    axes[class_idx, 1].set_ylabel('Amplitude')
                    axes[class_idx, 1].grid(True, alpha=0.3)
            else:
                # 如果没有该类别的样本，隐藏子图
                axes[class_idx, 0].set_visible(False)
                axes[class_idx, 1].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_feature_plot(self, features: torch.Tensor, labels: torch.Tensor,
                                       save_path: str = None) -> go.Figure:
        """创建交互式特征图"""
        # 使用PCA降维到3D
        if len(features.shape) > 2:
            features_flat = features.view(features.shape[0], -1)
        else:
            features_flat = features
        
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(features_flat.numpy())
        
        # 创建交互式3D散点图
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1[:len(self.modulation_labels)]
        
        for i, label in enumerate(self.modulation_labels):
            mask = labels.numpy() == i
            if mask.any():
                fig.add_trace(go.Scatter3d(
                    x=features_3d[mask, 0],
                    y=features_3d[mask, 1],
                    z=features_3d[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colors[i % len(colors)],
                        opacity=0.7
                    ),
                    name=label,
                    text=[f'{label}: Sample {j}' for j in range(mask.sum())],
                    hovertemplate='<b>%{text}</b><br>' +
                                 'PC1: %{x:.2f}<br>' +
                                 'PC2: %{y:.2f}<br>' +
                                 'PC3: %{z:.2f}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Interactive 3D Feature Visualization (PCA)',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_model_architecture_plot(self, save_path: str = None) -> plt.Figure:
        """创建模型架构可视化"""
        # 获取模型结构信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 统计不同类型的层
        layer_types = {}
        layer_params = {}
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                module_type = type(module).__name__
                if module_type not in layer_types:
                    layer_types[module_type] = 0
                    layer_params[module_type] = 0
                layer_types[module_type] += 1
                layer_params[module_type] += sum(p.numel() for p in module.parameters())
        
        # 创建可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 层类型分布
        if layer_types:
            ax1.pie(layer_types.values(), labels=layer_types.keys(), autopct='%1.1f%%')
            ax1.set_title('Layer Type Distribution')
        
        # 2. 参数分布
        if layer_params:
            bars = ax2.bar(layer_params.keys(), layer_params.values())
            ax2.set_title('Parameters per Layer Type')
            ax2.set_ylabel('Number of Parameters')
            ax2.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom')
        
        # 3. 模型统计信息
        stats_text = f"""Model Statistics:
        
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Non-trainable Parameters: {total_params - trainable_params:,}

Model Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)

Layer Summary:
{chr(10).join([f'{k}: {v} layers' for k, v in layer_types.items()])}"""
        
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Model Information')
        
        # 4. 参数效率分析
        if layer_params:
            efficiency = {k: v / layer_types[k] for k, v in layer_params.items()}
            bars = ax4.bar(efficiency.keys(), efficiency.values())
            ax4.set_title('Average Parameters per Layer')
            ax4.set_ylabel('Parameters per Layer')
            ax4.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def comprehensive_analysis(self, test_data: torch.Tensor, test_labels: torch.Tensor,
                              save_dir: str = "visualization_results") -> Dict[str, Any]:
        """综合分析和可视化"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 60)
        print("开始综合可视化分析")
        print("=" * 60)
        
        results = {}
        
        # 1. 模型预测
        print("\n1. 进行模型预测...")
        test_data_device = test_data.to(self.device)
        with torch.no_grad():
            outputs = self.model(test_data_device)
            predictions = torch.argmax(outputs, dim=1).cpu()
        
        # 2. 混淆矩阵
        print("2. 创建混淆矩阵...")
        cm_fig = self.create_confusion_matrix(
            test_labels.numpy(), predictions.numpy(),
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        results['confusion_matrix'] = cm_fig
        
        # 3. 提取特征
        print("3. 提取模型特征...")
        layer_names = self.feature_extractor.register_hooks()
        features = self.feature_extractor.extract_features(test_data)
        
        # 选择最后一个特征层进行t-SNE
        if features:
            last_layer = list(features.keys())[-1]
            last_features = features[last_layer]
            
            print("4. 创建t-SNE可视化...")
            tsne_fig = self.create_tsne_visualization(
                last_features, test_labels,
                save_path=os.path.join(save_dir, 'tsne_visualization.png')
            )
            results['tsne'] = tsne_fig
            
            print("5. 创建交互式3D可视化...")
            interactive_fig = self.create_interactive_feature_plot(
                last_features, test_labels,
                save_path=os.path.join(save_dir, 'interactive_features.html')
            )
            results['interactive'] = interactive_fig
        
        # 4. 特征分布
        if features:
            print("6. 创建特征分布图...")
            dist_fig = self.create_feature_distribution_plot(
                features, test_labels,
                save_path=os.path.join(save_dir, 'feature_distributions.png')
            )
            results['feature_distributions'] = dist_fig
        
        # 5. 信号可视化
        print("7. 创建信号可视化...")
        signal_fig = self.create_signal_visualization(
            test_data, test_labels,
            save_path=os.path.join(save_dir, 'signal_visualization.png')
        )
        results['signal_visualization'] = signal_fig
        
        # 6. 模型架构
        print("8. 创建模型架构图...")
        arch_fig = self.create_model_architecture_plot(
            save_path=os.path.join(save_dir, 'model_architecture.png')
        )
        results['model_architecture'] = arch_fig
        
        # 7. 注意力可视化（如果模型有注意力机制）
        print("9. 尝试提取注意力权重...")
        try:
            attention_weights = self.attention_visualizer.extract_attention_weights(test_data[:5])
            if attention_weights:
                for name, weights in attention_weights.items():
                    att_fig = self.attention_visualizer.visualize_attention_heatmap(
                        weights, sample_idx=0,
                        save_path=os.path.join(save_dir, f'attention_{name}.png')
                    )
                    results[f'attention_{name}'] = att_fig
                    print(f"   - 保存注意力图: attention_{name}.png")
            else:
                print("   - 未检测到注意力机制")
        except Exception as e:
            print(f"   - 注意力提取失败: {e}")
        
        # 清理
        self.feature_extractor.remove_hooks()
        
        # 8. 生成分析报告
        print("10. 生成分析报告...")
        self._generate_analysis_report(results, test_labels, predictions, save_dir)
        
        print(f"\n可视化分析完成! 结果保存在: {save_dir}/")
        return results
    
    def _generate_analysis_report(self, results: Dict[str, Any], 
                                 true_labels: torch.Tensor, predictions: torch.Tensor,
                                 save_dir: str):
        """生成分析报告"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        # 计算指标
        accuracy = accuracy_score(true_labels.numpy(), predictions.numpy())
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels.numpy(), predictions.numpy(), average=None, zero_division=0
        )
        
        # 生成报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(save_dir, f"analysis_report_{timestamp}.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 高级可视化分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 模型性能
            f.write("## 模型性能\n\n")
            f.write(f"- **总体准确率**: {accuracy:.4f}\n\n")
            
            # 各类别性能
            f.write("### 各类别详细性能\n\n")
            f.write("| 调制类型 | 精确率 | 召回率 | F1分数 | 支持数 |\n")
            f.write("|----------|--------|--------|--------|--------|\n")
            
            for i, label in enumerate(self.modulation_labels):
                if i < len(precision):
                    f.write(f"| {label} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} | {support[i]} |\n")
            
            f.write("\n")
            
            # 可视化文件
            f.write("## 生成的可视化文件\n\n")
            viz_files = [
                ('confusion_matrix.png', '混淆矩阵'),
                ('tsne_visualization.png', 't-SNE特征可视化'),
                ('interactive_features.html', '交互式3D特征图'),
                ('feature_distributions.png', '特征分布图'),
                ('signal_visualization.png', '信号波形图'),
                ('model_architecture.png', '模型架构图')
            ]
            
            for filename, description in viz_files:
                if os.path.exists(os.path.join(save_dir, filename)):
                    f.write(f"- **{description}**: `{filename}`\n")
            
            f.write("\n")
            
            # 分析建议
            f.write("## 分析建议\n\n")
            f.write("### 模型改进方向\n\n")
            
            if accuracy < 0.8:
                f.write("1. **准确率偏低**: 考虑增加模型复杂度或改进数据预处理\n")
            
            f.write("2. **特征分析**: 查看t-SNE图了解类别分离度\n")
            f.write("3. **混淆矩阵**: 识别容易混淆的调制类型\n")
            f.write("4. **信号分析**: 观察不同调制类型的时域特征\n\n")
            
            f.write("### 下一步工作\n\n")
            f.write("1. 根据混淆矩阵优化难分类别\n")
            f.write("2. 分析特征分布，改进特征提取\n")
            f.write("3. 考虑数据增强策略\n")
            f.write("4. 尝试集成学习方法\n")
        
        print(f"   - 分析报告: {report_file}")

# 测试代码
if __name__ == "__main__":
    # 创建简单测试模型
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
    
    # 创建测试数据
    model = TestModel()
    visualizer = AdvancedVisualizer(model)
    
    # 生成测试数据
    test_data = torch.randn(100, 2, 1024)
    test_labels = torch.randint(0, 11, (100,))
    
    print("开始高级可视化测试...")
    
    # 进行综合分析
    results = visualizer.comprehensive_analysis(test_data, test_labels)
    
    print("\n高级可视化测试完成!")
    print("查看 visualization_results/ 目录获取所有可视化结果")