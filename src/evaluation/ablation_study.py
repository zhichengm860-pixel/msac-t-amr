"""
ablation_study.py - 消融实验框架
系统性分析MSAC-T模型各组件的贡献：
1. 多尺度特征提取的影响
2. 复数注意力机制的作用
3. SNR自适应门控的效果
4. Transformer编码器的贡献
5. 不同组件组合的性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from collections import defaultdict
import time

from src.models.improved_msac_t import (
    ImprovedComplexConv1d, 
    ImprovedComplexBatchNorm1d,
    ComplexActivation,
    ImprovedMultiScaleComplexCNN,
    ImprovedPhaseAwareAttention,
    ImprovedSNRAdaptiveGating,
    ImprovedComplexTransformerBlock
)


# ==================== 消融模型变体 ====================

class BaselineComplexCNN(nn.Module):
    """基础复数CNN（无多尺度、无注意力、无SNR门控、无Transformer）"""
    def __init__(self, num_classes=11, base_channels=64):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            ImprovedComplexConv1d(1, base_channels, 7, padding=3),
            ImprovedComplexBatchNorm1d(base_channels),
            ComplexActivation('gelu'),
            
            ImprovedComplexConv1d(base_channels, base_channels * 2, 5, padding=2),
            ImprovedComplexBatchNorm1d(base_channels * 2),
            ComplexActivation('gelu'),
            
            ImprovedComplexConv1d(base_channels * 2, base_channels * 4, 3, padding=1),
            ImprovedComplexBatchNorm1d(base_channels * 4),
            ComplexActivation('gelu'),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4 * 2, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base_channels * 2, num_classes)
        )
    
    def forward(self, x, snr=None):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        
        x_real = self.global_pool(x[:, :, 0, :]).squeeze(-1)
        x_imag = self.global_pool(x[:, :, 1, :]).squeeze(-1)
        features = torch.cat([x_real, x_imag], dim=1)
        
        logits = self.classifier(features)
        return {'logits': logits}


class MultiScaleOnlyModel(nn.Module):
    """仅包含多尺度特征提取的模型"""
    def __init__(self, num_classes=11, base_channels=64):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            ImprovedComplexConv1d(1, base_channels, 7, padding=3),
            ImprovedComplexBatchNorm1d(base_channels),
            ComplexActivation('gelu')
        )
        
        self.multiscale1 = ImprovedMultiScaleComplexCNN(base_channels, base_channels * 2)
        self.multiscale2 = ImprovedMultiScaleComplexCNN(base_channels * 2, base_channels * 4)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4 * 2, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base_channels * 2, num_classes)
        )
    
    def forward(self, x, snr=None):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.multiscale1(x)
        x = self.multiscale2(x)
        
        x_real = self.global_pool(x[:, :, 0, :]).squeeze(-1)
        x_imag = self.global_pool(x[:, :, 1, :]).squeeze(-1)
        features = torch.cat([x_real, x_imag], dim=1)
        
        logits = self.classifier(features)
        return {'logits': logits}


class AttentionOnlyModel(nn.Module):
    """仅包含注意力机制的模型"""
    def __init__(self, num_classes=11, base_channels=64):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            ImprovedComplexConv1d(1, base_channels, 7, padding=3),
            ImprovedComplexBatchNorm1d(base_channels),
            ComplexActivation('gelu'),
            
            ImprovedComplexConv1d(base_channels, base_channels * 2, 5, padding=2),
            ImprovedComplexBatchNorm1d(base_channels * 2),
            ComplexActivation('gelu'),
            
            ImprovedComplexConv1d(base_channels * 2, base_channels * 4, 3, padding=1),
            ImprovedComplexBatchNorm1d(base_channels * 4),
            ComplexActivation('gelu'),
        )
        
        self.attention = ImprovedPhaseAwareAttention(base_channels * 4)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4 * 2, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base_channels * 2, num_classes)
        )
    
    def forward(self, x, snr=None):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.attention(x)
        
        x_real = self.global_pool(x[:, :, 0, :]).squeeze(-1)
        x_imag = self.global_pool(x[:, :, 1, :]).squeeze(-1)
        features = torch.cat([x_real, x_imag], dim=1)
        
        logits = self.classifier(features)
        return {'logits': logits}


class SNRGateOnlyModel(nn.Module):
    """仅包含SNR门控的模型"""
    def __init__(self, num_classes=11, base_channels=64):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            ImprovedComplexConv1d(1, base_channels, 7, padding=3),
            ImprovedComplexBatchNorm1d(base_channels),
            ComplexActivation('gelu'),
            
            ImprovedComplexConv1d(base_channels, base_channels * 2, 5, padding=2),
            ImprovedComplexBatchNorm1d(base_channels * 2),
            ComplexActivation('gelu'),
            
            ImprovedComplexConv1d(base_channels * 2, base_channels * 4, 3, padding=1),
            ImprovedComplexBatchNorm1d(base_channels * 4),
            ComplexActivation('gelu'),
        )
        
        self.snr_gate = ImprovedSNRAdaptiveGating(base_channels * 4)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4 * 2, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base_channels * 2, num_classes)
        )
    
    def forward(self, x, snr=None):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        
        if snr is not None:
            x = self.snr_gate(x, snr)
        
        x_real = self.global_pool(x[:, :, 0, :]).squeeze(-1)
        x_imag = self.global_pool(x[:, :, 1, :]).squeeze(-1)
        features = torch.cat([x_real, x_imag], dim=1)
        
        logits = self.classifier(features)
        return {'logits': logits}


class TransformerOnlyModel(nn.Module):
    """仅包含Transformer的模型"""
    def __init__(self, num_classes=11, base_channels=64):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            ImprovedComplexConv1d(1, base_channels, 7, padding=3),
            ImprovedComplexBatchNorm1d(base_channels),
            ComplexActivation('gelu'),
            
            ImprovedComplexConv1d(base_channels, base_channels * 2, 5, padding=2),
            ImprovedComplexBatchNorm1d(base_channels * 2),
            ComplexActivation('gelu'),
            
            ImprovedComplexConv1d(base_channels * 2, base_channels * 4, 3, padding=1),
            ImprovedComplexBatchNorm1d(base_channels * 4),
            ComplexActivation('gelu'),
        )
        
        self.transformer_blocks = nn.ModuleList([
            ImprovedComplexTransformerBlock(base_channels * 4, 8, base_channels * 8, 0.1)
            for _ in range(3)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4 * 2, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base_channels * 2, num_classes)
        )
    
    def forward(self, x, snr=None):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x_real = self.global_pool(x[:, :, 0, :]).squeeze(-1)
        x_imag = self.global_pool(x[:, :, 1, :]).squeeze(-1)
        features = torch.cat([x_real, x_imag], dim=1)
        
        logits = self.classifier(features)
        return {'logits': logits}


# ==================== 消融实验管理器 ====================

class AblationStudyManager:
    """消融实验管理器"""
    
    def __init__(self, train_loader, val_loader, test_loader, device='cuda', num_classes=11):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        
        # 定义所有模型变体
        self.model_variants = {
            'baseline': BaselineComplexCNN,
            'multiscale_only': MultiScaleOnlyModel,
            'attention_only': AttentionOnlyModel,
            'snr_gate_only': SNRGateOnlyModel,
            'transformer_only': TransformerOnlyModel,
        }
        
        # 组合模型
        self.combination_variants = {
            'multiscale_attention': self._create_multiscale_attention_model,
            'multiscale_snr': self._create_multiscale_snr_model,
            'multiscale_transformer': self._create_multiscale_transformer_model,
            'attention_snr': self._create_attention_snr_model,
            'attention_transformer': self._create_attention_transformer_model,
            'snr_transformer': self._create_snr_transformer_model,
            'full_model': self._create_full_model,
        }
        
        self.results = {}
    
    def _create_multiscale_attention_model(self, num_classes, base_channels):
        """多尺度 + 注意力"""
        class MultiScaleAttentionModel(nn.Module):
            def __init__(self, num_classes, base_channels):
                super().__init__()
                self.input_proj = nn.Sequential(
                    ImprovedComplexConv1d(1, base_channels, 7, padding=3),
                    ImprovedComplexBatchNorm1d(base_channels),
                    ComplexActivation('gelu')
                )
                self.multiscale1 = ImprovedMultiScaleComplexCNN(base_channels, base_channels * 2)
                self.multiscale2 = ImprovedMultiScaleComplexCNN(base_channels * 2, base_channels * 4)
                self.attention = ImprovedPhaseAwareAttention(base_channels * 4)
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Sequential(
                    nn.Linear(base_channels * 4 * 2, base_channels * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(base_channels * 2, num_classes)
                )
            
            def forward(self, x, snr=None):
                x = x.unsqueeze(1)
                x = self.input_proj(x)
                x = self.multiscale1(x)
                x = self.multiscale2(x)
                x = self.attention(x)
                
                x_real = self.global_pool(x[:, :, 0, :]).squeeze(-1)
                x_imag = self.global_pool(x[:, :, 1, :]).squeeze(-1)
                features = torch.cat([x_real, x_imag], dim=1)
                
                logits = self.classifier(features)
                return {'logits': logits}
        
        return MultiScaleAttentionModel(num_classes, base_channels)
    
    def _create_multiscale_snr_model(self, num_classes, base_channels):
        """多尺度 + SNR门控"""
        class MultiScaleSNRModel(nn.Module):
            def __init__(self, num_classes, base_channels):
                super().__init__()
                self.input_proj = nn.Sequential(
                    ImprovedComplexConv1d(1, base_channels, 7, padding=3),
                    ImprovedComplexBatchNorm1d(base_channels),
                    ComplexActivation('gelu')
                )
                self.multiscale1 = ImprovedMultiScaleComplexCNN(base_channels, base_channels * 2)
                self.multiscale2 = ImprovedMultiScaleComplexCNN(base_channels * 2, base_channels * 4)
                self.snr_gate = ImprovedSNRAdaptiveGating(base_channels * 4)
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Sequential(
                    nn.Linear(base_channels * 4 * 2, base_channels * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(base_channels * 2, num_classes)
                )
            
            def forward(self, x, snr=None):
                x = x.unsqueeze(1)
                x = self.input_proj(x)
                x = self.multiscale1(x)
                x = self.multiscale2(x)
                
                if snr is not None:
                    x = self.snr_gate(x, snr)
                
                x_real = self.global_pool(x[:, :, 0, :]).squeeze(-1)
                x_imag = self.global_pool(x[:, :, 1, :]).squeeze(-1)
                features = torch.cat([x_real, x_imag], dim=1)
                
                logits = self.classifier(features)
                return {'logits': logits}
        
        return MultiScaleSNRModel(num_classes, base_channels)
    
    def _create_multiscale_transformer_model(self, num_classes, base_channels):
        """多尺度 + Transformer"""
        class MultiScaleTransformerModel(nn.Module):
            def __init__(self, num_classes, base_channels):
                super().__init__()
                self.input_proj = nn.Sequential(
                    ImprovedComplexConv1d(1, base_channels, 7, padding=3),
                    ImprovedComplexBatchNorm1d(base_channels),
                    ComplexActivation('gelu')
                )
                self.multiscale1 = ImprovedMultiScaleComplexCNN(base_channels, base_channels * 2)
                self.multiscale2 = ImprovedMultiScaleComplexCNN(base_channels * 2, base_channels * 4)
                self.transformer_blocks = nn.ModuleList([
                    ImprovedComplexTransformerBlock(base_channels * 4, 8, base_channels * 8, 0.1)
                    for _ in range(3)
                ])
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Sequential(
                    nn.Linear(base_channels * 4 * 2, base_channels * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(base_channels * 2, num_classes)
                )
            
            def forward(self, x, snr=None):
                x = x.unsqueeze(1)
                x = self.input_proj(x)
                x = self.multiscale1(x)
                x = self.multiscale2(x)
                
                for block in self.transformer_blocks:
                    x = block(x)
                
                x_real = self.global_pool(x[:, :, 0, :]).squeeze(-1)
                x_imag = self.global_pool(x[:, :, 1, :]).squeeze(-1)
                features = torch.cat([x_real, x_imag], dim=1)
                
                logits = self.classifier(features)
                return {'logits': logits}
        
        return MultiScaleTransformerModel(num_classes, base_channels)
    
    def _create_attention_snr_model(self, num_classes, base_channels):
        """注意力 + SNR门控"""
        class AttentionSNRModel(nn.Module):
            def __init__(self, num_classes, base_channels):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    ImprovedComplexConv1d(1, base_channels, 7, padding=3),
                    ImprovedComplexBatchNorm1d(base_channels),
                    ComplexActivation('gelu'),
                    ImprovedComplexConv1d(base_channels, base_channels * 2, 5, padding=2),
                    ImprovedComplexBatchNorm1d(base_channels * 2),
                    ComplexActivation('gelu'),
                    ImprovedComplexConv1d(base_channels * 2, base_channels * 4, 3, padding=1),
                    ImprovedComplexBatchNorm1d(base_channels * 4),
                    ComplexActivation('gelu'),
                )
                self.attention = ImprovedPhaseAwareAttention(base_channels * 4)
                self.snr_gate = ImprovedSNRAdaptiveGating(base_channels * 4)
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Sequential(
                    nn.Linear(base_channels * 4 * 2, base_channels * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(base_channels * 2, num_classes)
                )
            
            def forward(self, x, snr=None):
                x = x.unsqueeze(1)
                x = self.conv_layers(x)
                x = self.attention(x)
                
                if snr is not None:
                    x = self.snr_gate(x, snr)
                
                x_real = self.global_pool(x[:, :, 0, :]).squeeze(-1)
                x_imag = self.global_pool(x[:, :, 1, :]).squeeze(-1)
                features = torch.cat([x_real, x_imag], dim=1)
                
                logits = self.classifier(features)
                return {'logits': logits}
        
        return AttentionSNRModel(num_classes, base_channels)
    
    def _create_attention_transformer_model(self, num_classes, base_channels):
        """注意力 + Transformer"""
        class AttentionTransformerModel(nn.Module):
            def __init__(self, num_classes, base_channels):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    ImprovedComplexConv1d(1, base_channels, 7, padding=3),
                    ImprovedComplexBatchNorm1d(base_channels),
                    ComplexActivation('gelu'),
                    ImprovedComplexConv1d(base_channels, base_channels * 2, 5, padding=2),
                    ImprovedComplexBatchNorm1d(base_channels * 2),
                    ComplexActivation('gelu'),
                    ImprovedComplexConv1d(base_channels * 2, base_channels * 4, 3, padding=1),
                    ImprovedComplexBatchNorm1d(base_channels * 4),
                    ComplexActivation('gelu'),
                )
                self.attention = ImprovedPhaseAwareAttention(base_channels * 4)
                self.transformer_blocks = nn.ModuleList([
                    ImprovedComplexTransformerBlock(base_channels * 4, 8, base_channels * 8, 0.1)
                    for _ in range(3)
                ])
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Sequential(
                    nn.Linear(base_channels * 4 * 2, base_channels * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(base_channels * 2, num_classes)
                )
            
            def forward(self, x, snr=None):
                x = x.unsqueeze(1)
                x = self.conv_layers(x)
                x = self.attention(x)
                
                for block in self.transformer_blocks:
                    x = block(x)
                
                x_real = self.global_pool(x[:, :, 0, :]).squeeze(-1)
                x_imag = self.global_pool(x[:, :, 1, :]).squeeze(-1)
                features = torch.cat([x_real, x_imag], dim=1)
                
                logits = self.classifier(features)
                return {'logits': logits}
        
        return AttentionTransformerModel(num_classes, base_channels)
    
    def _create_snr_transformer_model(self, num_classes, base_channels):
        """SNR门控 + Transformer"""
        class SNRTransformerModel(nn.Module):
            def __init__(self, num_classes, base_channels):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    ImprovedComplexConv1d(1, base_channels, 7, padding=3),
                    ImprovedComplexBatchNorm1d(base_channels),
                    ComplexActivation('gelu'),
                    ImprovedComplexConv1d(base_channels, base_channels * 2, 5, padding=2),
                    ImprovedComplexBatchNorm1d(base_channels * 2),
                    ComplexActivation('gelu'),
                    ImprovedComplexConv1d(base_channels * 2, base_channels * 4, 3, padding=1),
                    ImprovedComplexBatchNorm1d(base_channels * 4),
                    ComplexActivation('gelu'),
                )
                self.snr_gate = ImprovedSNRAdaptiveGating(base_channels * 4)
                self.transformer_blocks = nn.ModuleList([
                    ImprovedComplexTransformerBlock(base_channels * 4, 8, base_channels * 8, 0.1)
                    for _ in range(3)
                ])
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Sequential(
                    nn.Linear(base_channels * 4 * 2, base_channels * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(base_channels * 2, num_classes)
                )
            
            def forward(self, x, snr=None):
                x = x.unsqueeze(1)
                x = self.conv_layers(x)
                
                if snr is not None:
                    x = self.snr_gate(x, snr)
                
                for block in self.transformer_blocks:
                    x = block(x)
                
                x_real = self.global_pool(x[:, :, 0, :]).squeeze(-1)
                x_imag = self.global_pool(x[:, :, 1, :]).squeeze(-1)
                features = torch.cat([x_real, x_imag], dim=1)
                
                logits = self.classifier(features)
                return {'logits': logits}
        
        return SNRTransformerModel(num_classes, base_channels)
    
    def _create_full_model(self, num_classes, base_channels):
        """完整模型（所有组件）"""
        from src.models.improved_msac_t import ImprovedMSAC_T
        return ImprovedMSAC_T(num_classes=num_classes, base_channels=base_channels)
    
    def evaluate_model(self, model, data_loader, model_name):
        """评估单个模型"""
        model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f'Evaluating {model_name}'):
                if len(batch) == 3:
                    signals, labels, snr = batch
                    signals = signals.to(self.device)
                    labels = labels.to(self.device)
                    snr = snr.to(self.device)
                else:
                    signals, labels = batch
                    signals = signals.to(self.device)
                    labels = labels.to(self.device)
                    snr = None
                
                outputs = model(signals, snr)
                logits = outputs['logits']
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(data_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train_model_simple(self, model, epochs=50, lr=1e-4):
        """简单训练模型"""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in self.train_loader:
                if len(batch) == 3:
                    signals, labels, snr = batch
                    signals = signals.to(self.device)
                    labels = labels.to(self.device)
                    snr = snr.to(self.device)
                else:
                    signals, labels = batch
                    signals = signals.to(self.device)
                    labels = labels.to(self.device)
                    snr = None
                
                optimizer.zero_grad()
                outputs = model(signals, snr)
                loss = criterion(outputs['logits'], labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(self.train_loader)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def run_ablation_study(self, epochs=50, save_dir='ablation_results'):
        """运行完整的消融实验"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("Starting Ablation Study...")
        print("="*50)
        
        # 测试单个组件
        for variant_name, variant_class in self.model_variants.items():
            print(f"\nTraining and evaluating: {variant_name}")
            
            model = variant_class(self.num_classes).to(self.device)
            
            # 训练
            start_time = time.time()
            self.train_model_simple(model, epochs)
            train_time = time.time() - start_time
            
            # 评估
            val_results = self.evaluate_model(model, self.val_loader, variant_name)
            test_results = self.evaluate_model(model, self.test_loader, variant_name)
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            
            self.results[variant_name] = {
                'val_accuracy': val_results['accuracy'],
                'test_accuracy': test_results['accuracy'],
                'val_loss': val_results['loss'],
                'test_loss': test_results['loss'],
                'parameters': total_params,
                'train_time': train_time
            }
            
            print(f"  Val Acc: {val_results['accuracy']:.4f}")
            print(f"  Test Acc: {test_results['accuracy']:.4f}")
            print(f"  Parameters: {total_params:,}")
        
        # 测试组合模型
        for variant_name, variant_func in self.combination_variants.items():
            print(f"\nTraining and evaluating: {variant_name}")
            
            model = variant_func(self.num_classes, 64).to(self.device)
            
            # 训练
            start_time = time.time()
            self.train_model_simple(model, epochs)
            train_time = time.time() - start_time
            
            # 评估
            val_results = self.evaluate_model(model, self.val_loader, variant_name)
            test_results = self.evaluate_model(model, self.test_loader, variant_name)
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            
            self.results[variant_name] = {
                'val_accuracy': val_results['accuracy'],
                'test_accuracy': test_results['accuracy'],
                'val_loss': val_results['loss'],
                'test_loss': test_results['loss'],
                'parameters': total_params,
                'train_time': train_time
            }
            
            print(f"  Val Acc: {val_results['accuracy']:.4f}")
            print(f"  Test Acc: {test_results['accuracy']:.4f}")
            print(f"  Parameters: {total_params:,}")
        
        # 保存结果
        self.save_results(save_dir)
        
        # 生成报告
        self.generate_report(save_dir)
        
        print("\nAblation study completed!")
        return self.results
    
    def save_results(self, save_dir):
        """保存结果"""
        with open(os.path.join(save_dir, 'ablation_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_report(self, save_dir):
        """生成消融实验报告"""
        # 创建结果表格
        df = pd.DataFrame(self.results).T
        df = df.round(4)
        
        # 保存表格
        df.to_csv(os.path.join(save_dir, 'ablation_results.csv'))
        
        # 绘制结果图表
        self.plot_ablation_results(df, save_dir)
        
        # 生成文本报告
        self.write_text_report(df, save_dir)
    
    def plot_ablation_results(self, df, save_dir):
        """绘制消融实验结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 准确率对比
        test_acc = df['test_accuracy'].sort_values(ascending=True)
        axes[0, 0].barh(range(len(test_acc)), test_acc.values)
        axes[0, 0].set_yticks(range(len(test_acc)))
        axes[0, 0].set_yticklabels(test_acc.index, rotation=0)
        axes[0, 0].set_xlabel('Test Accuracy')
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 参数量对比
        params = df['parameters'] / 1e6  # 转换为百万
        axes[0, 1].bar(range(len(params)), params.values)
        axes[0, 1].set_xticks(range(len(params)))
        axes[0, 1].set_xticklabels(params.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Parameters (M)')
        axes[0, 1].set_title('Model Parameters Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 效率分析（准确率/参数量）
        efficiency = df['test_accuracy'] / (df['parameters'] / 1e6)
        axes[1, 0].bar(range(len(efficiency)), efficiency.values)
        axes[1, 0].set_xticks(range(len(efficiency)))
        axes[1, 0].set_xticklabels(efficiency.index, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Accuracy per Million Parameters')
        axes[1, 0].set_title('Model Efficiency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 训练时间对比
        train_time = df['train_time'] / 60  # 转换为分钟
        axes[1, 1].bar(range(len(train_time)), train_time.values)
        axes[1, 1].set_xticks(range(len(train_time)))
        axes[1, 1].set_xticklabels(train_time.index, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Training Time (minutes)')
        axes[1, 1].set_title('Training Time Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ablation_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def write_text_report(self, df, save_dir):
        """生成文本报告"""
        report = []
        report.append("# Ablation Study Report")
        report.append("=" * 50)
        report.append("")
        
        # 最佳模型
        best_model = df['test_accuracy'].idxmax()
        best_acc = df.loc[best_model, 'test_accuracy']
        report.append(f"## Best Model: {best_model}")
        report.append(f"Test Accuracy: {best_acc:.4f}")
        report.append("")
        
        # 组件贡献分析
        report.append("## Component Contribution Analysis")
        baseline_acc = df.loc['baseline', 'test_accuracy']
        
        component_contributions = {}
        for component in ['multiscale_only', 'attention_only', 'snr_gate_only', 'transformer_only']:
            if component in df.index:
                contribution = df.loc[component, 'test_accuracy'] - baseline_acc
                component_contributions[component] = contribution
                report.append(f"- {component}: +{contribution:.4f}")
        
        report.append("")
        
        # 组合效果分析
        report.append("## Combination Effects")
        for combo in ['multiscale_attention', 'multiscale_snr', 'attention_snr']:
            if combo in df.index:
                combo_acc = df.loc[combo, 'test_accuracy']
                report.append(f"- {combo}: {combo_acc:.4f}")
        
        report.append("")
        
        # 效率分析
        report.append("## Efficiency Analysis")
        efficiency = df['test_accuracy'] / (df['parameters'] / 1e6)
        most_efficient = efficiency.idxmax()
        report.append(f"Most efficient model: {most_efficient}")
        report.append(f"Efficiency score: {efficiency[most_efficient]:.4f} acc/M params")
        
        # 保存报告
        with open(os.path.join(save_dir, 'ablation_report.md'), 'w') as f:
            f.write('\n'.join(report))


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例用法
    print("Ablation Study Framework Created!")
    print("Usage:")
    print("1. Create data loaders")
    print("2. Initialize AblationStudyManager")
    print("3. Run ablation study")
    print("4. Analyze results")