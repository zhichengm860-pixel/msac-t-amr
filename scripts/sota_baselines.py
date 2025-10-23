#!/usr/bin/env python3
"""
SOTA基线模型实现
包含ResNet、DenseNet、EfficientNet等现代架构用于调制识别任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


class ResNet1D(nn.Module):
    """1D ResNet for modulation recognition"""
    
    def __init__(self, input_channels: int = 2, num_classes: int = 11, 
                 layers: List[int] = [2, 2, 2, 2], base_channels: int = 64):
        super(ResNet1D, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        
        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(base_channels, base_channels, layers[0])
        self.layer2 = self._make_layer(base_channels, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, layers[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 8, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """Create a residual layer"""
        layers = []
        
        # First block (may have stride > 1)
        layers.append(ResidualBlock1D(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ResidualBlock1D(nn.Module):
    """1D Residual Block"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class DenseNet1D(nn.Module):
    """1D DenseNet for modulation recognition"""
    
    def __init__(self, input_channels: int = 2, num_classes: int = 11,
                 growth_rate: int = 32, block_config: Tuple[int, ...] = (6, 12, 24, 16),
                 num_init_features: int = 64, bn_size: int = 4, drop_rate: float = 0.0):
        super(DenseNet1D, self).__init__()
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, num_init_features, kernel_size=7, 
                     stride=2, padding=3, bias=False),
            nn.BatchNorm1d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        
        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock1D(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = TransitionLayer1D(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))
        
        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool1d(out, 1)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class DenseBlock1D(nn.Module):
    """1D Dense Block"""
    
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int,
                 growth_rate: int, drop_rate: float):
        super(DenseBlock1D, self).__init__()
        
        for i in range(num_layers):
            layer = DenseLayer1D(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f'denselayer{i+1}', layer)
    
    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DenseLayer1D(nn.Module):
    """1D Dense Layer"""
    
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float):
        super(DenseLayer1D, self).__init__()
        
        self.norm1 = nn.BatchNorm1d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(num_input_features, bn_size * growth_rate,
                              kernel_size=1, stride=1, bias=False)
        
        self.norm2 = nn.BatchNorm1d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(bn_size * growth_rate, growth_rate,
                              kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = float(drop_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.norm1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        
        new_features = self.norm2(new_features)
        new_features = self.relu2(new_features)
        new_features = self.conv2(new_features)
        
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        
        return new_features


class TransitionLayer1D(nn.Module):
    """1D Transition Layer"""
    
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer1D, self).__init__()
        
        self.norm = nn.BatchNorm1d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(num_input_features, num_output_features,
                             kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out


class EfficientNet1D(nn.Module):
    """1D EfficientNet for modulation recognition"""
    
    def __init__(self, input_channels: int = 2, num_classes: int = 11,
                 width_coefficient: float = 1.0, depth_coefficient: float = 1.0,
                 dropout_rate: float = 0.2):
        super(EfficientNet1D, self).__init__()
        
        # Model scaling
        def round_filters(filters: int) -> int:
            """Round number of filters based on width multiplier"""
            multiplier = width_coefficient
            if not multiplier:
                return filters
            divisor = 8
            filters *= multiplier
            new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
            if new_filters < 0.9 * filters:
                new_filters += divisor
            return int(new_filters)
        
        def round_repeats(repeats: int) -> int:
            """Round number of repeats based on depth multiplier"""
            return int(math.ceil(depth_coefficient * repeats))
        
        # Building blocks configuration
        # [expand_ratio, channels, repeats, stride, kernel_size]
        blocks_args = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        
        # Stem
        out_channels = round_filters(32)
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # Building blocks
        self.blocks = nn.ModuleList([])
        in_channels = out_channels
        
        for expand_ratio, channels, repeats, stride, kernel_size in blocks_args:
            out_channels = round_filters(channels)
            repeats = round_repeats(repeats)
            
            for i in range(repeats):
                self.blocks.append(
                    MBConv1D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if i == 0 else 1,
                        kernel_size=kernel_size,
                        se_ratio=0.25
                    )
                )
                in_channels = out_channels
        
        # Head
        final_channels = round_filters(1280)
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(final_channels),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout_rate),
        )
        
        self.classifier = nn.Linear(final_channels, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


class MBConv1D(nn.Module):
    """1D Mobile Inverted Bottleneck Convolution"""
    
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int,
                 stride: int, kernel_size: int, se_ratio: float = 0.25):
        super(MBConv1D, self).__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv1d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()
            expanded_channels = in_channels
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(expanded_channels, expanded_channels, kernel_size=kernel_size,
                     stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm1d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitation1D(expanded_channels, se_channels)
        else:
            self.se = nn.Identity()
        
        # Pointwise convolution
        self.pointwise_conv = nn.Sequential(
            nn.Conv1d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.pointwise_conv(x)
        
        if self.use_residual:
            x = x + residual
        
        return x


class SqueezeExcitation1D(nn.Module):
    """1D Squeeze-and-Excitation module"""
    
    def __init__(self, in_channels: int, se_channels: int):
        super(SqueezeExcitation1D, self).__init__()
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, se_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(se_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class VisionTransformer1D(nn.Module):
    """1D Vision Transformer for modulation recognition"""
    
    def __init__(self, input_channels: int = 2, num_classes: int = 11,
                 signal_length: int = 1024, patch_size: int = 16,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(VisionTransformer1D, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = signal_length // patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv1d(input_channels, embed_dim, 
                                    kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock1D(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)    # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Use class token
        x = self.head(x)
        
        return x


class TransformerBlock1D(nn.Module):
    """1D Transformer Block"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(TransformerBlock1D, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


def create_sota_model(model_name: str, **kwargs) -> nn.Module:
    """Factory function to create SOTA models"""
    
    # 提取通用参数
    input_channels = kwargs.get('input_channels', 2)
    num_classes = kwargs.get('num_classes', 11)
    signal_length = kwargs.get('signal_length', 1024)
    
    # 为不同模型准备参数
    cnn_params = {
        'input_channels': input_channels,
        'num_classes': num_classes
    }
    
    vit_params = {
        'input_channels': input_channels,
        'num_classes': num_classes,
        'signal_length': signal_length
    }
    
    models = {
        'resnet18': lambda: ResNet1D(layers=[2, 2, 2, 2], **cnn_params),
        'resnet34': lambda: ResNet1D(layers=[3, 4, 6, 3], **cnn_params),
        'resnet50': lambda: ResNet1D(layers=[3, 4, 6, 3], **cnn_params),  # Would need bottleneck blocks
        'densenet121': lambda: DenseNet1D(block_config=(6, 12, 24, 16), **cnn_params),
        'densenet169': lambda: DenseNet1D(block_config=(6, 12, 32, 32), **cnn_params),
        'densenet201': lambda: DenseNet1D(block_config=(6, 12, 48, 32), **cnn_params),
        'efficientnet_b0': lambda: EfficientNet1D(width_coefficient=1.0, depth_coefficient=1.0, **cnn_params),
        'efficientnet_b1': lambda: EfficientNet1D(width_coefficient=1.0, depth_coefficient=1.1, **cnn_params),
        'efficientnet_b2': lambda: EfficientNet1D(width_coefficient=1.1, depth_coefficient=1.2, **cnn_params),
        'vit_small': lambda: VisionTransformer1D(embed_dim=384, depth=12, num_heads=6, **vit_params),
        'vit_base': lambda: VisionTransformer1D(embed_dim=768, depth=12, num_heads=12, **vit_params),
        'vit_large': lambda: VisionTransformer1D(embed_dim=1024, depth=24, num_heads=16, **vit_params),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name]()


if __name__ == "__main__":
    # Test all models
    input_channels = 2
    num_classes = 11
    signal_length = 1024
    batch_size = 4
    
    # Create test input
    x = torch.randn(batch_size, input_channels, signal_length)
    
    # Test models
    models_to_test = [
        'resnet18', 'resnet34', 'densenet121', 'efficientnet_b0', 'vit_small'
    ]
    
    for model_name in models_to_test:
        print(f"\n测试 {model_name}:")
        try:
            model = create_sota_model(
                model_name, 
                input_channels=input_channels,
                num_classes=num_classes,
                signal_length=signal_length
            )
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            
            print(f"  ✓ 输入形状: {x.shape}")
            print(f"  ✓ 输出形状: {output.shape}")
            print(f"  ✓ 参数数量: {num_params:,}")
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")