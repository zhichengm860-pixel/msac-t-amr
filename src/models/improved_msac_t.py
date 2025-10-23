"""
improved_msac_t.py - 改进的MSAC-T模型
修复性能问题，优化模型架构
主要改进：
1. 修复复数卷积的数学错误
2. 改进注意力机制
3. 优化网络深度和宽度
4. 添加残差连接和正则化
5. 改进特征融合策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ==================== 改进的复数神经网络层 ====================

class ImprovedComplexConv1d(nn.Module):
    """改进的复数域1D卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        # 复数卷积的正确实现
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for conv in [self.conv_r, self.conv_i]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
    
    def forward(self, x):
        # x shape: [batch, channels, 2, length] where dim=2 is [real, imag]
        real = x[:, :, 0, :]
        imag = x[:, :, 1, :]
        
        # 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        out_real = self.conv_r(real) - self.conv_i(imag)
        out_imag = self.conv_r(imag) + self.conv_i(real)
        
        return torch.stack([out_real, out_imag], dim=2)


class ImprovedComplexBatchNorm1d(nn.Module):
    """改进的复数域批归一化"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn_r = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        self.bn_i = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
    
    def forward(self, x):
        real = self.bn_r(x[:, :, 0, :])
        imag = self.bn_i(x[:, :, 1, :])
        return torch.stack([real, imag], dim=2)


class ComplexActivation(nn.Module):
    """改进的复数激活函数"""
    def __init__(self, activation='relu'):
        super().__init__()
        self.activation = activation
    
    def forward(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'gelu':
            return F.gelu(x)
        elif self.activation == 'swish':
            return x * torch.sigmoid(x)
        else:
            return F.relu(x)


# ==================== 改进的多尺度特征提取 ====================

class ImprovedMultiScaleComplexCNN(nn.Module):
    """改进的复数值多尺度CNN模块"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 9]):
        super().__init__()
        self.branches = nn.ModuleList()
        
        # 每个分支的输出通道数
        branch_channels = out_channels // len(kernel_sizes)
        
        for i, k in enumerate(kernel_sizes):
            # 最后一个分支处理剩余的通道
            if i == len(kernel_sizes) - 1:
                curr_channels = out_channels - branch_channels * i
            else:
                curr_channels = branch_channels
                
            branch = nn.Sequential(
                ImprovedComplexConv1d(in_channels, curr_channels, k, padding=k//2),
                ImprovedComplexBatchNorm1d(curr_channels),
                ComplexActivation('gelu'),
                nn.Dropout(0.1)
            )
            self.branches.append(branch)
        
        # 特征融合
        self.fusion = nn.Sequential(
            ImprovedComplexConv1d(out_channels, out_channels, 1),
            ImprovedComplexBatchNorm1d(out_channels),
            ComplexActivation('gelu')
        )
        
        # 残差连接的投影层
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                ImprovedComplexConv1d(in_channels, out_channels, 1),
                ImprovedComplexBatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        # 多尺度特征提取
        outputs = [branch(x) for branch in self.branches]
        fused = torch.cat(outputs, dim=1)
        
        # 特征融合
        out = self.fusion(fused)
        
        # 残差连接
        out = out + self.shortcut(x)
        
        return out


# ==================== 改进的相位感知注意力 ====================

class ImprovedPhaseAwareAttention(nn.Module):
    """改进的相位感知注意力机制"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        
        # 幅度注意力
        self.magnitude_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
        
        # 相位注意力
        self.phase_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        real = x[:, :, 0, :]
        imag = x[:, :, 1, :]
        
        # 计算幅度和相位
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase = torch.atan2(imag, real + 1e-8)
        
        # 通道注意力
        mag_attn = self.magnitude_attn(magnitude)
        phase_attn = self.phase_attn(phase)
        
        # 空间注意力
        spatial_input = torch.stack([
            torch.mean(magnitude, dim=1, keepdim=True),
            torch.max(magnitude, dim=1, keepdim=True)[0]
        ], dim=1).squeeze(2)
        spatial_attn = self.spatial_attn(spatial_input)
        
        # 应用注意力权重
        weighted_mag = magnitude * mag_attn * spatial_attn
        weighted_phase = phase * phase_attn
        
        # 重构复数信号
        weighted_real = weighted_mag * torch.cos(weighted_phase)
        weighted_imag = weighted_mag * torch.sin(weighted_phase)
        
        return torch.stack([weighted_real, weighted_imag], dim=2)


# ==================== 改进的SNR自适应门控 ====================

class ImprovedSNRAdaptiveGating(nn.Module):
    """改进的SNR自适应门控"""
    def __init__(self, channels, snr_range=(-20, 30), snr_bins=50):
        super().__init__()
        self.snr_range = snr_range
        self.snr_bins = snr_bins
        
        # SNR嵌入
        self.snr_embedding = nn.Embedding(snr_bins, channels // 4)
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(channels // 4, channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )
        
        # 自适应权重
        self.adaptive_weight = nn.Parameter(torch.ones(1))
    
    def _snr_to_index(self, snr):
        """将SNR值转换为索引"""
        snr_min, snr_max = self.snr_range
        normalized_snr = (snr - snr_min) / (snr_max - snr_min)
        indices = (normalized_snr * (self.snr_bins - 1)).long()
        return torch.clamp(indices, 0, self.snr_bins - 1)
    
    def forward(self, x, snr):
        if snr is None:
            return x
            
        # 处理SNR维度
        if snr.dim() == 2:
            snr = snr.squeeze(-1)
        
        # 获取SNR嵌入
        snr_idx = self._snr_to_index(snr)
        snr_emb = self.snr_embedding(snr_idx)
        
        # 计算门控权重
        gate_weights = self.gate_net(snr_emb)
        
        # 调整权重形状
        if x.dim() == 4:  # [batch, channels, 2, length]
            gate_weights = gate_weights.unsqueeze(2).unsqueeze(3)
        elif x.dim() == 3:  # [batch, channels, length]
            gate_weights = gate_weights.unsqueeze(2)
        
        # 应用自适应门控
        return x * (1 + self.adaptive_weight * gate_weights)


# ==================== 改进的Transformer ====================

class ImprovedComplexMultiHeadAttention(nn.Module):
    """改进的复数域多头自注意力"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.W_q = ImprovedComplexConv1d(d_model, d_model, 1)
        self.W_k = ImprovedComplexConv1d(d_model, d_model, 1)
        self.W_v = ImprovedComplexConv1d(d_model, d_model, 1)
        self.W_o = ImprovedComplexConv1d(d_model, d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x):
        batch_size, channels, _, seq_len = x.size()
        
        # 计算Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 重塑为多头
        Q = Q.view(batch_size, self.num_heads, self.d_k, 2, seq_len)
        K = K.view(batch_size, self.num_heads, self.d_k, 2, seq_len)
        V = V.view(batch_size, self.num_heads, self.d_k, 2, seq_len)
        
        # 计算注意力分数（使用幅度）
        Q_mag = torch.sqrt(Q[:, :, :, 0]**2 + Q[:, :, :, 1]**2 + 1e-8)
        K_mag = torch.sqrt(K[:, :, :, 0]**2 + K[:, :, :, 1]**2 + 1e-8)
        
        scores = torch.matmul(Q_mag.transpose(-2, -1), K_mag) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力到V
        V_real = torch.matmul(attn, V[:, :, :, 0].transpose(-2, -1)).transpose(-2, -1)
        V_imag = torch.matmul(attn, V[:, :, :, 1].transpose(-2, -1)).transpose(-2, -1)
        
        # 重塑回原始形状
        output_real = V_real.contiguous().view(batch_size, channels, seq_len)
        output_imag = V_imag.contiguous().view(batch_size, channels, seq_len)
        output = torch.stack([output_real, output_imag], dim=2)
        
        return self.W_o(output)


class ImprovedComplexTransformerBlock(nn.Module):
    """改进的复数Transformer编码器块"""
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = ImprovedComplexMultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = ImprovedComplexBatchNorm1d(d_model)
        
        # 前馈网络
        self.ff = nn.Sequential(
            ImprovedComplexConv1d(d_model, ff_dim, 1),
            ComplexActivation('gelu'),
            nn.Dropout(dropout),
            ImprovedComplexConv1d(ff_dim, d_model, 1)
        )
        self.norm2 = ImprovedComplexBatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 自注意力 + 残差连接
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络 + 残差连接
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


# ==================== 改进的MSAC-T主模型 ====================

class ImprovedMSAC_T(nn.Module):
    """改进的多尺度自适应复数Transformer模型"""
    def __init__(self, 
                 num_classes=11,
                 input_channels=1,
                 base_channels=64,
                 num_transformer_blocks=6,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            ImprovedComplexConv1d(input_channels, base_channels, 7, padding=3),
            ImprovedComplexBatchNorm1d(base_channels),
            ComplexActivation('gelu'),
            nn.Dropout(dropout)
        )
        
        # 多尺度特征提取
        self.multiscale1 = ImprovedMultiScaleComplexCNN(base_channels, base_channels * 2)
        self.multiscale2 = ImprovedMultiScaleComplexCNN(base_channels * 2, base_channels * 4)
        self.multiscale3 = ImprovedMultiScaleComplexCNN(base_channels * 4, base_channels * 4)
        
        # 相位感知注意力
        self.phase_attn = ImprovedPhaseAwareAttention(base_channels * 4)
        
        # SNR自适应门控
        self.snr_gate = ImprovedSNRAdaptiveGating(base_channels * 4)
        
        # Transformer编码器
        self.transformer_blocks = nn.ModuleList([
            ImprovedComplexTransformerBlock(
                base_channels * 4, 
                num_heads, 
                base_channels * 8, 
                dropout
            )
            for _ in range(num_transformer_blocks)
        ])
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
        feature_dim = base_channels * 4 * 2  # *2 for real+imag
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, num_classes)
        )
        
        # 辅助任务
        self.snr_estimator = nn.Sequential(
            nn.Linear(feature_dim, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 2, 1)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, snr=None, return_features=False):
        # 输入形状: [batch, 2, length]
        x = x.unsqueeze(1)  # [batch, 1, 2, length]
        
        # 特征提取
        x = self.input_proj(x)
        x = self.multiscale1(x)
        x = self.multiscale2(x)
        x = self.multiscale3(x)
        
        # 相位感知注意力
        x = self.phase_attn(x)
        
        # SNR自适应门控
        if snr is not None:
            x = self.snr_gate(x, snr)
        
        # Transformer编码
        for block in self.transformer_blocks:
            x = block(x)
        
        # 全局池化
        x_real = self.global_pool(x[:, :, 0, :]).squeeze(-1)
        x_imag = self.global_pool(x[:, :, 1, :]).squeeze(-1)
        features = torch.cat([x_real, x_imag], dim=1)
        
        # 分类
        logits = self.classifier(features)
        
        # 辅助任务
        snr_pred = self.snr_estimator(features).squeeze(-1)
        
        outputs = {
            'logits': logits,
            'snr_pred': snr_pred
        }
        
        if return_features:
            outputs['features'] = features
        
        return outputs


# ==================== 模型工厂函数 ====================

def create_improved_msac_t(num_classes=11, **kwargs):
    """创建改进的MSAC-T模型"""
    return ImprovedMSAC_T(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # 测试模型
    model = create_improved_msac_t(num_classes=11)
    x = torch.randn(4, 2, 1024)
    snr = torch.randn(4) * 20
    
    with torch.no_grad():
        outputs = model(x, snr)
        print(f"Logits shape: {outputs['logits'].shape}")
        print(f"SNR pred shape: {outputs['snr_pred'].shape}")
        
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")