"""
baselines.py - 对比基线模型
包含：
1. ResNet基线模型
2. CLDNN (Convolutional, LSTM, Deep Neural Network)
3. MCformer (Multi-scale Complex Transformer)
4. 传统机器学习方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ==================== ResNet基线 ====================

class BasicBlock(nn.Module):
    """ResNet基本块"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet1D(nn.Module):
    """1D ResNet for signal classification"""
    
    def __init__(self, num_classes=11, input_channels=2, layers=[2, 2, 2, 2]):
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * BasicBlock.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion),
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, snr=None):
        # x: [batch, 2, length]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ==================== CLDNN ====================

class CLDNN(nn.Module):
    """Convolutional, LSTM, Deep Neural Network"""
    
    def __init__(self, num_classes=11, input_channels=2, 
                 cnn_channels=[64, 128, 256], lstm_hidden=128, lstm_layers=2):
        super().__init__()
        
        # CNN部分
        cnn_layers = []
        in_ch = input_channels
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ])
            in_ch = out_ch
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if lstm_layers > 1 else 0
        )
        
        # DNN部分
        self.dnn = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, snr=None):
        # x: [batch, 2, length]
        
        # CNN
        x = self.cnn(x)  # [batch, channels, length']
        
        # 调整维度用于LSTM
        x = x.permute(0, 2, 1)  # [batch, length', channels]
        
        # LSTM
        x, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        x = x[:, -1, :]  # [batch, lstm_hidden*2]
        
        # DNN
        x = self.dnn(x)
        
        return x


# ==================== MCformer ====================

class ComplexMultiHeadAttention(nn.Module):
    """复数域多头注意力"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性投影并分头
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        self.self_attn = ComplexMultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class MCformer(nn.Module):
    """Multi-scale Complex Transformer"""
    
    def __init__(self, num_classes=11, input_channels=2, 
                 embed_dim=256, num_heads=8, num_layers=6, 
                 ff_dim=1024, dropout=0.1):
        super().__init__()
        
        # 多尺度卷积特征提取
        self.scale1 = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim//4),
            nn.ReLU(inplace=True)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim//4, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim//4),
            nn.ReLU(inplace=True)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim//4, kernel_size=7, padding=3),
            nn.BatchNorm1d(embed_dim//4),
            nn.ReLU(inplace=True)
        )
        
        self.scale4 = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim//4, kernel_size=9, padding=4),
            nn.BatchNorm1d(embed_dim//4),
            nn.ReLU(inplace=True)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.embed_dim = embed_dim
        
    def forward(self, x, snr=None):
        # x: [batch, 2, length]
        
        # 多尺度特征提取
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        s4 = self.scale4(x)
        
        # 拼接多尺度特征
        x = torch.cat([s1, s2, s3, s4], dim=1)  # [batch, embed_dim, length]
        
        # 调整维度用于Transformer
        x = x.permute(0, 2, 1)  # [batch, length, embed_dim]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 全局平均池化
        x = x.mean(dim=1)  # [batch, embed_dim]
        
        # 分类
        x = self.classifier(x)
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==================== 传统机器学习方法 ====================

class TraditionalMLBaseline:
    """传统机器学习基线方法"""
    
    def __init__(self, method='svm', **kwargs):
        """
        Args:
            method: 'svm', 'random_forest', 'naive_bayes'
        """
        self.method = method
        self.scaler = StandardScaler()
        
        if method == 'svm':
            self.model = SVC(kernel='rbf', **kwargs)
        elif method == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, **kwargs)
        elif method == 'naive_bayes':
            from sklearn.naive_bayes import GaussianNB
            self.model = GaussianNB(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def extract_features(self, signals):
        """
        提取手工特征
        signals: [N, 2, length]
        """
        N = signals.shape[0]
        features_list = []
        
        for i in range(N):
            signal = signals[i]  # [2, length]
            real, imag = signal[0], signal[1]
            
            # 幅度和相位
            amplitude = np.sqrt(real**2 + imag**2)
            phase = np.arctan2(imag, real)
            
            # 统计特征
            features = []
            
            # 1. 幅度统计特征
            features.extend([
                np.mean(amplitude),
                np.std(amplitude),
                np.max(amplitude),
                np.min(amplitude),
                np.median(amplitude),
                np.percentile(amplitude, 25),
                np.percentile(amplitude, 75),
            ])
            
            # 2. 相位统计特征
            features.extend([
                np.mean(phase),
                np.std(phase),
                np.max(phase),
                np.min(phase),
            ])
            
            # 3. 高阶统计量
            features.extend([
                np.mean(amplitude**2),  # 二阶矩
                np.mean(amplitude**4),  # 四阶矩
                np.mean(np.abs(amplitude)**3),  # 三阶矩
            ])
            
            # 4. 频域特征
            fft_real = np.fft.fft(real)
            fft_imag = np.fft.fft(imag)
            spectrum = np.abs(fft_real) + np.abs(fft_imag)
            
            features.extend([
                np.mean(spectrum),
                np.std(spectrum),
                np.max(spectrum),
            ])
            
            # 5. 瞬时特征
            inst_amplitude = amplitude
            inst_phase = np.diff(phase)
            inst_freq = inst_phase / (2 * np.pi)
            
            features.extend([
                np.mean(np.abs(inst_freq)),
                np.std(inst_freq),
            ])
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def fit(self, signals, labels):
        """训练模型"""
        # 提取特征
        features = self.extract_features(signals)
        
        # 标准化
        features = self.scaler.fit_transform(features)
        
        # 训练
        self.model.fit(features, labels)
    
    def predict(self, signals):
        """预测"""
        features = self.extract_features(signals)
        features = self.scaler.transform(features)
        return self.model.predict(features)
    
    def score(self, signals, labels):
        """评分"""
        features = self.extract_features(signals)
        features = self.scaler.transform(features)
        return self.model.score(features, labels)


# ==================== 模型工厂 ====================

def create_baseline_model(model_name, num_classes=11, **kwargs):
    """创建基线模型"""
    
    models = {
        'resnet': ResNet1D,
        'cldnn': CLDNN,
        'mcformer': MCformer,
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model_class = models[model_name.lower()]
    return model_class(num_classes=num_classes, **kwargs)


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 测试各个基线模型
    batch_size = 32
    input_channels = 2
    seq_length = 1024
    num_classes = 11
    
    x = torch.randn(batch_size, input_channels, seq_length)
    snr = torch.randint(-20, 30, (batch_size,)).float()
    
    print("="*50)
    print("Testing Baseline Models")
    print("="*50)
    
    # 1. ResNet
    print("\n1. ResNet1D")
    resnet = ResNet1D(num_classes=num_classes)
    out = resnet(x, snr)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in resnet.parameters()):,}")
    
    # 2. CLDNN
    print("\n2. CLDNN")
    cldnn = CLDNN(num_classes=num_classes)
    out = cldnn(x, snr)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in cldnn.parameters()):,}")
    
    # 3. MCformer
    print("\n3. MCformer")
    mcformer = MCformer(num_classes=num_classes)
    out = mcformer(x, snr)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in mcformer.parameters()):,}")
    
    # 4. 传统机器学习
    print("\n4. Traditional ML (SVM)")
    x_np = x.numpy()
    y_np = np.random.randint(0, num_classes, batch_size)
    
    ml_model = TraditionalMLBaseline(method='svm')
    ml_model.fit(x_np, y_np)
    predictions = ml_model.predict(x_np[:10])
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")