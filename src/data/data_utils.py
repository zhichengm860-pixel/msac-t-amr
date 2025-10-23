"""
data_utils.py - 数据预处理和增强
包含：
1. 数据标准化和归一化
2. 数据增强策略
3. 不同数据集格式的兼容性处理
4. 数据加载器
"""

import torch
import numpy as np
import pickle
import h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ==================== 数据标准化 ====================

class SignalNormalizer:
    """信号标准化器"""
    
    def __init__(self, method='minmax'):
        """
        Args:
            method: 'minmax', 'zscore', 'robust', 'maxabs'
        """
        self.method = method
        self.stats = {}
    
    def fit(self, signals):
        """
        计算统计量
        signals: [N, 2, length] or [N, length, 2]
        """
        if len(signals.shape) == 3:
            if signals.shape[1] == 2:
                # [N, 2, length]
                data = signals.reshape(signals.shape[0], -1)
            else:
                # [N, length, 2]
                data = signals.reshape(signals.shape[0], -1)
        else:
            data = signals
        
        if self.method == 'minmax':
            self.stats['min'] = np.min(data, axis=1, keepdims=True)
            self.stats['max'] = np.max(data, axis=1, keepdims=True)
        
        elif self.method == 'zscore':
            self.stats['mean'] = np.mean(data, axis=1, keepdims=True)
            self.stats['std'] = np.std(data, axis=1, keepdims=True) + 1e-8
        
        elif self.method == 'robust':
            self.stats['median'] = np.median(data, axis=1, keepdims=True)
            self.stats['q1'] = np.percentile(data, 25, axis=1, keepdims=True)
            self.stats['q3'] = np.percentile(data, 75, axis=1, keepdims=True)
        
        elif self.method == 'maxabs':
            self.stats['maxabs'] = np.max(np.abs(data), axis=1, keepdims=True) + 1e-8
        
        return self
    
    def transform(self, signals):
        """应用标准化"""
        original_shape = signals.shape
        
        if len(signals.shape) == 3:
            if signals.shape[1] == 2:
                data = signals.reshape(signals.shape[0], -1)
            else:
                data = signals.reshape(signals.shape[0], -1)
        else:
            data = signals
        
        if self.method == 'minmax':
            data = (data - self.stats['min']) / (self.stats['max'] - self.stats['min'] + 1e-8)
        
        elif self.method == 'zscore':
            data = (data - self.stats['mean']) / self.stats['std']
        
        elif self.method == 'robust':
            iqr = self.stats['q3'] - self.stats['q1'] + 1e-8
            data = (data - self.stats['median']) / iqr
        
        elif self.method == 'maxabs':
            data = data / self.stats['maxabs']
        
        return data.reshape(original_shape)
    
    def fit_transform(self, signals):
        """拟合并转换"""
        self.fit(signals)
        return self.transform(signals)
    
    def inverse_transform(self, signals):
        """逆变换"""
        original_shape = signals.shape
        
        if len(signals.shape) == 3:
            data = signals.reshape(signals.shape[0], -1)
        else:
            data = signals
        
        if self.method == 'minmax':
            data = data * (self.stats['max'] - self.stats['min']) + self.stats['min']
        
        elif self.method == 'zscore':
            data = data * self.stats['std'] + self.stats['mean']
        
        elif self.method == 'robust':
            iqr = self.stats['q3'] - self.stats['q1']
            data = data * iqr + self.stats['median']
        
        elif self.method == 'maxabs':
            data = data * self.stats['maxabs']
        
        return data.reshape(original_shape)


# ==================== 增强数据增强策略 ====================

class AdvancedSignalAugmentation:
    """增强版信号增强策略"""
    
    @staticmethod
    def awgn(signal, snr_db):
        """加性高斯白噪声"""
        signal_power = torch.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        return signal + noise
    
    @staticmethod
    def fading(signal, fade_type='rayleigh'):
        """信道衰落"""
        if fade_type == 'rayleigh':
            h_real = torch.randn_like(signal[0]) / np.sqrt(2)
            h_imag = torch.randn_like(signal[1]) / np.sqrt(2)
            
            real_faded = signal[0] * h_real - signal[1] * h_imag
            imag_faded = signal[0] * h_imag + signal[1] * h_real
            
            return torch.stack([real_faded, imag_faded])
        
        elif fade_type == 'rician':
            K = 3  # Rician K-factor
            los_real = np.sqrt(K / (K + 1))
            nlos_scale = np.sqrt(1 / (2 * (K + 1)))
            
            h_real = los_real + torch.randn_like(signal[0]) * nlos_scale
            h_imag = torch.randn_like(signal[1]) * nlos_scale
            
            real_faded = signal[0] * h_real - signal[1] * h_imag
            imag_faded = signal[0] * h_imag + signal[1] * h_real
            
            return torch.stack([real_faded, imag_faded])
    
    @staticmethod
    def frequency_selective_fading(signal, num_paths=3, max_delay=10):
        """频率选择性衰落"""
        faded_signal = torch.zeros_like(signal)
        
        for _ in range(num_paths):
            delay = np.random.randint(0, max_delay)
            amplitude = np.random.rayleigh(1.0)
            phase = np.random.uniform(-np.pi, np.pi)
            
            delayed = torch.roll(signal, delay, dims=-1)
            delayed[0] = delayed[0] * amplitude * np.cos(phase)
            delayed[1] = delayed[1] * amplitude * np.sin(phase)
            
            faded_signal += delayed
        
        return faded_signal / num_paths
    
    @staticmethod
    def time_warping(signal, warp_factor=0.1):
        """时间伸缩"""
        length = signal.shape[-1]
        scale = 1 + np.random.uniform(-warp_factor, warp_factor)
        new_length = int(length * scale)
        
        # 使用插值进行时间伸缩
        signal_np = signal.numpy()
        warped = np.zeros_like(signal_np)
        
        for i in range(signal.shape[0]):
            old_indices = np.linspace(0, length-1, new_length)
            new_indices = np.arange(length)
            warped[i] = np.interp(new_indices, 
                                 np.linspace(0, length-1, len(old_indices)), 
                                 np.interp(old_indices, new_indices, signal_np[i]))
        
        return torch.from_numpy(warped).float()
    
    @staticmethod
    def mixup(signal1, signal2, alpha=0.2):
        """Mixup数据增强"""
        lam = np.random.beta(alpha, alpha)
        return lam * signal1 + (1 - lam) * signal2, lam
    
    @staticmethod
    def cutout(signal, cut_ratio=0.1):
        """Cutout - 随机遮挡部分信号"""
        length = signal.shape[-1]
        cut_length = int(length * cut_ratio)
        start = np.random.randint(0, length - cut_length)
        
        signal_cut = signal.clone()
        signal_cut[:, start:start+cut_length] = 0
        
        return signal_cut
    
    @staticmethod
    def random_augment(signal, augment_prob=0.5):
        """随机应用多种增强"""
        augmented = signal.clone()
        
        # AWGN
        if np.random.rand() < augment_prob:
            snr = np.random.uniform(-5, 20)
            augmented = AdvancedSignalAugmentation.awgn(augmented, snr)
        
        # 时间偏移
        if np.random.rand() < augment_prob:
            shift = np.random.randint(-50, 50)
            augmented = torch.roll(augmented, shift, dims=-1)
        
        # 幅度缩放
        if np.random.rand() < augment_prob:
            scale = np.random.uniform(0.8, 1.2)
            augmented = augmented * scale
        
        # 相位偏移
        if np.random.rand() < augment_prob:
            phase = np.random.uniform(-np.pi, np.pi)
            real, imag = augmented[0], augmented[1]
            rotated_real = real * np.cos(phase) - imag * np.sin(phase)
            rotated_imag = real * np.sin(phase) + imag * np.cos(phase)
            augmented = torch.stack([rotated_real, rotated_imag])
        
        # 频率偏移
        if np.random.rand() < augment_prob:
            freq_shift = np.random.uniform(-0.1, 0.1)
            t = torch.arange(augmented.size(-1)).float()
            shift_factor = torch.exp(1j * 2 * np.pi * freq_shift * t)
            
            complex_signal = torch.complex(augmented[0], augmented[1])
            shifted = complex_signal * shift_factor
            augmented = torch.stack([shifted.real, shifted.imag])
        
        return augmented


# ==================== 数据集类 ====================

class RadioMLDataset(Dataset):
    """RadioML数据集"""
    
    def __init__(self, signals, labels, snrs, transform=None, augment=False):
        """
        Args:
            signals: [N, 2, length] - IQ信号
            labels: [N] - 调制类型标签
            snrs: [N] - SNR值
            transform: 数据变换函数
            augment: 是否应用数据增强
        """
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)
        self.snrs = torch.FloatTensor(snrs)
        self.transform = transform
        self.augment = augment
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        snr = self.snrs[idx]
        
        # 应用数据增强
        if self.augment:
            signal = AdvancedSignalAugmentation.random_augment(signal)
        
        # 应用变换
        if self.transform:
            signal = self.transform(signal)
        
        return signal, label, snr


# ==================== 数据加载器 ====================

class DatasetLoader:
    """数据集加载器 - 支持多种格式"""
    
    def __init__(self, config):
        """初始化数据加载器"""
        self.config = config
    
    def load_radioml_data(self, dataset_path, batch_size=128, train_ratio=0.7, val_ratio=0.15):
        """
        加载RadioML数据集并创建数据加载器
        """
        # 根据文件扩展名判断数据集类型
        if dataset_path.endswith('.pkl'):
            signals, labels, snrs, mod_types = self.load_radioml2016(dataset_path)
        elif dataset_path.endswith('.h5') or dataset_path.endswith('.hdf5'):
            signals, labels, snrs, mod_types = self.load_radioml2018(dataset_path)
        else:
            signals, labels, snrs, mod_types = self.load_custom(dataset_path)
        
        # 创建数据加载器
        test_ratio = 1.0 - train_ratio - val_ratio
        train_loader, val_loader, test_loader = self.create_dataloaders(
            signals, labels, snrs,
            batch_size=batch_size,
            test_size=test_ratio,
            val_size=val_ratio,
            augment_train=True,
            num_workers=0  # Windows上设为0避免多进程问题
        )
        
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def load_radioml2016(file_path):
        """
        加载RadioML2016.10a数据集
        格式: pickle文件，字典 {(mod, snr): [samples, 2, 128]}
        """
        print(f"Loading RadioML2016 from {file_path}...")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        signals_list = []
        labels_list = []
        snrs_list = []
        
        # 调制类型映射
        mod_types = sorted(list(set([key[0] for key in data.keys()])))
        mod_to_idx = {mod: idx for idx, mod in enumerate(mod_types)}
        
        for (mod, snr), samples in data.items():
            num_samples = samples.shape[0]
            signals_list.append(samples)
            labels_list.extend([mod_to_idx[mod]] * num_samples)
            snrs_list.extend([snr] * num_samples)
        
        signals = np.vstack(signals_list)
        labels = np.array(labels_list)
        snrs = np.array(snrs_list)
        
        print(f"Loaded {len(signals)} samples")
        print(f"Modulation types: {mod_types}")
        print(f"SNR range: {np.min(snrs)} to {np.max(snrs)} dB")
        
        return signals, labels, snrs, mod_types
    
    @staticmethod
    def load_radioml2018(file_path):
        """
        加载RadioML2018.01a数据集
        格式: HDF5文件
        """
        print(f"Loading RadioML2018 from {file_path}...")
        
        with h5py.File(file_path, 'r') as f:
            signals = f['X'][:]  # [N, 1024, 2]
            labels = f['Y'][:]   # [N, 24] - one-hot
            snrs = f['Z'][:]     # [N]
        
        # 转换格式: [N, 1024, 2] -> [N, 2, 1024]
        signals = np.transpose(signals, (0, 2, 1))
        
        # one-hot转索引
        labels = np.argmax(labels, axis=1)
        
        # 调制类型
        mod_types = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                    'FM', 'GMSK', 'OQPSK']
        
        print(f"Loaded {len(signals)} samples")
        print(f"Number of modulation types: {len(mod_types)}")
        print(f"SNR range: {np.min(snrs)} to {np.max(snrs)} dB")
        
        return signals, labels, snrs, mod_types
    
    @staticmethod
    def load_custom(file_path):
        """
        加载自定义格式数据
        假设是.npz文件，包含signals, labels, snrs
        """
        print(f"Loading custom dataset from {file_path}...")
        
        data = np.load(file_path)
        signals = data['signals']
        labels = data['labels']
        snrs = data['snrs']
        
        mod_types = data.get('mod_types', None)
        
        print(f"Loaded {len(signals)} samples")
        
        return signals, labels, snrs, mod_types
    
    @staticmethod
    def create_dataloaders(signals, labels, snrs, 
                          batch_size=128, 
                          test_size=0.2, 
                          val_size=0.1,
                          augment_train=True,
                          num_workers=4):
        """
        创建训练/验证/测试数据加载器
        """
        # 首先分离测试集
        X_temp, X_test, y_temp, y_test, snr_temp, snr_test = train_test_split(
            signals, labels, snrs, test_size=test_size, random_state=42, stratify=labels
        )
        
        # 再从剩余数据中分离验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
            X_temp, y_temp, snr_temp, test_size=val_size_adjusted, 
            random_state=42, stratify=y_temp
        )
        
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # 创建数据集
        train_dataset = RadioMLDataset(X_train, y_train, snr_train, augment=augment_train)
        val_dataset = RadioMLDataset(X_val, y_val, snr_val, augment=False)
        test_dataset = RadioMLDataset(X_test, y_test, snr_test, augment=False)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 示例：加载和处理数据
    
    # 方法1: 加载RadioML2016数据集
    # signals, labels, snrs, mod_types = DatasetLoader.load_radioml2016('RML2016.10a_dict.pkl')
    
    # 方法2: 生成模拟数据
    print("Generating simulated data...")
    num_samples = 10000
    signals = np.random.randn(num_samples, 2, 128)
    labels = np.random.randint(0, 11, num_samples)
    snrs = np.random.randint(-20, 30, num_samples)
    
    # 数据标准化
    print("\nNormalizing data...")
    normalizer = SignalNormalizer(method='zscore')
    signals_normalized = normalizer.fit_transform(signals)
    
    # 创建数据加载器
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = DatasetLoader.create_dataloaders(
        signals_normalized, labels, snrs,
        batch_size=128,
        test_size=0.2,
        val_size=0.1,
        augment_train=True
    )
    
    # 测试数据加载
    print("\nTesting dataloader...")
    for batch_signals, batch_labels, batch_snrs in train_loader:
        print(f"Batch signals shape: {batch_signals.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Batch SNRs shape: {batch_snrs.shape}")
        break
    
    print("\nData loading完成!")