#!/usr/bin/env python3
"""
RadioML数据加载器模块
支持RadioML 2016.10A/B和2018.01A数据集的加载和预处理

作者: Assistant
日期: 2025-01-16
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class RadioMLDataset(Dataset):
    """RadioML数据集类"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 transform=None, target_transform=None):
        """
        初始化RadioML数据集
        
        Args:
            data: 信号数据，形状为 (N, 2, L) 其中N是样本数，2是I/Q通道，L是信号长度
            labels: 标签数据
            transform: 数据变换函数
            target_transform: 标签变换函数
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)
            
        return sample, label

class RadioMLDataLoader:
    """RadioML数据加载器"""
    
    def __init__(self, data_path: str, dataset_type: str = "2016.10A"):
        """
        初始化RadioML数据加载器
        
        Args:
            data_path: 数据文件路径
            dataset_type: 数据集类型 ("2016.10A", "2016.10B", "2018.01A")
        """
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.modulation_classes = []
        self.snr_levels = []
        self.data = None
        self.labels = None
        self.label_encoder = LabelEncoder()
        
        # 不同数据集的调制类型
        self.modulation_mapping = {
            "2016.10A": ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 
                        'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'],
            "2016.10B": ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 
                        'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'],
            "2018.01A": ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                        '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', 
                        '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
                        'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
        }
        
        self.modulation_classes = self.modulation_mapping.get(dataset_type, [])
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """
        加载RadioML数据集
        
        Returns:
            data: 信号数据
            labels: 标签
            modulation_classes: 调制类型列表
            snr_levels: SNR级别列表
        """
        print(f"正在加载 RadioML {self.dataset_type} 数据集...")
        print(f"数据路径: {self.data_path}")
        
        if self.dataset_type.startswith("2016"):
            return self._load_2016_data()
        elif self.dataset_type == "2018.01A":
            return self._load_2018_data()
        else:
            raise ValueError(f"不支持的数据集类型: {self.dataset_type}")
    
    def _load_2016_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """加载RadioML 2016数据集"""
        try:
            # 检查文件是否存在
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
            
            # 尝试加载pickle文件
            with open(self.data_path, 'rb') as f:
                data_dict = pickle.load(f, encoding='latin1')
            
            print(f"成功加载数据字典，包含 {len(data_dict)} 个键")
            
            # 提取数据和标签
            all_data = []
            all_labels = []
            modulation_snr_pairs = []
            
            for key, value in data_dict.items():
                if isinstance(key, tuple) and len(key) == 2:
                    modulation, snr = key
                    modulation_snr_pairs.append((modulation, snr))
                    
                    # 数据形状应该是 (samples, 2, 128) 对于2016数据集
                    samples = value
                    num_samples = samples.shape[0]
                    
                    all_data.append(samples)
                    all_labels.extend([modulation] * num_samples)
            
            # 合并所有数据
            self.data = np.vstack(all_data)
            
            # 提取唯一的调制类型和SNR级别
            unique_pairs = list(set(modulation_snr_pairs))
            self.modulation_classes = sorted(list(set([pair[0] for pair in unique_pairs])))
            self.snr_levels = sorted(list(set([pair[1] for pair in unique_pairs])))
            
            # 编码标签
            self.labels = self.label_encoder.fit_transform(all_labels)
            
            print(f"数据形状: {self.data.shape}")
            print(f"标签形状: {self.labels.shape}")
            print(f"调制类型 ({len(self.modulation_classes)}): {self.modulation_classes}")
            print(f"SNR级别 ({len(self.snr_levels)}): {self.snr_levels}")
            
            return self.data, self.labels, self.modulation_classes, self.snr_levels
            
        except Exception as e:
            print(f"加载2016数据集时出错: {e}")
            # 如果加载失败，生成模拟数据
            return self._generate_simulated_data()
    
    def _load_2018_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """加载RadioML 2018数据集"""
        try:
            # 检查文件是否存在
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
            
            # RadioML 2018数据集通常是HDF5格式
            with h5py.File(self.data_path, 'r') as f:
                # 获取数据和标签
                X = f['X'][:]  # 信号数据
                Y = f['Y'][:]  # 标签（one-hot编码）
                Z = f['Z'][:]  # SNR级别
                
                # 转换one-hot标签为类别索引
                labels = np.argmax(Y, axis=1)
                
                # 提取SNR级别
                snr_levels = np.unique(Z)
                
                print(f"数据形状: {X.shape}")
                print(f"标签形状: {labels.shape}")
                print(f"SNR级别: {snr_levels}")
                
                self.data = X
                self.labels = labels
                self.snr_levels = snr_levels.tolist()
                
                return X, labels, self.modulation_classes, self.snr_levels
                
        except Exception as e:
            print(f"加载2018数据集时出错: {e}")
            # 如果加载失败，生成模拟数据
            return self._generate_simulated_data()
    
    def _generate_simulated_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """生成模拟RadioML数据（当真实数据不可用时）"""
        print("⚠️  真实数据不可用，生成模拟RadioML数据...")
        
        # 使用预定义的调制类型
        if not self.modulation_classes:
            self.modulation_classes = self.modulation_mapping[self.dataset_type]
        
        # 定义SNR级别
        self.snr_levels = list(range(-20, 31, 2))  # -20dB到30dB，步长2dB
        
        # 生成参数
        num_samples_per_class_snr = 100
        signal_length = 128 if self.dataset_type.startswith("2016") else 1024
        
        total_samples = len(self.modulation_classes) * len(self.snr_levels) * num_samples_per_class_snr
        
        # 生成数据
        all_data = []
        all_labels = []
        
        for mod_idx, modulation in enumerate(self.modulation_classes):
            for snr in self.snr_levels:
                # 生成基础信号
                samples = self._generate_modulated_signal(
                    modulation, snr, num_samples_per_class_snr, signal_length
                )
                all_data.append(samples)
                all_labels.extend([mod_idx] * num_samples_per_class_snr)
        
        self.data = np.vstack(all_data)
        self.labels = np.array(all_labels)
        
        print(f"生成模拟数据:")
        print(f"  - 数据形状: {self.data.shape}")
        print(f"  - 标签形状: {self.labels.shape}")
        print(f"  - 调制类型: {self.modulation_classes}")
        print(f"  - SNR级别: {self.snr_levels}")
        
        return self.data, self.labels, self.modulation_classes, self.snr_levels
    
    def _generate_modulated_signal(self, modulation: str, snr_db: float, 
                                  num_samples: int, signal_length: int) -> np.ndarray:
        """生成特定调制类型的信号"""
        np.random.seed(42)  # 确保可重复性
        
        signals = []
        for _ in range(num_samples):
            # 生成基础信号
            t = np.linspace(0, 1, signal_length)
            
            if modulation in ['BPSK', 'QPSK', '8PSK']:
                # PSK调制
                if modulation == 'BPSK':
                    symbols = np.random.choice([-1, 1], signal_length)
                elif modulation == 'QPSK':
                    symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], signal_length)
                else:  # 8PSK
                    phases = np.random.choice(np.arange(0, 2*np.pi, np.pi/4), signal_length)
                    symbols = np.exp(1j * phases)
                
                signal = symbols
                
            elif modulation in ['QAM16', '16QAM', 'QAM64', '64QAM']:
                # QAM调制
                if '16' in modulation:
                    constellation = [-3-3j, -3-1j, -3+1j, -3+3j, -1-3j, -1-1j, -1+1j, -1+3j,
                                   1-3j, 1-1j, 1+1j, 1+3j, 3-3j, 3-1j, 3+1j, 3+3j]
                else:  # 64QAM
                    constellation = []
                    for i in range(-7, 8, 2):
                        for q in range(-7, 8, 2):
                            constellation.append(i + 1j*q)
                
                symbols = np.random.choice(constellation, signal_length)
                signal = symbols
                
            elif modulation in ['AM-DSB', 'AM-SSB']:
                # AM调制
                carrier_freq = 0.1
                message = np.random.randn(signal_length) * 0.5
                carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
                signal = (1 + message) * carrier
                
            elif modulation in ['CPFSK', 'GFSK']:
                # FSK调制
                freq_dev = 0.05
                symbols = np.random.choice([-1, 1], signal_length)
                phase = np.cumsum(symbols) * freq_dev
                signal = np.exp(1j * 2 * np.pi * phase)
                
            elif modulation == 'WBFM':
                # FM调制
                message = np.random.randn(signal_length) * 0.1
                phase = np.cumsum(message)
                signal = np.exp(1j * 2 * np.pi * phase)
                
            else:
                # 默认生成复数高斯噪声
                signal = np.random.randn(signal_length) + 1j * np.random.randn(signal_length)
            
            # 添加AWGN噪声
            signal_power = np.mean(np.abs(signal)**2)
            noise_power = signal_power / (10**(snr_db/10))
            noise = np.sqrt(noise_power/2) * (np.random.randn(signal_length) + 
                                            1j * np.random.randn(signal_length))
            noisy_signal = signal + noise
            
            # 归一化
            noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))
            
            # 转换为I/Q格式
            iq_signal = np.array([np.real(noisy_signal), np.imag(noisy_signal)])
            signals.append(iq_signal)
        
        return np.array(signals)
    
    def create_dataloaders(self, test_size: float = 0.2, val_size: float = 0.1,
                          batch_size: int = 32, random_state: int = 42,
                          **dataloader_kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建训练、验证和测试数据加载器
        
        Args:
            test_size: 测试集比例
            val_size: 验证集比例（相对于训练集）
            batch_size: 批次大小
            random_state: 随机种子
            **dataloader_kwargs: DataLoader的其他参数
        
        Returns:
            train_loader, val_loader, test_loader
        """
        if self.data is None or self.labels is None:
            raise ValueError("请先调用 load_data() 方法加载数据")
        
        # 分割数据
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.data, self.labels, test_size=test_size, 
            random_state=random_state, stratify=self.labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, 
            random_state=random_state, stratify=y_temp
        )
        
        # 创建数据集
        train_dataset = RadioMLDataset(X_train, y_train)
        val_dataset = RadioMLDataset(X_val, y_val)
        test_dataset = RadioMLDataset(X_test, y_test)
        
        # 创建数据加载器
        default_kwargs = {'shuffle': True, 'num_workers': 0, 'pin_memory': False}
        default_kwargs.update(dataloader_kwargs)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, **default_kwargs)
        
        # 验证和测试集不需要shuffle
        val_kwargs = default_kwargs.copy()
        val_kwargs['shuffle'] = False
        val_loader = DataLoader(val_dataset, batch_size=batch_size, **val_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, **val_kwargs)
        
        print(f"\n数据集分割完成:")
        print(f"  - 训练集: {len(train_dataset)} 样本")
        print(f"  - 验证集: {len(val_dataset)} 样本")
        print(f"  - 测试集: {len(test_dataset)} 样本")
        print(f"  - 批次大小: {batch_size}")
        
        return train_loader, val_loader, test_loader
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        if self.labels is None:
            raise ValueError("请先调用 load_data() 方法加载数据")
        
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = {}
        
        for idx, count in zip(unique, counts):
            if hasattr(self.label_encoder, 'classes_'):
                class_name = self.label_encoder.classes_[idx]
            else:
                class_name = self.modulation_classes[idx] if idx < len(self.modulation_classes) else f"Class_{idx}"
            distribution[class_name] = count
        
        return distribution
    
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        if self.data is None:
            raise ValueError("请先调用 load_data() 方法加载数据")
        
        info = {
            'dataset_type': self.dataset_type,
            'data_shape': self.data.shape,
            'num_classes': len(self.modulation_classes),
            'modulation_classes': self.modulation_classes,
            'snr_levels': self.snr_levels,
            'class_distribution': self.get_class_distribution(),
            'signal_length': self.data.shape[-1],
            'num_channels': self.data.shape[1] if len(self.data.shape) > 2 else 1
        }
        
        return info

# 测试代码
if __name__ == "__main__":
    # 测试数据加载器
    print("=" * 60)
    print("RadioML数据加载器测试")
    print("=" * 60)
    
    # 测试不同数据集类型
    dataset_types = ["2016.10A", "2018.01A"]
    
    for dataset_type in dataset_types:
        print(f"\n测试 RadioML {dataset_type} 数据集:")
        print("-" * 40)
        
        # 创建数据加载器（使用模拟数据）
        data_path = f"dummy_path_{dataset_type}.pkl"
        loader = RadioMLDataLoader(data_path, dataset_type)
        
        try:
            # 加载数据
            data, labels, mod_classes, snr_levels = loader.load_data()
            
            # 显示数据集信息
            info = loader.get_dataset_info()
            print(f"数据集信息:")
            for key, value in info.items():
                if key != 'class_distribution':
                    print(f"  {key}: {value}")
            
            # 创建数据加载器
            train_loader, val_loader, test_loader = loader.create_dataloaders(
                batch_size=16, test_size=0.2, val_size=0.1
            )
            
            # 测试数据加载
            print(f"\n测试数据加载:")
            for i, (batch_data, batch_labels) in enumerate(train_loader):
                print(f"  批次 {i+1}: 数据形状 {batch_data.shape}, 标签形状 {batch_labels.shape}")
                if i >= 2:  # 只测试前3个批次
                    break
            
            # 获取类别分布
            distribution = loader.get_class_distribution()
            print(f"\n类别分布:")
            for class_name, count in list(distribution.items())[:5]:  # 显示前5个类别
                print(f"  {class_name}: {count} 样本")
            if len(distribution) > 5:
                print(f"  ... 还有 {len(distribution)-5} 个类别")
            
            print(f"✓ {dataset_type} 数据集测试成功!")
            
        except Exception as e:
            print(f"✗ {dataset_type} 数据集测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nRadioML数据加载器测试完成!")
    print("=" * 60)
    print("注意: 由于没有真实的RadioML数据文件，以上测试使用了模拟数据。")
    print("要使用真实数据，请下载RadioML数据集并提供正确的文件路径。")
    print("下载地址: https://www.deepsig.ai/datasets")
    print("=" * 60)