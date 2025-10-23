"""
dataset_config.py - 数据集配置文件
定义不同数据集的参数、路径和预处理配置
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class DatasetConfig:
    """数据集配置基类"""
    name: str
    path: str
    num_classes: int
    modulation_types: List[str]
    snr_range: Tuple[int, int]
    sample_rate: float
    signal_length: int
    file_format: str
    description: str


# ==================== RadioML 数据集配置 ====================

class RadioMLConfig:
    """RadioML数据集配置"""
    
    # RadioML 2016.10A 配置
    RADIOML_2016_10A = DatasetConfig(
        name="RadioML2016.10A",
        path="dataset/RadioML 2016.10A/RML2016.10a_dict.pkl",
        num_classes=11,
        modulation_types=[
            "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", 
            "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"
        ],
        snr_range=(-20, 18),
        sample_rate=200000,  # 200 kHz
        signal_length=128,
        file_format="pickle",
        description="RadioML 2016.10A dataset with 11 modulation types"
    )
    
    # RadioML 2016.10B 配置
    RADIOML_2016_10B = DatasetConfig(
        name="RadioML2016.10B",
        path="dataset/RadioML 2016.10B/RML2016.10b.dat",
        num_classes=10,
        modulation_types=[
            "8PSK", "AM-DSB", "BPSK", "CPFSK", "GFSK", 
            "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"
        ],
        snr_range=(-20, 18),
        sample_rate=200000,  # 200 kHz
        signal_length=128,
        file_format="binary",
        description="RadioML 2016.10B dataset with 10 modulation types"
    )
    
    # RadioML 2018.01A 配置
    RADIOML_2018_01A = DatasetConfig(
        name="RadioML2018.01A",
        path="dataset/RadioML 2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5",
        num_classes=24,
        modulation_types=[
            "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
            "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", 
            "64QAM", "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", 
            "AM-DSB-SC", "FM", "GMSK", "OQPSK"
        ],
        snr_range=(-20, 30),
        sample_rate=200000,  # 200 kHz
        signal_length=1024,
        file_format="hdf5",
        description="RadioML 2018.01A dataset with 24 modulation types"
    )


# ==================== 数据预处理配置 ====================

@dataclass
class PreprocessConfig:
    """数据预处理配置"""
    normalize: bool = True
    normalize_method: str = "standard"  # "standard", "minmax", "robust"
    add_noise: bool = False
    noise_std: float = 0.1
    augmentation: bool = True
    augmentation_methods: List[str] = None
    
    def __post_init__(self):
        if self.augmentation_methods is None:
            self.augmentation_methods = ["time_shift", "frequency_shift", "amplitude_scale"]


@dataclass
class DataLoaderConfig:
    """数据加载器配置"""
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


# ==================== 完整的数据集配置管理器 ====================

class DatasetConfigManager:
    """数据集配置管理器"""
    
    def __init__(self, base_path: str = "."):
        """
        Args:
            base_path: 项目根目录路径
        """
        self.base_path = base_path
        self.available_datasets = {
            "radioml_2016_10a": RadioMLConfig.RADIOML_2016_10A,
            "radioml_2016_10b": RadioMLConfig.RADIOML_2016_10B,
            "radioml_2018_01a": RadioMLConfig.RADIOML_2018_01A,
        }
        
        # 默认预处理配置
        self.default_preprocess_config = PreprocessConfig()
        
        # 默认数据加载器配置
        self.default_dataloader_config = DataLoaderConfig()
    
    def get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """获取数据集配置"""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.available_datasets.keys())}")
        
        config = self.available_datasets[dataset_name]
        # 更新路径为绝对路径
        config.path = os.path.join(self.base_path, config.path)
        return config
    
    def get_preprocess_config(self, **kwargs) -> PreprocessConfig:
        """获取预处理配置"""
        config_dict = {
            'normalize': kwargs.get('normalize', self.default_preprocess_config.normalize),
            'normalize_method': kwargs.get('normalize_method', self.default_preprocess_config.normalize_method),
            'add_noise': kwargs.get('add_noise', self.default_preprocess_config.add_noise),
            'noise_std': kwargs.get('noise_std', self.default_preprocess_config.noise_std),
            'augmentation': kwargs.get('augmentation', self.default_preprocess_config.augmentation),
            'augmentation_methods': kwargs.get('augmentation_methods', self.default_preprocess_config.augmentation_methods),
        }
        return PreprocessConfig(**config_dict)
    
    def get_dataloader_config(self, **kwargs) -> DataLoaderConfig:
        """获取数据加载器配置"""
        config_dict = {
            'batch_size': kwargs.get('batch_size', self.default_dataloader_config.batch_size),
            'shuffle': kwargs.get('shuffle', self.default_dataloader_config.shuffle),
            'num_workers': kwargs.get('num_workers', self.default_dataloader_config.num_workers),
            'pin_memory': kwargs.get('pin_memory', self.default_dataloader_config.pin_memory),
            'drop_last': kwargs.get('drop_last', self.default_dataloader_config.drop_last),
            'train_ratio': kwargs.get('train_ratio', self.default_dataloader_config.train_ratio),
            'val_ratio': kwargs.get('val_ratio', self.default_dataloader_config.val_ratio),
            'test_ratio': kwargs.get('test_ratio', self.default_dataloader_config.test_ratio),
        }
        return DataLoaderConfig(**config_dict)
    
    def list_available_datasets(self) -> List[str]:
        """列出可用的数据集"""
        return list(self.available_datasets.keys())
    
    def check_dataset_exists(self, dataset_name: str) -> bool:
        """检查数据集文件是否存在"""
        config = self.get_dataset_config(dataset_name)
        return os.path.exists(config.path)
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """获取数据集详细信息"""
        config = self.get_dataset_config(dataset_name)
        return {
            'name': config.name,
            'path': config.path,
            'exists': os.path.exists(config.path),
            'num_classes': config.num_classes,
            'modulation_types': config.modulation_types,
            'snr_range': config.snr_range,
            'sample_rate': config.sample_rate,
            'signal_length': config.signal_length,
            'file_format': config.file_format,
            'description': config.description
        }
    
    def validate_split_ratios(self, train_ratio: float, val_ratio: float, test_ratio: float) -> bool:
        """验证数据集分割比例"""
        total = train_ratio + val_ratio + test_ratio
        return abs(total - 1.0) < 1e-6


# ==================== 数据集特定的配置函数 ====================

def get_radioml_config(dataset_version: str = "2016.10a", **kwargs) -> Tuple[DatasetConfig, PreprocessConfig, DataLoaderConfig]:
    """
    获取RadioML数据集的完整配置
    
    Args:
        dataset_version: 数据集版本 ("2016.10a", "2016.10b", "2018.01a")
        **kwargs: 其他配置参数
    
    Returns:
        (dataset_config, preprocess_config, dataloader_config)
    """
    manager = DatasetConfigManager()
    
    # 映射版本名称
    version_map = {
        "2016.10a": "radioml_2016_10a",
        "2016.10b": "radioml_2016_10b", 
        "2018.01a": "radioml_2018_01a"
    }
    
    if dataset_version not in version_map:
        raise ValueError(f"Unknown RadioML version: {dataset_version}")
    
    dataset_name = version_map[dataset_version]
    
    dataset_config = manager.get_dataset_config(dataset_name)
    preprocess_config = manager.get_preprocess_config(**kwargs)
    dataloader_config = manager.get_dataloader_config(**kwargs)
    
    return dataset_config, preprocess_config, dataloader_config


def create_custom_dataset_config(
    name: str,
    path: str,
    num_classes: int,
    modulation_types: List[str],
    snr_range: Tuple[int, int],
    signal_length: int,
    **kwargs
) -> DatasetConfig:
    """
    创建自定义数据集配置
    
    Args:
        name: 数据集名称
        path: 数据集路径
        num_classes: 类别数量
        modulation_types: 调制类型列表
        snr_range: SNR范围
        signal_length: 信号长度
        **kwargs: 其他参数
    
    Returns:
        DatasetConfig对象
    """
    return DatasetConfig(
        name=name,
        path=path,
        num_classes=num_classes,
        modulation_types=modulation_types,
        snr_range=snr_range,
        sample_rate=kwargs.get('sample_rate', 200000),
        signal_length=signal_length,
        file_format=kwargs.get('file_format', 'pickle'),
        description=kwargs.get('description', f'Custom dataset: {name}')
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建配置管理器
    manager = DatasetConfigManager()
    
    # 列出可用数据集
    print("Available datasets:")
    for dataset in manager.list_available_datasets():
        print(f"  - {dataset}")
    
    # 获取RadioML 2016.10A配置
    dataset_config, preprocess_config, dataloader_config = get_radioml_config("2016.10a")
    
    print(f"\nDataset: {dataset_config.name}")
    print(f"Classes: {dataset_config.num_classes}")
    print(f"Signal length: {dataset_config.signal_length}")
    print(f"SNR range: {dataset_config.snr_range}")
    
    # 检查数据集是否存在
    for dataset_name in manager.list_available_datasets():
        exists = manager.check_dataset_exists(dataset_name)
        print(f"{dataset_name}: {'✓' if exists else '✗'}")