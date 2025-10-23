"""
config.py - 配置管理系统
"""

import yaml
import json
from dataclasses import dataclass, asdict
from typing import List, Optional
import os


@dataclass
class ModelConfig:
    """模型配置 - 基于最佳训练策略优化的架构参数"""
    num_classes: int = 11
    feature_dim: int = 256  # 保持256维特征，平衡性能和效率
    
    # 多尺度卷积配置 - 优化通道数以配合大批量训练
    conv_channels: List[int] = None
    kernel_sizes: List[int] = None
    
    # 注意力机制配置 - 优化以配合Adam优化器和低学习率
    num_heads: int = 8  # 保持8头注意力，适合256维特征
    attention_dropout: float = 0.1  # 适度dropout，配合大批量训练
    
    # SNR嵌入配置
    snr_embed_dim: int = 32
    
    # 分类器配置 - 降低dropout以配合大批量+低学习率策略
    classifier_hidden: List[int] = None
    dropout: float = 0.3  # 优化：从0.5降低到0.3，配合大批量训练的稳定性
    
    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [64, 128, 256, 512]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 5, 7, 9]
        if self.classifier_hidden is None:
            self.classifier_hidden = [512, 256]


@dataclass
class TrainingConfig:
    """训练配置 - 基于策略分析优化的最佳配置"""
    # 基本设置 - 使用最佳策略：大批量+低学习率
    epochs: int = 100
    batch_size: int = 128  # 优化：从64增加到128（大批量）
    learning_rate: float = 3e-4  # 优化：从1e-4调整到3e-4（低学习率范围内的最佳值）
    weight_decay: float = 1e-4  # 保持最佳策略的权重衰减
    
    # 优化器 - 使用最佳策略的Adam优化器
    optimizer: str = 'adam'  # 优化：改为adam（最佳策略使用的优化器）
    momentum: float = 0.9  # for SGD
    
    # 学习率调度
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau'
    lr_patience: int = 10  # for ReduceLROnPlateau
    lr_factor: float = 0.1
    
    # 早停
    early_stopping: bool = True
    patience: int = 20
    
    # 数据增强
    augmentation: bool = True
    augment_prob: float = 0.5
    
    # 损失函数
    loss_function: str = 'cross_entropy'  # 'cross_entropy', 'focal', 'label_smoothing'
    label_smoothing: float = 0.1
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # 混合精度训练
    mixed_precision: bool = True
    
    # 梯度裁剪
    grad_clip: Optional[float] = 1.0


@dataclass
class PretrainConfig:
    """预训练配置"""
    pretrain_epochs: int = 100
    pretrain_batch_size: int = 128
    pretrain_lr: float = 1e-3
    
    # 多任务权重
    recon_weight: float = 1.0
    contrast_weight: float = 0.5
    mask_weight: float = 0.3
    
    # 对比学习
    temperature: float = 0.5
    projection_dim: int = 128
    
    # 掩码预测
    mask_ratio: float = 0.15
    mask_length: int = 10


@dataclass
class DataConfig:
    """数据配置"""
    dataset_path: str = './dataset/RadioML 2016.10A/RML2016.10a_dict.pkl'
    dataset_type: str = 'radioml2016'  # 'radioml2016', 'radioml2018', 'custom'
    
    # 数据划分
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed: int = 42
    
    # 数据预处理
    normalize: bool = True
    normalization_method: str = 'zscore'  # 'zscore', 'minmax', 'robust'
    
    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 实验信息
    experiment_name: str = 'amrnet_experiment'
    description: str = ''
    tags: List[str] = None
    
    # 设备
    device: str = 'cpu'
    gpu_ids: List[int] = None
    
    # 保存和日志
    save_dir: str = 'experiments'
    log_interval: int = 10
    save_interval: int = 10
    
    # 可视化
    visualize: bool = True
    
    # 随机种子
    seed: int = 42
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.gpu_ids is None:
            self.gpu_ids = [0]


class Config:
    """完整配置类"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.pretrain = PretrainConfig()
        self.data = DataConfig()
        self.experiment = ExperimentConfig()
    
    def to_dict(self):
        """转换为字典"""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'pretrain': asdict(self.pretrain),
            'data': asdict(self.data),
            'experiment': asdict(self.experiment)
        }
    
    def save_yaml(self, path):
        """保存为YAML文件"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        print(f"Config saved to {path}")
    
    def save_json(self, path):
        """保存为JSON文件"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"Config saved to {path}")
    
    @classmethod
    def from_yaml(cls, path):
        """从YAML文件加载"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        config.model = ModelConfig(**config_dict['model'])
        config.training = TrainingConfig(**config_dict['training'])
        config.pretrain = PretrainConfig(**config_dict['pretrain'])
        config.data = DataConfig(**config_dict['data'])
        config.experiment = ExperimentConfig(**config_dict['experiment'])
        
        return config
    
    @classmethod
    def from_json(cls, path):
        """从JSON文件加载"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.model = ModelConfig(**config_dict['model'])
        config.training = TrainingConfig(**config_dict['training'])
        config.pretrain = PretrainConfig(**config_dict['pretrain'])
        config.data = DataConfig(**config_dict['data'])
        config.experiment = ExperimentConfig(**config_dict['experiment'])
        
        return config
    
    def print_config(self):
        """打印配置"""
        print("\n" + "="*70)
        print("EXPERIMENT CONFIGURATION")
        print("="*70)
        
        print("\n[Model Configuration]")
        for key, value in asdict(self.model).items():
            print(f"  {key}: {value}")
        
        print("\n[Training Configuration]")
        for key, value in asdict(self.training).items():
            print(f"  {key}: {value}")
        
        print("\n[Pretrain Configuration]")
        for key, value in asdict(self.pretrain).items():
            print(f"  {key}: {value}")
        
        print("\n[Data Configuration]")
        for key, value in asdict(self.data).items():
            print(f"  {key}: {value}")
        
        print("\n[Experiment Configuration]")
        for key, value in asdict(self.experiment).items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*70)


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 创建默认配置
    config = Config()
    
    # 修改配置
    config.model.num_classes = 11
    config.training.epochs = 100
    config.experiment.experiment_name = 'my_experiment'
    
    # 打印配置
    config.print_config()
    
    # 保存配置
    os.makedirs('configs', exist_ok=True)
    config.save_yaml('configs/default_config.yaml')
    config.save_json('configs/default_config.json')
    
    # 加载配置
    loaded_config = Config.from_yaml('configs/default_config.yaml')
    loaded_config.print_config()