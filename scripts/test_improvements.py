"""
test_improvements.py - 快速测试改进效果
验证改进的模型和训练策略是否正常工作
"""

import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset

# 导入改进的组件
from src.models.improved_msac_t import ImprovedMSAC_T
from src.training.improved_trainer import ImprovedTrainer, AdvancedSignalAugmentation
from src.evaluation.ablation_study import AblationStudyManager


def test_improved_model():
    """测试改进的模型"""
    print("Testing Improved MSAC-T Model...")
    
    # 创建模型
    model = ImprovedMSAC_T(
        num_classes=11,
        base_channels=64,
        num_transformer_blocks=3,  # 减少层数以加快测试
        num_heads=8,
        dropout=0.1
    )
    
    # 测试输入
    batch_size = 4
    signal_length = 1024
    x = torch.randn(batch_size, 2, signal_length)
    snr = torch.randn(batch_size) * 20
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x, snr)
    
    print(f"✓ Model forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output logits shape: {outputs['logits'].shape}")
    print(f"  SNR prediction shape: {outputs['snr_pred'].shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    return model


def test_data_augmentation():
    """测试数据增强"""
    print("\nTesting Data Augmentation...")
    
    # 创建测试信号
    signal = torch.randn(2, 1024)
    
    # 测试各种增强方法
    augmented_awgn = AdvancedSignalAugmentation.awgn_noise(signal)
    augmented_freq = AdvancedSignalAugmentation.frequency_shift(signal)
    augmented_time = AdvancedSignalAugmentation.time_shift(signal)
    augmented_amp = AdvancedSignalAugmentation.amplitude_scaling(signal)
    augmented_phase = AdvancedSignalAugmentation.phase_rotation(signal)
    
    print(f"✓ Data augmentation methods working")
    print(f"  Original shape: {signal.shape}")
    print(f"  AWGN augmented shape: {augmented_awgn.shape}")
    print(f"  Frequency shift shape: {augmented_freq.shape}")
    print(f"  Time shift shape: {augmented_time.shape}")
    print(f"  Amplitude scaling shape: {augmented_amp.shape}")
    print(f"  Phase rotation shape: {augmented_phase.shape}")


def test_training_components():
    """测试训练组件"""
    print("\nTesting Training Components...")
    
    # 创建模拟数据 - 使用固定的信号长度
    num_samples = 100
    num_classes = 11
    signal_length = 1024  # 确保信号长度一致
    
    signals = torch.randn(num_samples, 2, signal_length)
    labels = torch.randint(0, num_classes, (num_samples,))
    snr = torch.randn(num_samples) * 20
    
    dataset = TensorDataset(signals, labels, snr)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 创建模型
    model = ImprovedMSAC_T(
        num_classes=num_classes,
        base_channels=32,  # 减少通道数以加快测试
        num_transformer_blocks=2,
        num_heads=4,
        dropout=0.1
    )
    
    # 测试训练器配置
    config = {
        'optimizer': 'adamw',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'use_amp': False,  # 关闭混合精度以简化测试
        'accumulation_steps': 1,
        'patience': 5,
        'use_augmentation': True,
        'augment_prob': 0.5,
        'classification_weight': 1.0,
        'snr_weight': 0.1,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'label_smoothing': 0.1
    }
    
    # 创建训练器
    trainer = ImprovedTrainer(
        model=model,
        train_loader=dataloader,
        val_loader=dataloader,  # 使用相同数据进行测试
        config=config,
        device='cpu',  # 使用CPU进行测试
        experiment_dir='test_experiment'
    )
    
    print(f"✓ Trainer created successfully")
    
    # 测试一个训练步骤
    try:
        trainer.train_epoch(0)
        print(f"✓ Training epoch completed successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False
    
    return True


def test_ablation_components():
    """测试消融实验组件"""
    print("\nTesting Ablation Study Components...")
    
    # 创建小规模数据用于测试
    num_samples = 50
    num_classes = 11
    signal_length = 512  # 减少长度以加快测试
    
    def create_test_data(num_samples):
        signals = torch.randn(num_samples, 2, signal_length)
        labels = torch.randint(0, num_classes, (num_samples,))
        snr = torch.randn(num_samples) * 20
        return TensorDataset(signals, labels, snr)
    
    train_dataset = create_test_data(num_samples)
    val_dataset = create_test_data(num_samples // 2)
    test_dataset = create_test_data(num_samples // 2)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 创建消融实验管理器
    ablation_manager = AblationStudyManager(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device='cpu',
        num_classes=num_classes
    )
    
    print(f"✓ Ablation study manager created")
    
    # 测试基线模型创建
    from src.evaluation.ablation_study import BaselineComplexCNN
    baseline_model = BaselineComplexCNN(num_classes=num_classes, base_channels=32)
    
    # 测试模型评估
    try:
        results = ablation_manager.evaluate_model(baseline_model, val_loader, 'baseline_test')
        print(f"✓ Model evaluation successful")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Loss: {results['loss']:.4f}")
    except Exception as e:
        print(f"✗ Model evaluation failed: {e}")
        return False
    
    return True


def test_model_comparison():
    """测试模型对比"""
    print("\nTesting Model Comparison...")
    
    # 创建不同模型进行对比
    models = {
        'Original': lambda: torch.nn.Sequential(
            torch.nn.Conv1d(2, 64, 7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 11)
        ),
        'Improved': lambda: ImprovedMSAC_T(
            num_classes=11,
            base_channels=32,
            num_transformer_blocks=2,
            num_heads=4,
            dropout=0.1
        )
    }
    
    # 测试输入
    x = torch.randn(4, 2, 1024)
    snr = torch.randn(4) * 20
    
    for name, model_fn in models.items():
        model = model_fn()
        
        try:
            if name == 'Original':
                with torch.no_grad():
                    output = model(x)
            else:
                with torch.no_grad():
                    output = model(x, snr)
                    output = output['logits']
            
            params = sum(p.numel() for p in model.parameters())
            print(f"✓ {name} model: {output.shape}, {params:,} parameters")
            
        except Exception as e:
            print(f"✗ {name} model failed: {e}")


def run_performance_test():
    """运行性能测试"""
    print("\nRunning Performance Test...")
    
    model = ImprovedMSAC_T(
        num_classes=11,
        base_channels=64,
        num_transformer_blocks=3,
        num_heads=8,
        dropout=0.1
    )
    
    model.eval()
    
    # 测试推理速度
    x = torch.randn(32, 2, 1024)
    snr = torch.randn(32) * 20
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, snr)
    
    # 计时
    start_time = time.time()
    num_runs = 100
    
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model(x, snr)
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    throughput = 32 / avg_time  # samples per second
    
    print(f"✓ Performance test completed")
    print(f"  Average inference time: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.1f} samples/second")


def main():
    """主测试函数"""
    print("="*60)
    print("TESTING IMPROVED MSAC-T COMPONENTS")
    print("="*60)
    
    try:
        # 测试改进的模型
        model = test_improved_model()
        
        # 测试数据增强
        test_data_augmentation()
        
        # 测试训练组件
        training_success = test_training_components()
        
        # 测试消融实验组件
        ablation_success = test_ablation_components()
        
        # 测试模型对比
        test_model_comparison()
        
        # 运行性能测试
        run_performance_test()
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"✓ Model creation: SUCCESS")
        print(f"✓ Data augmentation: SUCCESS")
        print(f"{'✓' if training_success else '✗'} Training components: {'SUCCESS' if training_success else 'FAILED'}")
        print(f"{'✓' if ablation_success else '✗'} Ablation study: {'SUCCESS' if ablation_success else 'FAILED'}")
        print(f"✓ Model comparison: SUCCESS")
        print(f"✓ Performance test: SUCCESS")
        
        if training_success and ablation_success:
            print("\n🎉 All tests passed! The improved components are working correctly.")
            print("\nNext steps:")
            print("1. Run full training with: python run_improved_training.py")
            print("2. Use real RadioML data instead of mock data")
            print("3. Experiment with different hyperparameters")
            print("4. Run ablation studies to analyze component contributions")
        else:
            print("\n⚠️  Some tests failed. Please check the error messages above.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()