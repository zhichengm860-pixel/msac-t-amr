
import torch
import numpy as np

# 测试基本功能
def test_basic_functionality():
    print("Testing basic PyTorch functionality...")
    
    # 测试张量操作
    x = torch.randn(4, 2, 1024)
    print(f"✓ Created tensor with shape: {x.shape}")
    
    # 测试数据增强
    from src.training.improved_trainer import AdvancedSignalAugmentation
    
    # 测试每个增强函数
    signal = torch.randn(2, 1024)
    
    try:
        aug_signal = AdvancedSignalAugmentation.awgn_noise(signal)
        print(f"✓ AWGN augmentation: {aug_signal.shape}")
    except Exception as e:
        print(f"✗ AWGN augmentation failed: {e}")
    
    try:
        aug_signal = AdvancedSignalAugmentation.frequency_shift(signal)
        print(f"✓ Frequency shift: {aug_signal.shape}")
    except Exception as e:
        print(f"✗ Frequency shift failed: {e}")
    
    try:
        aug_signal = AdvancedSignalAugmentation.time_shift(signal)
        print(f"✓ Time shift: {aug_signal.shape}")
    except Exception as e:
        print(f"✗ Time shift failed: {e}")
    
    try:
        aug_signal = AdvancedSignalAugmentation.amplitude_scaling(signal)
        print(f"✓ Amplitude scaling: {aug_signal.shape}")
    except Exception as e:
        print(f"✗ Amplitude scaling failed: {e}")
    
    try:
        aug_signal = AdvancedSignalAugmentation.phase_rotation(signal)
        print(f"✓ Phase rotation: {aug_signal.shape}")
    except Exception as e:
        print(f"✗ Phase rotation failed: {e}")

if __name__ == "__main__":
    test_basic_functionality()
