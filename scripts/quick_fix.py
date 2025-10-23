"""
quick_fix.py - 快速修复当前问题
解决张量维度不匹配和依赖缺失问题
"""

import torch
import numpy as np
import subprocess
import sys

def install_thop():
    """安装thop库"""
    try:
        import thop
        print("✓ thop is already installed")
        return True
    except ImportError:
        print("Installing thop...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "thop"])
            print("✓ thop installed successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to install thop: {e}")
            return False

def test_data_augmentation():
    """测试修复后的数据增强函数"""
    print("Testing data augmentation functions...")
    
    # 导入修复后的数据增强类
    from src.training.improved_trainer import AdvancedSignalAugmentation
    
    # 创建测试信号
    signal = torch.randn(2, 1024)  # [I/Q, length]
    print(f"Original signal shape: {signal.shape}")
    
    # 测试每个增强函数
    try:
        aug_signal = AdvancedSignalAugmentation.awgn_noise(signal)
        print(f"✓ AWGN noise: {aug_signal.shape}")
        assert aug_signal.shape == signal.shape, f"Shape mismatch: {aug_signal.shape} vs {signal.shape}"
    except Exception as e:
        print(f"✗ AWGN noise failed: {e}")
        return False
    
    try:
        aug_signal = AdvancedSignalAugmentation.frequency_shift(signal)
        print(f"✓ Frequency shift: {aug_signal.shape}")
        assert aug_signal.shape == signal.shape, f"Shape mismatch: {aug_signal.shape} vs {signal.shape}"
    except Exception as e:
        print(f"✗ Frequency shift failed: {e}")
        return False
    
    try:
        aug_signal = AdvancedSignalAugmentation.time_shift(signal)
        print(f"✓ Time shift: {aug_signal.shape}")
        assert aug_signal.shape == signal.shape, f"Shape mismatch: {aug_signal.shape} vs {signal.shape}"
    except Exception as e:
        print(f"✗ Time shift failed: {e}")
        return False
    
    try:
        aug_signal = AdvancedSignalAugmentation.amplitude_scaling(signal)
        print(f"✓ Amplitude scaling: {aug_signal.shape}")
        assert aug_signal.shape == signal.shape, f"Shape mismatch: {aug_signal.shape} vs {signal.shape}"
    except Exception as e:
        print(f"✗ Amplitude scaling failed: {e}")
        return False
    
    try:
        aug_signal = AdvancedSignalAugmentation.phase_rotation(signal)
        print(f"✓ Phase rotation: {aug_signal.shape}")
        assert aug_signal.shape == signal.shape, f"Shape mismatch: {aug_signal.shape} vs {signal.shape}"
    except Exception as e:
        print(f"✗ Phase rotation failed: {e}")
        return False
    
    # 测试组合增强
    try:
        aug_signal = AdvancedSignalAugmentation.apply_augmentation(signal, augment_prob=1.0)
        print(f"✓ Combined augmentation: {aug_signal.shape}")
        assert aug_signal.shape == signal.shape, f"Shape mismatch: {aug_signal.shape} vs {signal.shape}"
    except Exception as e:
        print(f"✗ Combined augmentation failed: {e}")
        return False
    
    return True

def test_model_creation():
    """测试模型创建"""
    print("Testing model creation...")
    
    try:
        from src.models.improved_msac_t import ImprovedMSAC_T
        
        model = ImprovedMSAC_T(
            num_classes=11,
            base_channels=32,  # 减少通道数以加快测试
            num_transformer_blocks=2,
            num_heads=4,
            dropout=0.1
        )
        
        # 测试前向传播
        x = torch.randn(2, 2, 1024)
        snr = torch.randn(2) * 20
        
        with torch.no_grad():
            outputs = model(x, snr)
        
        print(f"✓ Model forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output logits shape: {outputs['logits'].shape}")
        print(f"  SNR prediction shape: {outputs['snr_pred'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_training_step():
    """测试训练步骤"""
    print("Testing training step...")
    
    try:
        from src.models.improved_msac_t import ImprovedMSAC_T
        from src.training.improved_trainer import CombinedLoss
        
        # 创建模型和损失函数
        model = ImprovedMSAC_T(
            num_classes=11,
            base_channels=32,
            num_transformer_blocks=2,
            num_heads=4,
            dropout=0.1
        )
        
        criterion = CombinedLoss()
        
        # 创建测试数据
        x = torch.randn(4, 2, 1024)
        labels = torch.randint(0, 11, (4,))
        snr = torch.randn(4) * 20
        
        # 前向传播
        outputs = model(x, snr)
        
        # 计算损失
        loss, loss_dict = criterion(outputs, labels, snr)
        
        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Loss components: {loss_dict}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        return False

def run_quick_test():
    """运行快速测试"""
    print("Running quick test with fixed signal length...")
    
    try:
        # 测试改进的训练脚本
        from torch.utils.data import DataLoader, TensorDataset
        
        # 创建固定长度的测试数据
        num_samples = 32
        signal_length = 1024
        signals = torch.randn(num_samples, 2, signal_length)
        labels = torch.randint(0, 11, (num_samples,))
        snr = torch.randn(num_samples) * 20
        
        dataset = TensorDataset(signals, labels, snr)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        print(f"✓ Created test dataset with {num_samples} samples")
        print(f"  Signal shape: {signals.shape}")
        print(f"  Signal length: {signal_length}")
        
        # 测试数据加载
        for batch in dataloader:
            signals_batch, labels_batch, snr_batch = batch
            print(f"✓ Batch loaded successfully")
            print(f"  Batch signals shape: {signals_batch.shape}")
            print(f"  Batch labels shape: {labels_batch.shape}")
            print(f"  Batch SNR shape: {snr_batch.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        return False

def main():
    """主修复函数"""
    print("="*60)
    print("QUICK FIX FOR CURRENT ISSUES")
    print("="*60)
    
    success_count = 0
    total_tests = 5
    
    # 1. 安装thop
    if install_thop():
        success_count += 1
    
    # 2. 测试数据增强
    if test_data_augmentation():
        success_count += 1
    
    # 3. 测试模型创建
    if test_model_creation():
        success_count += 1
    
    # 4. 测试训练步骤
    if test_training_step():
        success_count += 1
    
    # 5. 运行快速测试
    if run_quick_test():
        success_count += 1
    
    print("\n" + "="*60)
    print("QUICK FIX SUMMARY")
    print("="*60)
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All tests passed! The issues have been fixed.")
        print("\nYou can now run:")
        print("1. python test_improvements.py")
        print("2. python run_improved_training.py --epochs 10")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting tips:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that the src module can be imported correctly")
        print("3. Ensure signal lengths are consistent (1024)")

if __name__ == "__main__":
    main()