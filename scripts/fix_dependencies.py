"""
fix_dependencies.py - 修复依赖和常见问题
"""

import subprocess
import sys
import os

def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def check_and_install_dependencies():
    """检查并安装缺失的依赖"""
    print("Checking and installing dependencies...")
    
    # 必需的包
    required_packages = [
        "thop",  # 用于FLOPs计算
        "optuna",  # 用于超参数优化（可选）
        "tensorboard",  # 用于训练监控
        "seaborn",  # 用于可视化
        "scikit-learn",  # 用于评估指标
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            install_package(package)

def update_requirements():
    """更新requirements.txt文件"""
    print("Updating requirements.txt...")
    
    additional_requirements = [
        "thop>=0.0.31",
        "optuna>=3.0.0",
    ]
    
    requirements_file = "requirements.txt"
    
    # 读取现有的requirements
    existing_requirements = []
    if os.path.exists(requirements_file):
        with open(requirements_file, 'r') as f:
            existing_requirements = f.read().splitlines()
    
    # 添加新的requirements
    updated_requirements = existing_requirements.copy()
    for req in additional_requirements:
        package_name = req.split('>=')[0].split('==')[0]
        # 检查是否已经存在
        if not any(package_name in line for line in existing_requirements):
            updated_requirements.append(req)
            print(f"Added {req} to requirements.txt")
    
    # 写回文件
    with open(requirements_file, 'w') as f:
        f.write('\n'.join(updated_requirements))
    
    print("✓ Requirements.txt updated")

def fix_import_issues():
    """修复导入问题"""
    print("Checking import issues...")
    
    # 检查src模块是否可以正确导入
    try:
        import sys
        sys.path.append('.')
        from src.models.improved_msac_t import ImprovedMSAC_T
        print("✓ Model imports working correctly")
    except Exception as e:
        print(f"✗ Model import issue: {e}")
        print("Please check the src module structure")
    
    try:
        from src.training.improved_trainer import ImprovedTrainer
        print("✓ Trainer imports working correctly")
    except Exception as e:
        print(f"✗ Trainer import issue: {e}")

def create_simple_test():
    """创建简单的测试来验证修复"""
    print("Creating simple test...")
    
    test_code = '''
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
'''
    
    with open('simple_test.py', 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("✓ Created simple_test.py")

def main():
    """主修复函数"""
    print("="*60)
    print("FIXING DEPENDENCIES AND COMMON ISSUES")
    print("="*60)
    
    # 1. 检查并安装依赖
    check_and_install_dependencies()
    
    # 2. 更新requirements.txt
    update_requirements()
    
    # 3. 修复导入问题
    fix_import_issues()
    
    # 4. 创建简单测试
    create_simple_test()
    
    print("\n" + "="*60)
    print("FIX SUMMARY")
    print("="*60)
    print("✓ Dependencies checked and installed")
    print("✓ Requirements.txt updated")
    print("✓ Import issues checked")
    print("✓ Simple test created")
    
    print("\nNext steps:")
    print("1. Run: python simple_test.py")
    print("2. Run: python test_improvements.py")
    print("3. Run: python run_improved_training.py --epochs 10")

if __name__ == "__main__":
    main()