#!/usr/bin/env python3
"""
快速测试超参数优化功能
"""

import os
import sys
import torch
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hyperparameter_optimization import HyperparameterOptimizer

def test_hyperparameter_optimization():
    """测试超参数优化功能"""
    print("="*60)
    print("测试超参数优化功能")
    print("="*60)
    
    # 创建优化器（使用较少的试验次数进行快速测试）
    optimizer = HyperparameterOptimizer(
        n_trials=5,  # 快速测试只用5次试验
        timeout=300,  # 5分钟超时
        device='cpu',
        experiment_name=f"test_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    print(f"✓ 优化器创建成功")
    print(f"  实验名称: {optimizer.experiment_name}")
    print(f"  试验次数: {optimizer.n_trials}")
    print(f"  设备: {optimizer.device}")
    
    try:
        # 测试模拟数据创建
        train_loader, val_loader, test_loader = optimizer.create_mock_data(batch_size=32)
        print(f"✓ 模拟数据创建成功")
        print(f"  训练集批次数: {len(train_loader)}")
        print(f"  验证集批次数: {len(val_loader)}")
        print(f"  测试集批次数: {len(test_loader)}")
        
        # 测试数据形状
        for batch_idx, (signals, labels) in enumerate(train_loader):
            print(f"  数据形状: {signals.shape}, 标签形状: {labels.shape}")
            break
        
        # 执行优化
        print("\n开始执行超参数优化...")
        study = optimizer.optimize()
        
        print(f"✓ 超参数优化完成")
        print(f"  最佳验证准确率: {optimizer.best_value:.4f}")
        print(f"  最佳参数: {optimizer.best_params}")
        
        # 执行敏感性分析
        print("\n开始敏感性分析...")
        importance = optimizer.analyze_sensitivity(study)
        
        print(f"✓ 敏感性分析完成")
        print(f"  参数重要性: {importance}")
        
        # 生成报告
        optimizer.generate_report(study, importance)
        print(f"✓ 报告生成完成")
        
        # 检查生成的文件
        expected_files = [
            'best_hyperparameters.json',
            'optimization_history.json',
            'trials_dataframe.csv',
            'optimization_report.md',
            'sensitivity_analysis/parameter_importance.png',
            'sensitivity_analysis/optimization_history.png'
        ]
        
        print(f"\n检查生成的文件:")
        for file_path in expected_files:
            full_path = os.path.join(optimizer.experiment_dir, file_path)
            if os.path.exists(full_path):
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (缺失)")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_optuna_installation():
    """测试Optuna安装"""
    try:
        import optuna
        print(f"✓ Optuna已安装，版本: {optuna.__version__}")
        return True
    except ImportError:
        print("✗ Optuna未安装，请运行: pip install optuna")
        return False

def main():
    """主函数"""
    print("超参数优化功能测试")
    print("="*60)
    
    # 1. 检查依赖
    print("1. 检查依赖...")
    if not test_optuna_installation():
        return
    
    # 2. 测试超参数优化
    print("\n2. 测试超参数优化...")
    success = test_hyperparameter_optimization()
    
    # 3. 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    if success:
        print("🎉 所有测试通过!")
        print("超参数优化功能正常工作")
        print("\n下一步:")
        print("1. 运行完整优化: python hyperparameter_optimization.py --n_trials 50")
        print("2. 使用GPU加速: python hyperparameter_optimization.py --device cuda")
        print("3. 查看优化结果: experiments/*/optimization_report.md")
    else:
        print("❌ 测试失败")
        print("请检查错误信息并修复问题")

if __name__ == "__main__":
    main()