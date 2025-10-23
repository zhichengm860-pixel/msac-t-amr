#!/usr/bin/env python3
"""
检查数据集文件是否存在
"""

from src.data import DatasetConfigManager
import os

def main():
    print("=== 数据集检查 ===")
    
    manager = DatasetConfigManager()
    
    # 检查所有数据集
    for dataset_name in manager.list_available_datasets():
        exists = manager.check_dataset_exists(dataset_name)
        status = "✓" if exists else "✗"
        print(f"{dataset_name}: {status}")
        
        # 显示详细信息
        info = manager.get_dataset_info(dataset_name)
        print(f"  路径: {info['path']}")
        print(f"  类别数: {info['num_classes']}")
        print(f"  信号长度: {info['signal_length']}")
        print()
    
    # 检查数据集目录结构
    print("=== 数据集目录结构 ===")
    dataset_dir = "dataset"
    if os.path.exists(dataset_dir):
        for root, dirs, files in os.walk(dataset_dir):
            level = root.replace(dataset_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print("数据集目录不存在")

if __name__ == "__main__":
    main()