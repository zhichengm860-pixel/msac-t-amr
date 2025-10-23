#!/usr/bin/env python3
"""
超参数优化和敏感性分析
使用Optuna进行自动超参数搜索，包括学习率、批大小、模型架构参数等
"""

import os
import json
import torch
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
from src.models.improved_msac_t import ImprovedMSAC_T
from src.training.improved_trainer import ImprovedTrainer, CombinedLoss
from src.utils.experiment_tracker import ExperimentTracker

class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, 
                 n_trials: int = 50,
                 timeout: int = 3600,  # 1小时
                 device: str = 'cpu',
                 experiment_name: str = None):
        """
        初始化超参数优化器
        
        Args:
            n_trials: 试验次数
            timeout: 超时时间（秒）
            device: 设备类型
            experiment_name: 实验名称
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.device = device
        self.experiment_name = experiment_name or f"hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建实验目录
        self.experiment_dir = f"experiments/{self.experiment_name}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 初始化实验跟踪器
        self.tracker = ExperimentTracker(self.experiment_dir)
        
        # 存储最佳参数和结果
        self.best_params = None
        self.best_value = None
        self.optimization_history = []
        
    def create_mock_data(self, batch_size: int, signal_length: int = 1024) -> Tuple[torch.utils.data.DataLoader, ...]:
        """创建模拟数据"""
        # 生成模拟数据
        n_samples = 1000
        n_classes = 11
        
        # I/Q数据 (复数信号)
        signals = torch.randn(n_samples, 2, signal_length)  # [N, 2, L]
        labels = torch.randint(0, n_classes, (n_samples,))
        
        # 创建数据集
        dataset = torch.utils.data.TensorDataset(signals, labels)
        
        # 分割数据集
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna目标函数"""
        try:
            # 建议超参数
            params = self.suggest_hyperparameters(trial)
            
            # 创建模型
            model = ImprovedMSAC_T(
                input_channels=2,
                num_classes=11,
                base_channels=params['base_channels'],
                num_transformer_blocks=params['num_transformer_blocks'],
                num_heads=params['num_heads'],
                dropout_rate=params['dropout_rate']
            ).to(self.device)
            
            # 创建数据加载器
            train_loader, val_loader, test_loader = self.create_mock_data(
                batch_size=params['batch_size'],
                signal_length=params['signal_length']
            )
            
            # 创建训练器
            trainer = ImprovedTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=self.device,
                experiment_dir=self.experiment_dir
            )
            
            # 设置优化器和损失函数
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
            
            criterion = CombinedLoss(
                focal_alpha=params['focal_alpha'],
                focal_gamma=params['focal_gamma'],
                label_smoothing=params['label_smoothing']
            )
            
            # 训练模型（较少的epoch用于快速评估）
            max_epochs = 5  # 快速评估
            best_val_acc = 0.0
            
            for epoch in range(max_epochs):
                # 训练一个epoch
                train_loss, train_acc = trainer.train_epoch(
                    optimizer, criterion, epoch
                )
                
                # 验证
                val_loss, val_acc = trainer.validate(criterion)
                
                # 更新最佳验证准确率
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                # 报告中间结果给Optuna
                trial.report(val_acc, epoch)
                
                # 检查是否应该剪枝
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # 记录试验结果
            self.optimization_history.append({
                'trial_number': trial.number,
                'params': params,
                'best_val_acc': best_val_acc,
                'status': 'completed'
            })
            
            return best_val_acc
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            # 记录失败的试验
            self.optimization_history.append({
                'trial_number': trial.number,
                'params': params if 'params' in locals() else {},
                'best_val_acc': 0.0,
                'status': 'failed',
                'error': str(e)
            })
            return 0.0
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """建议超参数"""
        return {
            # 模型架构参数
            'base_channels': trial.suggest_categorical('base_channels', [16, 32, 48, 64]),
            'num_transformer_blocks': trial.suggest_int('num_transformer_blocks', 1, 4),
            'num_heads': trial.suggest_categorical('num_heads', [2, 4, 6, 8]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            
            # 训练参数
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 48, 64]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            
            # 损失函数参数
            'focal_alpha': trial.suggest_float('focal_alpha', 0.1, 0.9),
            'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0),
            'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
            
            # 数据参数
            'signal_length': trial.suggest_categorical('signal_length', [512, 768, 1024])
        }
    
    def optimize(self) -> optuna.Study:
        """执行超参数优化"""
        print(f"开始超参数优化: {self.experiment_name}")
        print(f"试验次数: {self.n_trials}")
        print(f"超时时间: {self.timeout}秒")
        print(f"设备: {self.device}")
        print("="*60)
        
        # 创建研究
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=2
            ),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # 执行优化
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[self._progress_callback]
        )
        
        # 保存最佳参数
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        print("\n" + "="*60)
        print("超参数优化完成!")
        print(f"最佳验证准确率: {self.best_value:.4f}")
        print(f"最佳参数: {self.best_params}")
        
        # 保存结果
        self.save_results(study)
        
        return study
    
    def _progress_callback(self, study: optuna.Study, trial: optuna.Trial):
        """进度回调函数"""
        if trial.number % 5 == 0:
            print(f"Trial {trial.number}: Best value = {study.best_value:.4f}")
    
    def save_results(self, study: optuna.Study):
        """保存优化结果"""
        # 保存最佳参数
        best_params_file = os.path.join(self.experiment_dir, 'best_hyperparameters.json')
        with open(best_params_file, 'w', encoding='utf-8') as f:
            json.dump({
                'best_params': self.best_params,
                'best_value': self.best_value,
                'n_trials': len(study.trials)
            }, f, indent=2, ensure_ascii=False)
        
        # 保存优化历史
        history_file = os.path.join(self.experiment_dir, 'optimization_history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.optimization_history, f, indent=2, ensure_ascii=False)
        
        # 保存试验数据框
        df = study.trials_dataframe()
        df.to_csv(os.path.join(self.experiment_dir, 'trials_dataframe.csv'), index=False)
        
        print(f"结果已保存到: {self.experiment_dir}")
    
    def analyze_sensitivity(self, study: optuna.Study):
        """敏感性分析"""
        print("\n执行敏感性分析...")
        
        # 创建分析目录
        analysis_dir = os.path.join(self.experiment_dir, 'sensitivity_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 1. 参数重要性分析（添加错误处理）
        importance = {}
        try:
            if len(study.trials) >= 10:  # 需要足够的试验次数
                importance = optuna.importance.get_param_importances(study)
            else:
                print(f"警告: 试验次数不足 ({len(study.trials)} < 10)，跳过参数重要性分析")
                # 使用简单的方差分析作为备用
                importance = self._calculate_simple_importance(study)
        except Exception as e:
            print(f"警告: 参数重要性分析失败: {e}")
            # 使用简单的方差分析作为备用
            importance = self._calculate_simple_importance(study)
        
        # 绘制参数重要性图
        plt.figure(figsize=(12, 8))
        params = list(importance.keys())
        values = list(importance.values())
        
        plt.barh(params, values)
        plt.xlabel('重要性')
        plt.title('超参数重要性分析')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'parameter_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 优化历史图
        plt.figure(figsize=(12, 6))
        
        # 绘制目标值历史
        plt.subplot(1, 2, 1)
        values = [trial.value for trial in study.trials if trial.value is not None]
        plt.plot(values, 'b-', alpha=0.7)
        plt.xlabel('试验次数')
        plt.ylabel('验证准确率')
        plt.title('优化历史')
        plt.grid(True, alpha=0.3)
        
        # 绘制最佳值历史
        plt.subplot(1, 2, 2)
        best_values = []
        best_so_far = 0
        for trial in study.trials:
            if trial.value is not None and trial.value > best_so_far:
                best_so_far = trial.value
            best_values.append(best_so_far)
        
        plt.plot(best_values, 'r-', linewidth=2)
        plt.xlabel('试验次数')
        plt.ylabel('最佳验证准确率')
        plt.title('最佳值历史')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'optimization_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 参数分布分析
        self._plot_parameter_distributions(study, analysis_dir)
        
        # 4. 相关性分析
        self._plot_parameter_correlations(study, analysis_dir)
        
        # 保存重要性分析结果
        importance_file = os.path.join(analysis_dir, 'parameter_importance.json')
        with open(importance_file, 'w', encoding='utf-8') as f:
            json.dump(importance, f, indent=2, ensure_ascii=False)
        
        print(f"敏感性分析完成，结果保存到: {analysis_dir}")
        
        return importance
    
    def _calculate_simple_importance(self, study: optuna.Study):
        """计算简单的参数重要性（基于方差）"""
        import pandas as pd
        
        # 获取所有试验的参数和值
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data = trial.params.copy()
                trial_data['value'] = trial.value
                trials_data.append(trial_data)
        
        if len(trials_data) < 2:
            return {}
        
        df = pd.DataFrame(trials_data)
        
        # 计算每个参数与目标值的相关性
        importance = {}
        for param in df.columns:
            if param != 'value':
                try:
                    # 计算相关系数的绝对值作为重要性
                    corr = abs(df[param].corr(df['value']))
                    if not pd.isna(corr):
                        importance[param] = corr
                except:
                    importance[param] = 0.0
        
        return importance
    
    def _plot_parameter_distributions(self, study: optuna.Study, analysis_dir: str):
        """绘制参数分布图"""
        df = study.trials_dataframe()
        
        # 获取数值型参数
        numeric_params = []
        for col in df.columns:
            if col.startswith('params_') and df[col].dtype in ['float64', 'int64']:
                numeric_params.append(col)
        
        if not numeric_params:
            return
        
        # 绘制参数分布
        n_params = len(numeric_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        plt.figure(figsize=(5*n_cols, 4*n_rows))
        
        for i, param in enumerate(numeric_params):
            plt.subplot(n_rows, n_cols, i+1)
            
            # 根据目标值着色
            scatter = plt.scatter(
                df[param], 
                df['value'], 
                c=df['value'], 
                cmap='viridis', 
                alpha=0.6
            )
            
            plt.xlabel(param.replace('params_', ''))
            plt.ylabel('验证准确率')
            plt.title(f'{param.replace("params_", "")} vs 性能')
            plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'parameter_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_correlations(self, study: optuna.Study, analysis_dir: str):
        """绘制参数相关性图"""
        df = study.trials_dataframe()
        
        # 获取数值型参数
        numeric_cols = []
        for col in df.columns:
            if col.startswith('params_') and df[col].dtype in ['float64', 'int64']:
                numeric_cols.append(col)
        
        if len(numeric_cols) < 2:
            return
        
        # 添加目标值
        numeric_cols.append('value')
        
        # 计算相关性矩阵
        corr_matrix = df[numeric_cols].corr()
        
        # 绘制相关性热图
        plt.figure(figsize=(12, 10))
        
        # 创建掩码来隐藏上三角
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f'
        )
        
        plt.title('超参数相关性分析')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'parameter_correlations.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, study: optuna.Study, importance: Dict[str, float]):
        """生成优化报告"""
        report_file = os.path.join(self.experiment_dir, 'optimization_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 超参数优化报告\n\n")
            f.write(f"**实验名称**: {self.experiment_name}\n")
            f.write(f"**优化时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**试验次数**: {len(study.trials)}\n")
            f.write(f"**设备**: {self.device}\n\n")
            
            f.write(f"## 最佳结果\n\n")
            f.write(f"**最佳验证准确率**: {self.best_value:.4f}\n\n")
            f.write(f"**最佳超参数**:\n")
            for param, value in self.best_params.items():
                f.write(f"- {param}: {value}\n")
            f.write("\n")
            
            f.write(f"## 参数重要性\n\n")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {param}: {imp:.4f}\n")
            f.write("\n")
            
            f.write(f"## 优化统计\n\n")
            completed_trials = [t for t in study.trials if t.value is not None]
            f.write(f"- 完成的试验: {len(completed_trials)}\n")
            f.write(f"- 失败的试验: {len(study.trials) - len(completed_trials)}\n")
            f.write(f"- 平均性能: {np.mean([t.value for t in completed_trials]):.4f}\n")
            f.write(f"- 性能标准差: {np.std([t.value for t in completed_trials]):.4f}\n")
        
        print(f"优化报告已生成: {report_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='超参数优化')
    parser.add_argument('--n_trials', type=int, default=30, help='试验次数')
    parser.add_argument('--timeout', type=int, default=1800, help='超时时间（秒）')
    parser.add_argument('--device', type=str, default='cpu', help='设备类型')
    parser.add_argument('--experiment_name', type=str, default=None, help='实验名称')
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = HyperparameterOptimizer(
        n_trials=args.n_trials,
        timeout=args.timeout,
        device=args.device,
        experiment_name=args.experiment_name
    )
    
    # 执行优化
    study = optimizer.optimize()
    
    # 敏感性分析
    importance = optimizer.analyze_sensitivity(study)
    
    # 生成报告
    optimizer.generate_report(study, importance)
    
    print("\n🎉 超参数优化完成!")
    print(f"📁 结果保存在: {optimizer.experiment_dir}")
    print(f"🏆 最佳验证准确率: {optimizer.best_value:.4f}")


if __name__ == "__main__":
    main()