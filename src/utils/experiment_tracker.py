"""
experiment_tracker.py - 实验跟踪和日志管理
包含：
1. 实验日志记录
2. TensorBoard集成
3. 模型检查点管理
4. 结果保存和加载
"""

import os
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import logging


# ==================== 日志管理器 ====================

class Logger:
    """日志管理器"""
    
    def __init__(self, log_dir, log_file='experiment.log'):
        """
        Args:
            log_dir: 日志目录
            log_file: 日志文件名
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        fh = logging.FileHandler(os.path.join(log_dir, log_file))
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def info(self, message):
        """记录信息"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录警告"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录错误"""
        self.logger.error(message)
    
    def debug(self, message):
        """记录调试信息"""
        self.logger.debug(message)


# ==================== 检查点管理器 ====================

class CheckpointManager:
    """模型检查点管理器"""
    
    def __init__(self, checkpoint_dir, max_keep=5):
        """
        Args:
            checkpoint_dir: 检查点目录
            max_keep: 最多保留的检查点数量
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_keep = max_keep
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.checkpoints = []
    
    def save_checkpoint(self, state_dict, epoch, metric_value, is_best=False):
        """
        保存检查点
        
        Args:
            state_dict: 状态字典
            epoch: 当前epoch
            metric_value: 指标值（用于排序）
            is_best: 是否是最佳模型
        """
        # 保存当前检查点
        checkpoint_name = f'checkpoint_epoch_{epoch}.pth'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        torch.save(state_dict, checkpoint_path)
        
        # 记录检查点信息
        self.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'metric': metric_value,
            'timestamp': time.time()
        })
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            shutil.copy(checkpoint_path, best_path)
            print(f"Best model saved to {best_path}")
        
        # 清理旧检查点
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_checkpoints(self):
        """清理旧检查点，只保留最新的max_keep个"""
        if len(self.checkpoints) > self.max_keep:
            # 按metric排序（降序）
            self.checkpoints.sort(key=lambda x: x['metric'], reverse=True)
            
            # 删除超出的检查点
            for ckpt in self.checkpoints[self.max_keep:]:
                if os.path.exists(ckpt['path']) and 'best_model' not in ckpt['path']:
                    os.remove(ckpt['path'])
            
            # 更新列表
            self.checkpoints = self.checkpoints[:self.max_keep]
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
    
    def load_best_checkpoint(self):
        """加载最佳检查点"""
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        return self.load_checkpoint(best_path)
    
    def get_latest_checkpoint(self):
        """获取最新的检查点路径"""
        if not self.checkpoints:
            return None
        
        latest = max(self.checkpoints, key=lambda x: x['timestamp'])
        return latest['path']


# ==================== 实验跟踪器 ====================

class ExperimentTracker:
    """实验跟踪器"""
    
    def __init__(self, experiment_name, base_dir='experiments'):
        """
        Args:
            experiment_name: 实验名称
            base_dir: 基础目录
        """
        # 创建实验目录（带时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = f"{experiment_name}_{timestamp}"
        self.experiment_dir = os.path.join(base_dir, self.experiment_name)
        
        # 创建子目录
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')
        self.result_dir = os.path.join(self.experiment_dir, 'results')
        self.plot_dir = os.path.join(self.experiment_dir, 'plots')
        
        for dir_path in [self.checkpoint_dir, self.log_dir, self.result_dir, self.plot_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 初始化组件
        self.logger = Logger(self.log_dir)
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        self.tensorboard = SummaryWriter(log_dir=self.log_dir)
        
        # 实验信息
        self.start_time = time.time()
        self.metrics_history = {
            'train': [],
            'val': [],
            'test': []
        }
        
        self.logger.info(f"Experiment '{self.experiment_name}' initialized")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def log_config(self, config):
        """记录配置"""
        config_path = os.path.join(self.experiment_dir, 'config.json')
        
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        self.logger.info("Configuration saved")
    
    def log_metrics(self, metrics, step, phase='train'):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 步骤（epoch或iteration）
            phase: 阶段（'train', 'val', 'test'）
        """
        # 记录到TensorBoard
        for key, value in metrics.items():
            self.tensorboard.add_scalar(f'{phase}/{key}', value, step)
        
        # 保存到历史
        self.metrics_history[phase].append({
            'step': step,
            **metrics
        })
        
        # 日志输出
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f"[{phase.upper()}] Step {step} - {metrics_str}")
    
    def log_model_graph(self, model, input_sample):
        """记录模型结构图"""
        try:
            self.tensorboard.add_graph(model, input_sample)
            self.logger.info("Model graph logged to TensorBoard")
        except Exception as e:
            self.logger.warning(f"Failed to log model graph: {e}")
    
    def log_images(self, tag, images, step):
        """记录图像"""
        self.tensorboard.add_images(tag, images, step)
    
    def log_histogram(self, tag, values, step):
        """记录直方图"""
        self.tensorboard.add_histogram(tag, values, step)
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, is_best=False):
        """保存检查点"""
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'metrics_history': self.metrics_history
        }
        
        metric_value = metrics.get('val_acc', metrics.get('accuracy', 0))
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            state_dict, epoch, metric_value, is_best
        )
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def save_results(self, results, filename='results.json'):
        """保存实验结果"""
        result_path = os.path.join(self.result_dir, filename)
        
        # 转换numpy类型为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_to_serializable(results)
        
        with open(result_path, 'w') as f:
            json.dump(results_serializable, f, indent=4)
        
        self.logger.info(f"Results saved to {result_path}")
    
    def save_plot(self, figure, filename):
        """保存图表"""
        plot_path = os.path.join(self.plot_dir, filename)
        figure.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plot saved to {plot_path}")
    
    def finish(self):
        """完成实验"""
        duration = time.time() - self.start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        self.logger.info(f"Experiment completed in {hours}h {minutes}m {seconds}s")
        
        # 保存最终历史
        history_path = os.path.join(self.result_dir, 'metrics_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
        
        # 关闭TensorBoard
        self.tensorboard.close()
        
        # 创建README
        self._create_readme()
    
    def _create_readme(self):
        """创建实验README"""
        readme_path = os.path.join(self.experiment_dir, 'README.md')
        
        with open(readme_path, 'w') as f:
            f.write(f"# Experiment: {self.experiment_name}\n\n")
            f.write(f"**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write(f"{self.experiment_name}/\n")
            f.write("├── checkpoints/     # Model checkpoints\n")
            f.write("├── logs/           # Logs and TensorBoard files\n")
            f.write("├── results/        # Experiment results\n")
            f.write("├── plots/          # Visualization plots\n")
            f.write("├── config.json     # Experiment configuration\n")
            f.write("└── README.md       # This file\n")
            f.write("```\n\n")
            
            f.write("## Quick Start\n\n")
            f.write("### View TensorBoard\n")
            f.write("```bash\n")
            f.write(f"tensorboard --logdir={self.log_dir}\n")
            f.write("```\n\n")
            
            f.write("### Load Best Model\n")
            f.write("```python\n")
            f.write("import torch\n")
            f.write(f"checkpoint = torch.load('{self.checkpoint_dir}/best_model.pth')\n")
            f.write("model.load_state_dict(checkpoint['model_state_dict'])\n")
            f.write("```\n")
        
        self.logger.info(f"README created at {readme_path}")


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 创建实验跟踪器
    tracker = ExperimentTracker('test_experiment')
    
    # 记录配置
    from .config import Config
    config = Config()
    tracker.log_config(config)
    
    # 模拟训练过程
    for epoch in range(10):
        # 训练指标
        train_metrics = {
            'loss': np.random.rand(),
            'accuracy': 0.5 + epoch * 0.05
        }
        tracker.log_metrics(train_metrics, epoch, phase='train')
        
        # 验证指标
        val_metrics = {
            'loss': np.random.rand(),
            'accuracy': 0.5 + epoch * 0.04
        }
        tracker.log_metrics(val_metrics, epoch, phase='val')
    
    # 完成实验
    tracker.finish()