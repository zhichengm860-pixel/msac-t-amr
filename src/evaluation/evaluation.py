"""
evaluation.py - 完整的评估系统
包含：
1. SNR特定准确率分析
2. 计算效率评估（FLOPs、参数量、推理时间）
3. 消融实验框架
4. 详细的可视化工具
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
from itertools import cycle
import time
from tqdm import tqdm
import pandas as pd
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. FLOPs calculation will be skipped.")


# ==================== 评估指标计算 ====================

class MetricsCalculator:
    """评估指标计算器"""
    
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """计算准确率"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def calculate_per_class_accuracy(y_true, y_pred, num_classes):
        """计算每个类别的准确率"""
        per_class_acc = {}
        for i in range(num_classes):
            mask = y_true == i
            if mask.sum() > 0:
                per_class_acc[i] = np.mean(y_pred[mask] == y_true[mask])
            else:
                per_class_acc[i] = 0.0
        return per_class_acc
    
    @staticmethod
    def calculate_per_snr_accuracy(y_true, y_pred, snrs, snr_values):
        """计算每个SNR的准确率"""
        per_snr_acc = {}
        for snr in snr_values:
            mask = snrs == snr
            if mask.sum() > 0:
                per_snr_acc[snr] = np.mean(y_pred[mask] == y_true[mask])
            else:
                per_snr_acc[snr] = 0.0
        return per_snr_acc
    
    @staticmethod
    def calculate_confusion_matrix(y_true, y_pred, num_classes):
        """计算混淆矩阵"""
        return confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    @staticmethod
    def calculate_f1_score(y_true, y_pred, average='weighted'):
        """计算F1分数"""
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    
    @staticmethod
    def calculate_precision_recall(y_true, y_pred, average='weighted'):
        """计算精确率和召回率"""
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        return precision, recall


# ==================== 模型评估器 ====================

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, device='cuda', mod_types=None):
        self.model = model.to(device)
        self.device = device
        self.mod_types = mod_types
        
    def evaluate(self, dataloader, return_predictions=False):
        """
        评估模型
        返回：准确率、损失、预测结果（可选）
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_snrs = []
        all_probs = []
        total_loss = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for signals, labels, snrs in tqdm(dataloader, desc='Evaluating'):
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                snrs = snrs.to(self.device)
                
                outputs = self.model(signals, snrs)
                loss = criterion(outputs['logits'], labels)
                
                probs = torch.softmax(outputs['logits'], dim=1)
                preds = torch.argmax(outputs['logits'], dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_snrs.extend(snrs.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                total_loss += loss.item()
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_snrs = np.array(all_snrs)
        all_probs = np.array(all_probs)
        
        # 计算指标
        accuracy = MetricsCalculator.calculate_accuracy(all_labels, all_preds)
        avg_loss = total_loss / len(dataloader)
        
        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'predictions': all_preds if return_predictions else None,
            'labels': all_labels if return_predictions else None,
            'snrs': all_snrs if return_predictions else None,
            'probabilities': all_probs if return_predictions else None
        }
        
        return results
    
    def detailed_evaluation(self, dataloader):
        """详细评估"""
        print("\n" + "="*70)
        print("DETAILED EVALUATION REPORT")
        print("="*70)
        
        # 获取预测结果
        results = self.evaluate(dataloader, return_predictions=True)
        
        y_true = results['labels']
        y_pred = results['predictions']
        snrs = results['snrs']
        probs = results['probabilities']
        
        num_classes = len(np.unique(y_true))
        
        # 1. 整体准确率
        print(f"\n1. Overall Accuracy: {results['accuracy']:.4f}")
        print(f"   Overall Loss: {results['loss']:.4f}")
        
        # 2. F1, Precision, Recall
        f1 = MetricsCalculator.calculate_f1_score(y_true, y_pred)
        precision, recall = MetricsCalculator.calculate_precision_recall(y_true, y_pred)
        print(f"\n2. Weighted Metrics:")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        
        # 3. 每个类别的准确率
        per_class_acc = MetricsCalculator.calculate_per_class_accuracy(
            y_true, y_pred, num_classes
        )
        print(f"\n3. Per-Class Accuracy:")
        for cls_idx, acc in per_class_acc.items():
            cls_name = self.mod_types[cls_idx] if self.mod_types else f"Class {cls_idx}"
            print(f"   {cls_name}: {acc:.4f}")
        
        # 4. 每个SNR的准确率
        snr_values = sorted(np.unique(snrs))
        per_snr_acc = MetricsCalculator.calculate_per_snr_accuracy(
            y_true, y_pred, snrs, snr_values
        )
        print(f"\n4. Per-SNR Accuracy:")
        for snr, acc in sorted(per_snr_acc.items()):
            print(f"   SNR {snr:>3.0f} dB: {acc:.4f}")
        
        # 5. 分类报告
        print(f"\n5. Classification Report:")
        target_names = self.mod_types if self.mod_types else [f"Class {i}" for i in range(num_classes)]
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
        
        return {
            'overall_accuracy': results['accuracy'],
            'per_class_accuracy': per_class_acc,
            'per_snr_accuracy': per_snr_acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': MetricsCalculator.calculate_confusion_matrix(y_true, y_pred, num_classes),
            'predictions': y_pred,
            'labels': y_true,
            'snrs': snrs,
            'probabilities': probs
        }


# ==================== 计算效率评估 ====================

class EfficiencyEvaluator:
    """计算效率评估器"""
    
    @staticmethod
    def count_parameters(model):
        """计算模型参数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    @staticmethod
    def calculate_flops(model, input_size=(1, 2, 1024), device='cuda'):
        """计算FLOPs"""
        if not THOP_AVAILABLE:
            print("THOP not available, skipping FLOPs calculation")
            return None
            
        model = model.to(device)
        dummy_input = torch.randn(input_size).to(device)
        dummy_snr = torch.tensor([0.0]).to(device)
        
        try:
            flops, params = profile(model, inputs=(dummy_input, dummy_snr), verbose=False)
            flops_str, params_str = clever_format([flops, params], "%.3f")
            
            return {
                'flops': flops,
                'flops_str': flops_str,
                'params': params,
                'params_str': params_str
            }
        except Exception as e:
            print(f"FLOPs calculation failed: {e}")
            return None
    
    @staticmethod
    def measure_inference_time(model, dataloader, device='cuda', num_iterations=100):
        """测量推理时间"""
        model = model.to(device)
        model.eval()
        
        times = []
        
        with torch.no_grad():
            # 预热
            for i, (signals, _, snrs) in enumerate(dataloader):
                if i >= 10:
                    break
                signals = signals.to(device)
                snrs = snrs.to(device)
                _ = model(signals, snrs)
            
            # 实际测量
            for i, (signals, _, snrs) in enumerate(dataloader):
                if i >= num_iterations:
                    break
                
                signals = signals.to(device)
                snrs = snrs.to(device)
                
                torch.cuda.synchronize() if device == 'cuda' else None
                start_time = time.time()
                
                _ = model(signals, snrs)
                
                torch.cuda.synchronize() if device == 'cuda' else None
                end_time = time.time()
                
                times.append(end_time - start_time)
        
        times = np.array(times)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
    
    @staticmethod
    def comprehensive_efficiency_report(model, dataloader, device='cuda'):
        """综合效率报告"""
        print("\n" + "="*70)
        print("COMPUTATIONAL EFFICIENCY REPORT")
        print("="*70)
        
        # 1. 参数量
        params_info = EfficiencyEvaluator.count_parameters(model)
        print(f"\n1. Model Parameters:")
        print(f"   Total: {params_info['total']:,}")
        print(f"   Trainable: {params_info['trainable']:,}")
        print(f"   Non-trainable: {params_info['non_trainable']:,}")
        
        # 2. FLOPs
        flops_info = EfficiencyEvaluator.calculate_flops(model, device=device)
        if flops_info:
            print(f"\n2. FLOPs:")
            print(f"   {flops_info['flops_str']}")
        
        # 3. 推理时间
        time_info = EfficiencyEvaluator.measure_inference_time(model, dataloader, device=device)
        print(f"\n3. Inference Time (per batch):")
        print(f"   Mean: {time_info['mean']*1000:.2f} ms")
        print(f"   Std: {time_info['std']*1000:.2f} ms")
        print(f"   Min: {time_info['min']*1000:.2f} ms")
        print(f"   Max: {time_info['max']*1000:.2f} ms")
        print(f"   Median: {time_info['median']*1000:.2f} ms")
        
        # 4. 吞吐量
        batch_size = next(iter(dataloader))[0].size(0)
        throughput = batch_size / time_info['mean']
        print(f"\n4. Throughput:")
        print(f"   {throughput:.2f} samples/second")
        
        return {
            'parameters': params_info,
            'flops': flops_info,
            'inference_time': time_info,
            'throughput': throughput
        }


# ==================== 消融实验框架 ====================

class AblationStudy:
    """消融实验框架"""
    
    def __init__(self, base_model_class, device='cuda'):
        self.base_model_class = base_model_class
        self.device = device
        self.results = {}
    
    def run_ablation(self, config_variations, train_loader, val_loader, 
                     test_loader, epochs=50):
        """
        运行消融实验
        
        Args:
            config_variations: 字典，每个键是实验名称，值是模型配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            epochs: 训练轮数
        """
        print("\n" + "="*70)
        print("ABLATION STUDY")
        print("="*70)
        
        for exp_name, config in config_variations.items():
            print(f"\n{'='*70}")
            print(f"Experiment: {exp_name}")
            print(f"{'='*70}")
            
            # 创建模型
            model = self.base_model_class(**config).to(self.device)
            
            # 训练
            from ..training import Trainer
            trainer = Trainer(model, device=self.device)
            history = trainer.train(train_loader, val_loader, epochs=epochs)
            
            # 评估
            evaluator = ModelEvaluator(model, device=self.device)
            test_results = evaluator.evaluate(test_loader)
            
            # 保存结果
            self.results[exp_name] = {
                'config': config,
                'train_history': history,
                'test_accuracy': test_results['accuracy'],
                'test_loss': test_results['loss']
            }
            
            print(f"\nTest Accuracy: {test_results['accuracy']:.4f}")
            print(f"Test Loss: {test_results['loss']:.4f}")
        
        return self.results
    
    def compare_results(self):
        """比较消融实验结果"""
        print("\n" + "="*70)
        print("ABLATION STUDY COMPARISON")
        print("="*70)
        
        comparison_data = []
        for exp_name, result in self.results.items():
            comparison_data.append({
                'Experiment': exp_name,
                'Test Accuracy': result['test_accuracy'],
                'Test Loss': result['test_loss']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test Accuracy', ascending=False)
        
        print("\n", df.to_string(index=False))
        
        return df


# ==================== 可视化工具 ====================

class Visualizer:
    """可视化工具"""
    
    @staticmethod
    def plot_confusion_matrix(cm, class_names, save_path=None, figsize=(12, 10)):
        """绘制混淆矩阵"""
        plt.figure(figsize=figsize)
        
        # 归一化
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 绘制热图
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Accuracy'})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_per_snr_accuracy(per_snr_acc, save_path=None, figsize=(12, 6)):
        """绘制SNR-准确率曲线"""
        snrs = sorted(per_snr_acc.keys())
        accs = [per_snr_acc[snr] for snr in snrs]
        
        plt.figure(figsize=figsize)
        plt.plot(snrs, accs, marker='o', linewidth=2, markersize=8)
        plt.grid(True, alpha=0.3)
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Classification Accuracy vs SNR', fontsize=14, fontweight='bold')
        plt.ylim([0, 1.05])
        
        # 添加水平参考线
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% baseline')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SNR-Accuracy curve saved to {save_path}")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_training_history(history, save_path=None, figsize=(15, 5)):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 损失曲线
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
        axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 学习率曲线
        if 'lr' in history:
            axes[2].plot(history['lr'], linewidth=2, color='green')
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('Learning Rate', fontsize=12)
            axes[2].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_roc_curves(y_true, y_probs, class_names, save_path=None, figsize=(12, 10)):
        """绘制多类别ROC曲线"""
        n_classes = len(class_names)
        
        # 二值化标签
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # 计算每个类别的ROC曲线和AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 计算micro-average ROC曲线
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # 绘制
        plt.figure(figsize=figsize)
        
        # 绘制micro-average
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'micro-average (AUC = {roc_auc["micro"]:.2f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        # 绘制每个类别
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 
                       'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, linewidth=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Multi-class ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_model_comparison(results_dict, metric='accuracy', save_path=None, figsize=(12, 6)):
        """绘制模型对比图"""
        models = list(results_dict.keys())
        values = [results_dict[model][metric] for model in models]
        
        plt.figure(figsize=figsize)
        bars = plt.bar(models, values, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f'Model Comparison - {metric.capitalize()}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, max(values) * 1.1])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison saved to {save_path}")
        
        plt.show()
        plt.close()


# ==================== 使用示例 ====================

if __name__ == '__main__':
    """评估系统使用示例"""
    
    # 假设已经有了训练好的模型和数据
    from ..models import AMRNet
    from ..data import DatasetLoader
    
    # 加载数据
    print("Loading data...")
    # signals, labels, snrs, mod_types = DatasetLoader.load_radioml2016('RML2016.10a_dict.pkl')
    
    # 模拟数据
    num_samples = 1000
    signals = np.random.randn(num_samples, 2, 128)
    labels = np.random.randint(0, 11, num_samples)
    snrs = np.random.randint(-20, 30, num_samples)
    mod_types = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'AM-DSB', 'AM-SSB', 
                 'WBFM', 'GFSK', 'CPFSK', 'PAM4']
    
    # 创建数据加载器
    _, _, test_loader = DatasetLoader.create_dataloaders(
        signals, labels, snrs, batch_size=64
    )
    
    # 创建模型
    model = AMRNet(num_classes=11)
    
    # 1. 模型评估
    print("\n1. Model Evaluation")
    evaluator = ModelEvaluator(model, device='cuda', mod_types=mod_types)
    eval_results = evaluator.detailed_evaluation(test_loader)
    
    # 2. 效率评估
    print("\n2. Efficiency Evaluation")
    eff_results = EfficiencyEvaluator.comprehensive_efficiency_report(
        model, test_loader, device='cuda'
    )
    
    # 3. 可视化
    print("\n3. Visualization")
    
    # 混淆矩阵
    Visualizer.plot_confusion_matrix(
        eval_results['confusion_matrix'],
        mod_types,
        save_path='confusion_matrix.png'
    )
    
    # SNR-准确率曲线
    Visualizer.plot_per_snr_accuracy(
        eval_results['per_snr_accuracy'],
        save_path='snr_accuracy.png'
    )
    
    # ROC曲线
    Visualizer.plot_roc_curves(
        eval_results['labels'],
        eval_results['probabilities'],
        mod_types,
        save_path='roc_curves.png'
    )
    
    print("\nEvaluation completed!")