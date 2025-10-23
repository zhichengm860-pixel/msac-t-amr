"""
main.py - 主运行脚本
完整的训练、评估、测试流程
"""

import argparse
import torch
import numpy as np
import random
import os

from src.utils import Config, ExperimentTracker
from src.models import AMRNet, create_baseline_model
from src.data import DatasetLoader
from src.training import Trainer, PretrainTrainer, create_pretrain_dataloader
from src.evaluation import ModelEvaluator, EfficiencyEvaluator, Visualizer


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_mode(args):
    """训练模式"""
    print("\n" + "="*70)
    print("MODE: TRAINING")
    print("="*70)
    
    # 加载配置
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # 覆盖配置（如果提供了命令行参数）
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    
    # 设置随机种子
    set_seed(config.experiment.seed)
    
    # 创建实验跟踪器
    tracker = ExperimentTracker(
        config.experiment.experiment_name,
        base_dir=config.experiment.save_dir
    )
    tracker.log_config(config)
    
    # 加载数据
    print("\nLoading dataset...")
    if config.data.dataset_type == 'radioml2016':
        signals, labels, snrs, mod_types = DatasetLoader.load_radioml2016(
            config.data.dataset_path
        )
    elif config.data.dataset_type == 'radioml2018':
        signals, labels, snrs, mod_types = DatasetLoader.load_radioml2018(
            config.data.dataset_path
        )
    else:
        signals, labels, snrs, mod_types = DatasetLoader.load_custom(
            config.data.dataset_path
        )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = DatasetLoader.create_dataloaders(
        signals, labels, snrs,
        batch_size=config.training.batch_size,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        augment_train=config.training.augmentation,
        num_workers=config.data.num_workers
    )
    
    # 创建模型
    print("\nCreating model...")
    if args.model == 'amrnet':
            model = AMRNet(
                num_classes=config.model.num_classes,
                input_channels=1,
                base_channels=64,
                num_transformer_blocks=4,
                num_heads=config.model.num_heads,
                dropout=config.model.dropout
            )
    else:
        model = create_baseline_model(args.model, num_classes=config.model.num_classes)
    
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 预训练（如果需要）
    if args.pretrain:
        print("\n" + "="*70)
        print("PRETRAINING")
        print("="*70)
        
        pretrain_loader = create_pretrain_dataloader(
            torch.FloatTensor(signals),
            torch.FloatTensor(snrs),
            batch_size=config.pretrain.pretrain_batch_size
        )
        
        pretrain_trainer = PretrainTrainer(
            model,
            device=config.experiment.device,
            learning_rate=config.pretrain.pretrain_lr
        )
        
        pretrain_history = pretrain_trainer.pretrain_multi_task(
            pretrain_loader,
            epochs=config.pretrain.pretrain_epochs,
            recon_weight=config.pretrain.recon_weight,
            contrast_weight=config.pretrain.contrast_weight,
            mask_weight=config.pretrain.mask_weight
        )
        
        # 保存预训练模型
        pretrain_path = os.path.join(tracker.checkpoint_dir, 'pretrained_model.pth')
        pretrain_trainer.save_checkpoint(pretrain_path)
    
    # 训练
    print("\n" + "="*70)
    print("SUPERVISED TRAINING")
    print("="*70)
    
    trainer = Trainer(
        model,
        config,
        device=config.experiment.device,
        experiment_tracker=tracker
    )
    
    history = trainer.train(train_loader, val_loader)
    
    # 测试
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    evaluator = ModelEvaluator(
        model,
        device=config.experiment.device,
        mod_types=mod_types
    )
    
    test_results = evaluator.detailed_evaluation(test_loader)
    
    # 保存结果
    tracker.save_results(test_results, 'test_results.json')
    
    # 可视化
    if config.experiment.visualize:
        print("\nGenerating visualizations...")
        
        # 训练历史
        Visualizer.plot_training_history(history)
        
        # 混淆矩阵
        Visualizer.plot_confusion_matrix(
            test_results['confusion_matrix'],
            mod_types,
            save_path=os.path.join(tracker.plot_dir, 'confusion_matrix.png')
        )
        
        # SNR-准确率曲线
        Visualizer.plot_per_snr_accuracy(
            test_results['per_snr_accuracy'],
            save_path=os.path.join(tracker.plot_dir, 'snr_accuracy.png')
        )
        
        # ROC曲线
        Visualizer.plot_roc_curves(
            test_results['labels'],
            test_results['probabilities'],
            mod_types,
            save_path=os.path.join(tracker.plot_dir, 'roc_curves.png')
        )
    
    # 完成实验
    tracker.finish()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {tracker.experiment_dir}")
    print("="*70)


def evaluate_mode(args):
    """评估模式"""
    print("\n" + "="*70)
    print("MODE: EVALUATION")
    print("="*70)
    
    # 加载配置
    config = Config.from_yaml(args.config) if args.config else Config()
    
    # 加载数据
    print("\nLoading dataset...")
    if config.data.dataset_type == 'radioml2016':
        signals, labels, snrs, mod_types = DatasetLoader.load_radioml2016(
            config.data.dataset_path
        )
    elif config.data.dataset_type == 'radioml2018':
        signals, labels, snrs, mod_types = DatasetLoader.load_radioml2018(
            config.data.dataset_path
        )
    else:
        signals, labels, snrs, mod_types = DatasetLoader.load_custom(
            config.data.dataset_path
        )
    
    _, _, test_loader = DatasetLoader.create_dataloaders(
        signals, labels, snrs,
        batch_size=config.training.batch_size
    )
    
    # 加载模型
    print(f"\nLoading model from {args.checkpoint}...")
    model = AMRNet(
        num_classes=config.model.num_classes,
        input_channels=1,
        base_channels=64,
        num_transformer_blocks=4,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout
    )
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估
    evaluator = ModelEvaluator(model, device=config.experiment.device, mod_types=mod_types)
    results = evaluator.detailed_evaluation(test_loader)
    
    # 效率评估
    if args.efficiency:
        print("\n" + "="*70)
        print("EFFICIENCY EVALUATION")
        print("="*70)
        EfficiencyEvaluator.comprehensive_efficiency_report(
            model, test_loader, device=config.experiment.device
        )
    
    print("\nEvaluation completed!")


def compare_mode(args):
    """模型对比模式"""
    print("\n" + "="*70)
    print("MODE: MODEL COMPARISON")
    print("="*70)
    
    # 加载配置
    config = Config.from_yaml(args.config) if args.config else Config()
    
    # 加载数据
    print("\nLoading dataset...")
    if config.data.dataset_type == 'radioml2016':
        signals, labels, snrs, mod_types = DatasetLoader.load_radioml2016(
            config.data.dataset_path
        )
    else:
        # 模拟数据
        num_samples = 10000
        signals = np.random.randn(num_samples, 2, 128)
        labels = np.random.randint(0, 11, num_samples)
        snrs = np.random.randint(-20, 30, num_samples)
        mod_types = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'AM-DSB', 
                     'AM-SSB', 'WBFM', 'GFSK', 'CPFSK', 'PAM4']
    
    train_loader, val_loader, test_loader = DatasetLoader.create_dataloaders(
        signals, labels, snrs,
        batch_size=config.training.batch_size
    )
    
    # 要对比的模型
    models_to_compare = ['amrnet', 'resnet', 'cldnn', 'mcformer']
    
    results = {}
    
    for model_name in models_to_compare:
        print(f"\n{'='*70}")
        print(f"Training and Evaluating: {model_name.upper()}")
        print(f"{'='*70}")
        
        # 创建模型
        if model_name == 'amrnet':
            model = AMRNet(
                num_classes=config.model.num_classes,
                input_channels=1,
                base_channels=64,
                num_transformer_blocks=4,
                num_heads=config.model.num_heads,
                dropout=config.model.dropout
            )
        else:
            model = create_baseline_model(model_name, num_classes=config.model.num_classes)
        
        # 训练
        trainer = Trainer(model, config, device=config.experiment.device)
        trainer.train(train_loader, val_loader, epochs=20)  # 快速训练
        
        # 评估
        evaluator = ModelEvaluator(model, device=config.experiment.device)
        eval_results = evaluator.evaluate(test_loader)
        
        # 效率评估
        eff_results = EfficiencyEvaluator.comprehensive_efficiency_report(
            model, test_loader, device=config.experiment.device
        )
        
        results[model_name] = {
            'accuracy': eval_results['accuracy'],
            'parameters': eff_results['parameters']['total'],
            'throughput': eff_results['throughput']
        }
    
    # 对比可视化
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Parameters: {metrics['parameters']:,}")
        print(f"  Throughput: {metrics['throughput']:.2f} samples/s")
    
    Visualizer.plot_model_comparison(results, metric='accuracy')


def main():
    parser = argparse.ArgumentParser(description='AMRNet - Automatic Modulation Recognition')
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'compare'],
                       help='运行模式')
    
    parser.add_argument('--model', type=str, default='amrnet',
                       choices=['amrnet', 'resnet', 'cldnn', 'mcformer'],
                       help='模型类型')
    
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='检查点路径（用于评估）')
    
    parser.add_argument('--pretrain', action='store_true',
                       help='是否进行预训练')
    
    parser.add_argument('--efficiency', action='store_true',
                       help='是否进行效率评估')
    
    # 训练参数（覆盖配置文件）
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批次大小')
    
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率')
    
    args = parser.parse_args()
    
    # 根据模式执行
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            print("Error: --checkpoint is required for evaluation mode")
            return
        evaluate_mode(args)
    elif args.mode == 'compare':
        compare_mode(args)


if __name__ == '__main__':
    main()