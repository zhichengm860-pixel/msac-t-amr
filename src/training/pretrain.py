"""
pretrain.py - 自监督预训练模块
包含：
1. 信号重建预训练
2. 对比学习预训练
3. 掩码预测预训练
4. 多任务联合预训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random


# ==================== 预训练数据集 ====================

class PretrainDataset(Dataset):
    """预训练数据集，支持多种自监督任务"""
    
    def __init__(self, signals, task_type='reconstruction'):
        """
        Args:
            signals: [N, 2, length] 原始信号
            task_type: 'reconstruction', 'contrastive', 'masked'
        """
        self.signals = signals
        self.task_type = task_type
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        
        if self.task_type == 'reconstruction':
            # 重建任务：添加噪声
            noisy_signal = self.add_noise(signal)
            return noisy_signal, signal
            
        elif self.task_type == 'contrastive':
            # 对比学习：创建正负样本对
            aug1 = self.augment_signal(signal)
            aug2 = self.augment_signal(signal)
            return aug1, aug2
            
        elif self.task_type == 'masked':
            # 掩码预测：随机掩码部分信号
            masked_signal, mask = self.mask_signal(signal)
            return masked_signal, signal, mask
            
        else:
            return signal, signal
    
    def add_noise(self, signal, noise_std=0.1):
        """添加高斯噪声"""
        noise = torch.randn_like(signal) * noise_std
        return signal + noise
    
    def augment_signal(self, signal):
        """信号增强"""
        # 随机选择增强方式
        aug_type = random.choice(['noise', 'scale', 'shift', 'flip'])
        
        if aug_type == 'noise':
            return self.add_noise(signal, noise_std=0.05)
        elif aug_type == 'scale':
            scale = 0.8 + 0.4 * torch.rand(1)
            return signal * scale
        elif aug_type == 'shift':
            shift_len = random.randint(-10, 10)
            return torch.roll(signal, shift_len, dims=-1)
        elif aug_type == 'flip':
            if random.random() > 0.5:
                return torch.flip(signal, dims=[-1])
            return signal
        
        return signal
    
    def mask_signal(self, signal, mask_ratio=0.15):
        """随机掩码信号"""
        length = signal.size(-1)
        mask_len = int(length * mask_ratio)
        
        # 创建掩码
        mask = torch.zeros(length, dtype=torch.bool)
        mask_indices = torch.randperm(length)[:mask_len]
        mask[mask_indices] = True
        
        # 应用掩码
        masked_signal = signal.clone()
        masked_signal[:, mask] = 0
        
        return masked_signal, mask


# ==================== 投影头 ====================

class ProjectionHead(nn.Module):
    """对比学习投影头"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)


# ==================== 对比损失 ====================

class InfoNCELoss(nn.Module):
    """InfoNCE对比损失"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """
        Args:
            z1, z2: [batch_size, feature_dim] 投影后的特征
        """
        batch_size = z1.size(0)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # 正样本在对角线上
        labels = torch.arange(batch_size).to(z1.device)
        
        # 计算损失
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


# ==================== 预训练器 ====================

class PretrainTrainer:
    """预训练器"""
    
    def __init__(self, model, device='cuda', learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # 损失历史
        self.history = {
            'recon_loss': [],
            'contrast_loss': [],
            'mask_loss': [],
            'total_loss': []
        }
    
    def pretrain_reconstruction(self, dataloader, epochs=50):
        """阶段1: 重建预训练"""
        print("\n" + "="*50)
        print("Stage 1: Reconstruction Pretraining")
        print("="*50)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for noisy_signals, clean_signals in pbar:
                noisy_signals = noisy_signals.to(self.device)
                clean_signals = clean_signals.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(noisy_signals)
                
                # 如果模型返回字典，提取重建信号
                if isinstance(outputs, dict):
                    if 'signal_recon' in outputs:
                        recon_signals = outputs['signal_recon']
                    else:
                        # 使用特征进行重建
                        recon_signals = outputs.get('features', outputs['logits'])
                else:
                    recon_signals = outputs
                
                # 计算重建损失
                recon_loss = F.mse_loss(recon_signals, clean_signals)
                
                recon_loss.backward()
                self.optimizer.step()
                
                total_loss += recon_loss.item()
                pbar.set_postfix({'loss': recon_loss.item()})
            
            avg_loss = total_loss / len(dataloader)
            self.history['recon_loss'].append(avg_loss)
            self.scheduler.step()
            
            print(f'Epoch {epoch+1}/{epochs}, Reconstruction Loss: {avg_loss:.6f}')
        
        return self.history['recon_loss']
    
    def pretrain_contrastive(self, dataloader, epochs=50):
        """阶段2: 对比学习预训练"""
        print("\n" + "="*50)
        print("Stage 2: Contrastive Learning Pretraining")
        print("="*50)
        
        # 添加投影头
        projection_head = ProjectionHead(
            input_dim=self.model.classifier[0].in_features,  # 假设分类器第一层的输入维度
            hidden_dim=256,
            output_dim=128
        ).to(self.device)
        
        # 优化器包含模型和投影头
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(projection_head.parameters()),
            lr=1e-3, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        contrast_loss_fn = InfoNCELoss(temperature=0.1)
        
        self.model.train()
        projection_head.train()
        
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for aug1, aug2 in pbar:
                aug1 = aug1.to(self.device)
                aug2 = aug2.to(self.device)
                batch_size = aug1.size(0)
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs1 = self.model(aug1)
                outputs2 = self.model(aug2)
                
                # 提取特征
                if isinstance(outputs1, dict):
                    features1 = outputs1.get('features', outputs1['logits'])
                    features2 = outputs2.get('features', outputs2['logits'])
                else:
                    features1 = outputs1
                    features2 = outputs2
                
                # 全局平均池化
                if len(features1.shape) > 2:
                    features1 = F.adaptive_avg_pool1d(features1, 1).squeeze(-1)
                    features2 = F.adaptive_avg_pool1d(features2, 1).squeeze(-1)
                
                # 投影
                z1 = projection_head(features1)
                z2 = projection_head(features2)
                
                # 对比损失
                contrast_loss = contrast_loss_fn(z1, z2)
                
                contrast_loss.backward()
                optimizer.step()
                
                total_loss += contrast_loss.item()
                pbar.set_postfix({'loss': contrast_loss.item()})
            
            avg_loss = total_loss / len(dataloader)
            self.history['contrast_loss'].append(avg_loss)
            scheduler.step()
            
            print(f'Epoch {epoch+1}/{epochs}, Contrastive Loss: {avg_loss:.6f}')
        
        return self.history['contrast_loss']
    
    def pretrain_masked(self, dataloader, epochs=50, mask_ratio=0.15):
        """阶段3: 掩码预测预训练"""
        print("\n" + "="*50)
        print("Stage 3: Masked Signal Prediction Pretraining")
        print("="*50)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for masked_signals, original_signals, masks in pbar:
                masked_signals = masked_signals.to(self.device)
                original_signals = original_signals.to(self.device)
                masks = masks.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(masked_signals)
                
                # 提取重建信号
                if isinstance(outputs, dict):
                    if 'signal_recon' in outputs:
                        pred_signals = outputs['signal_recon']
                    else:
                        pred_signals = outputs.get('features', outputs['logits'])
                else:
                    pred_signals = outputs
                
                # 只计算掩码位置的损失
                mask_loss = F.mse_loss(
                    pred_signals[:, :, masks], 
                    original_signals[:, :, masks]
                )
                
                mask_loss.backward()
                self.optimizer.step()
                
                total_loss += mask_loss.item()
                pbar.set_postfix({'loss': mask_loss.item()})
            
            avg_loss = total_loss / len(dataloader)
            self.history['mask_loss'].append(avg_loss)
            self.scheduler.step()
            
            print(f'Epoch {epoch+1}/{epochs}, Masked Loss: {avg_loss:.6f}')
        
        return self.history['mask_loss']
    
    def pretrain_multi_task(self, dataloader, epochs=100, 
                           recon_weight=1.0, contrast_weight=0.5, mask_weight=0.3):
        """多任务联合预训练"""
        print("\n" + "="*50)
        print("Multi-Task Joint Pretraining")
        print("="*50)
        
        # 创建不同任务的数据加载器
        recon_dataset = PretrainDataset(dataloader.dataset.signals, 'reconstruction')
        contrast_dataset = PretrainDataset(dataloader.dataset.signals, 'contrastive')
        mask_dataset = PretrainDataset(dataloader.dataset.signals, 'masked')
        
        recon_loader = DataLoader(recon_dataset, batch_size=dataloader.batch_size, shuffle=True)
        contrast_loader = DataLoader(contrast_dataset, batch_size=dataloader.batch_size, shuffle=True)
        mask_loader = DataLoader(mask_dataset, batch_size=dataloader.batch_size, shuffle=True)
        
        # 投影头
        projection_head = ProjectionHead(
            input_dim=self.model.classifier[0].in_features,
            hidden_dim=256,
            output_dim=128
        ).to(self.device)
        
        # 优化器
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(projection_head.parameters()),
            lr=1e-3, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        contrast_loss_fn = InfoNCELoss(temperature=0.1)
        
        self.model.train()
        projection_head.train()
        
        for epoch in range(epochs):
            total_loss = 0
            recon_loss_sum = 0
            contrast_loss_sum = 0
            mask_loss_sum = 0
            
            # 创建混合迭代器
            loaders = [recon_loader, contrast_loader, mask_loader]
            min_len = min(len(loader) for loader in loaders)
            
            pbar = tqdm(range(min_len), desc=f'Epoch {epoch+1}/{epochs}')
            
            recon_iter = iter(recon_loader)
            contrast_iter = iter(contrast_loader)
            mask_iter = iter(mask_loader)
            
            for _ in pbar:
                optimizer.zero_grad()
                
                # 重建任务
                noisy_signals, clean_signals = next(recon_iter)
                noisy_signals = noisy_signals.to(self.device)
                clean_signals = clean_signals.to(self.device)
                
                outputs = self.model(noisy_signals)
                if isinstance(outputs, dict):
                    recon_signals = outputs.get('signal_recon', outputs.get('features', outputs['logits']))
                else:
                    recon_signals = outputs
                
                recon_loss = F.mse_loss(recon_signals, clean_signals)
                
                # 对比学习任务
                aug1, aug2 = next(contrast_iter)
                aug1 = aug1.to(self.device)
                aug2 = aug2.to(self.device)
                
                outputs1 = self.model(aug1)
                outputs2 = self.model(aug2)
                
                if isinstance(outputs1, dict):
                    features1 = outputs1.get('features', outputs1['logits'])
                    features2 = outputs2.get('features', outputs2['logits'])
                else:
                    features1 = outputs1
                    features2 = outputs2
                
                if len(features1.shape) > 2:
                    features1 = F.adaptive_avg_pool1d(features1, 1).squeeze(-1)
                    features2 = F.adaptive_avg_pool1d(features2, 1).squeeze(-1)
                
                z1 = projection_head(features1)
                z2 = projection_head(features2)
                contrast_loss = contrast_loss_fn(z1, z2)
                
                # 掩码预测任务
                masked_signals, original_signals, masks = next(mask_iter)
                masked_signals = masked_signals.to(self.device)
                original_signals = original_signals.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(masked_signals)
                if isinstance(outputs, dict):
                    pred_signals = outputs.get('signal_recon', outputs.get('features', outputs['logits']))
                else:
                    pred_signals = outputs
                
                mask_loss = F.mse_loss(
                    pred_signals[:, :, masks], 
                    original_signals[:, :, masks]
                )
                
                # 总损失
                total_task_loss = (recon_weight * recon_loss + 
                                 contrast_weight * contrast_loss + 
                                 mask_weight * mask_loss)
                
                total_task_loss.backward()
                optimizer.step()
                
                total_loss += total_task_loss.item()
                recon_loss_sum += recon_loss.item()
                contrast_loss_sum += contrast_loss.item()
                mask_loss_sum += mask_loss.item()
                
                pbar.set_postfix({
                    'total': total_task_loss.item(),
                    'recon': recon_loss.item(),
                    'contrast': contrast_loss.item(),
                    'mask': mask_loss.item()
                })
            
            # 记录损失
            avg_total_loss = total_loss / min_len
            avg_recon_loss = recon_loss_sum / min_len
            avg_contrast_loss = contrast_loss_sum / min_len
            avg_mask_loss = mask_loss_sum / min_len
            
            self.history['total_loss'].append(avg_total_loss)
            self.history['recon_loss'].append(avg_recon_loss)
            self.history['contrast_loss'].append(avg_contrast_loss)
            self.history['mask_loss'].append(avg_mask_loss)
            
            scheduler.step()
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Total Loss: {avg_total_loss:.6f}')
            print(f'  Recon Loss: {avg_recon_loss:.6f}')
            print(f'  Contrast Loss: {avg_contrast_loss:.6f}')
            print(f'  Mask Loss: {avg_mask_loss:.6f}')
            
            # 保存检查点
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(f'pretrain_epoch_{epoch+1}.pth')
        
        return self.history
    
    def save_checkpoint(self, path):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")


# ==================== 数据加载器创建函数 ====================

def create_pretrain_dataloader(signals, task_type='reconstruction', batch_size=128, shuffle=True):
    """创建预训练数据加载器"""
    dataset = PretrainDataset(signals, task_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ==================== 预训练流程函数 ====================

def run_pretraining(model, signals, device='cuda', save_dir='./pretrain_checkpoints'):
    """运行完整的预训练流程"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建预训练器
    trainer = PretrainTrainer(model, device)
    
    # 阶段1: 重建预训练
    recon_loader = create_pretrain_dataloader(signals, 'reconstruction')
    trainer.pretrain_reconstruction(recon_loader, epochs=30)
    trainer.save_checkpoint(os.path.join(save_dir, 'stage1_reconstruction.pth'))
    
    # 阶段2: 对比学习预训练
    contrast_loader = create_pretrain_dataloader(signals, 'contrastive')
    trainer.pretrain_contrastive(contrast_loader, epochs=30)
    trainer.save_checkpoint(os.path.join(save_dir, 'stage2_contrastive.pth'))
    
    # 阶段3: 掩码预测预训练
    mask_loader = create_pretrain_dataloader(signals, 'masked')
    trainer.pretrain_masked(mask_loader, epochs=30)
    trainer.save_checkpoint(os.path.join(save_dir, 'stage3_masked.pth'))
    
    # 最终保存
    trainer.save_checkpoint(os.path.join(save_dir, 'final_pretrained.pth'))
    
    return trainer