import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

from dataset import VIOLogDataset
from set_transformer import AKITTransformer

def train_epoch(model, loader, optimizer, device, stats, epoch, writer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_q_loss = 0
    total_r_loss = 0
    
    q_mean = torch.tensor(stats['q_mean'], device=device)
    q_std = torch.tensor(stats['q_std'], device=device)
    r_mean = torch.tensor(stats['r_mean'], device=device)
    r_std = torch.tensor(stats['r_std'], device=device)
    
    for batch_idx, (ctx, mset, targets) in enumerate(loader):
        ctx, mset, targets = ctx.to(device), mset.to(device), targets.to(device)
        
        # Forward pass
        q_scales, r_scale = model(ctx, mset)
        
        # Denormalize targets
        q_target = targets[:, 0] * q_std + q_mean
        r_target = targets[:, 1] * r_std + r_mean
        
        # Convert predictions to same space as targets
        q_pred = torch.log(q_scales.mean(dim=1) + 1e-6)
        r_pred = torch.log(r_scale.squeeze() + 1e-6)
        
        # Compute losses
        loss_q = F.mse_loss(q_pred, q_target)
        loss_r = F.mse_loss(r_pred, r_target)
        
        # Adaptive weighting based on visual quality
        with torch.no_grad():
            # Use num_matches feature (index 2 in set features) as visual quality indicator
            visual_quality = mset[:, :, 2].mean(dim=1)  # average over set
            r_weight = 0.3 + 0.7 * (1 - visual_quality)  # higher weight when visual quality poor
            r_weight = r_weight.mean()  # average over batch
        
        # Combined loss
        loss = loss_q + r_weight * loss_r
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_q_loss += loss_q.item()
        total_r_loss += loss_r.item()
        
        # Log to tensorboard
        step = epoch * len(loader) + batch_idx
        writer.add_scalar('Train/Batch_Loss', loss.item(), step)
        writer.add_scalar('Train/Batch_Q_Loss', loss_q.item(), step)
        writer.add_scalar('Train/Batch_R_Loss', loss_r.item(), step)
        writer.add_scalar('Train/R_Weight', r_weight.item(), step)
        
    return (total_loss / len(loader), 
            total_q_loss / len(loader), 
            total_r_loss / len(loader))

def validate(model, loader, device, stats, epoch, writer):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_q_loss = 0
    total_r_loss = 0
    
    q_mean = torch.tensor(stats['q_mean'], device=device)
    q_std = torch.tensor(stats['q_std'], device=device)
    r_mean = torch.tensor(stats['r_mean'], device=device)
    r_std = torch.tensor(stats['r_std'], device=device)
    
    all_q_pred = []
    all_q_target = []
    all_r_pred = []
    all_r_target = []
    
    with torch.no_grad():
        for ctx, mset, targets in loader:
            ctx, mset, targets = ctx.to(device), mset.to(device), targets.to(device)
            
            q_scales, r_scale = model(ctx, mset)
            
            q_target = targets[:, 0] * q_std + q_mean
            r_target = targets[:, 1] * r_std + r_mean
            
            q_pred = torch.log(q_scales.mean(dim=1) + 1e-6)
            r_pred = torch.log(r_scale.squeeze() + 1e-6)
            
            loss_q = F.mse_loss(q_pred, q_target)
            loss_r = F.mse_loss(r_pred, r_target)
            
            # Store for correlation analysis
            all_q_pred.append(q_pred.cpu())
            all_q_target.append(q_target.cpu())
            all_r_pred.append(r_pred.cpu())
            all_r_target.append(r_target.cpu())
            
            total_loss += (loss_q + loss_r).item()
            total_q_loss += loss_q.item()
            total_r_loss += loss_r.item()
    
    # Compute correlations
    q_pred_cat = torch.cat(all_q_pred).numpy()
    q_target_cat = torch.cat(all_q_target).numpy()
    r_pred_cat = torch.cat(all_r_pred).numpy()
    r_target_cat = torch.cat(all_r_target).numpy()
    
    q_corr = np.corrcoef(q_pred_cat, q_target_cat)[0, 1]
    r_corr = np.corrcoef(r_pred_cat, r_target_cat)[0, 1]
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', total_loss / len(loader), epoch)
    writer.add_scalar('Val/Q_Loss', total_q_loss / len(loader), epoch)
    writer.add_scalar('Val/R_Loss', total_r_loss / len(loader), epoch)
    writer.add_scalar('Val/Q_Correlation', q_corr, epoch)
    writer.add_scalar('Val/R_Correlation', r_corr, epoch)
    
    return (total_loss / len(loader), 
            total_q_loss / len(loader), 
            total_r_loss / len(loader),
            q_corr, r_corr)

def plot_predictions(model, loader, device, stats, save_path):
    """Plot model predictions vs targets"""
    model.eval()
    q_preds = []
    q_targets = []
    r_preds = []
    r_targets = []
    
    q_mean = stats['q_mean']
    q_std = stats['q_std']
    r_mean = stats['r_mean']
    r_std = stats['r_std']
    
    with torch.no_grad():
        for ctx, mset, targets in loader:
            ctx, mset = ctx.to(device), mset.to(device)
            q_scales, r_scale = model(ctx, mset)
            
            q_pred = torch.log(q_scales.mean(dim=1) + 1e-6).cpu().numpy()
            r_pred = torch.log(r_scale.squeeze() + 1e-6).cpu().numpy()
            
            q_target = (targets[:, 0].numpy() * q_std + q_mean)
            r_target = (targets[:, 1].numpy() * r_std + r_mean)
            
            q_preds.extend(q_pred)
            q_targets.extend(q_target)
            r_preds.extend(r_pred)
            r_targets.extend(r_target)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Q predictions vs targets
    axes[0, 0].scatter(q_targets, q_preds, alpha=0.5)
    axes[0, 0].plot([min(q_targets), max(q_targets)], 
                    [min(q_targets), max(q_targets)], 'r--')
    axes[0, 0].set_xlabel('Target Q (log)')
    axes[0, 0].set_ylabel('Predicted Q (log)')
    axes[0, 0].set_title(f'Q Prediction (corr: {np.corrcoef(q_targets, q_preds)[0,1]:.3f})')
    axes[0, 0].grid(True)
    
    # Q over time
    axes[0, 1].plot(q_targets[:500], label='Target', alpha=0.7)
    axes[0, 1].plot(q_preds[:500], label='Predicted', alpha=0.7)
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Q (log)')
    axes[0, 1].set_title('Q Predictions Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # R predictions vs targets
    axes[1, 0].scatter(r_targets, r_preds, alpha=0.5)
    axes[1, 0].plot([min(r_targets), max(r_targets)], 
                    [min(r_targets), max(r_targets)], 'r--')
    axes[1, 0].set_xlabel('Target R (log)')
    axes[1, 0].set_ylabel('Predicted R (log)')
    axes[1, 0].set_title(f'R Prediction (corr: {np.corrcoef(r_targets, r_preds)[0,1]:.3f})')
    axes[1, 0].grid(True)
    
    # R over time
    axes[1, 1].plot(r_targets[:500], label='Target', alpha=0.7)
    axes[1, 1].plot(r_preds[:500], label='Predicted', alpha=0.7)
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel('R (log)')
    axes[1, 1].set_title('R Predictions Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Configuration
    config = {
        'data_path': 'logs/vio_full.log',
        'window_size': 10,
        'batch_size': 32,
        'hidden_dim': 128,
        'num_inducing': 32,
        'num_heads': 4,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42
    }
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f'runs/akit_experiment_{timestamp}'
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    import json
    with open(f'{exp_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Initialize tensorboard
    writer = SummaryWriter(exp_dir)
    
    print(f"Training on device: {config['device']}")
    print(f"Experiment directory: {exp_dir}")
    
    # Load and split dataset
    print("Loading dataset...")
    full_dataset = VIOLogDataset(config['data_path'], 
                                  window=config['window_size'],
                                  mode='train')
    
    # Split into train/val/test
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Get normalization stats
    stats = full_dataset.get_normalization_stats()
    
    # Initialize model
    print("Initializing model...")
    model = AKITTransformer(
        context_dim=7,  # from context_cols
        set_dim=11,     # from set_cols
        hidden_dim=config['hidden_dim'],
        num_inducing=config['num_inducing'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(config['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5,
        verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15
    
    for epoch in range(config['num_epochs']):
        # Train
        train_loss, train_q_loss, train_r_loss = train_epoch(
            model, train_loader, optimizer, config['device'], stats, epoch, writer
        )
        
        # Validate
        val_loss, val_q_loss, val_r_loss, q_corr, r_corr = validate(
            model, val_loader, config['device'], stats, epoch, writer
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"  Train - Loss: {train_loss:.4f} (Q: {train_q_loss:.4f}, R: {train_r_loss:.4f})")
        print(f"  Val   - Loss: {val_loss:.4f} (Q: {val_q_loss:.4f}, R: {val_r_loss:.4f})")
        print(f"  Corr  - Q: {q_corr:.3f}, R: {r_corr:.3f}")
        print(f"  LR    - {current_lr:.2e}")
        
        # Log to tensorboard
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Q_Loss', train_q_loss, epoch)
        writer.add_scalar('Train/R_Loss', train_r_loss, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Q_Loss', val_q_loss, epoch)
        writer.add_scalar('Val/R_Loss', val_r_loss, epoch)
        writer.add_scalar('Val/Q_Correlation', q_corr, epoch)
        writer.add_scalar('Val/R_Correlation', r_corr, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'stats': stats,
                'config': config
            }, f'{exp_dir}/best_model.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\nTraining complete!")
    
    # Load best model for testing
    print("\nEvaluating on test set...")
    checkpoint = torch.load(f'{exp_dir}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_q_loss, test_r_loss, q_corr, r_corr = validate(
        model, test_loader, config['device'], stats, epoch, writer
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f} (Q: {test_q_loss:.4f}, R: {test_r_loss:.4f})")
    print(f"  Correlation - Q: {q_corr:.3f}, R: {r_corr:.3f}")
    
    # Plot predictions
    plot_predictions(model, test_loader, config['device'], stats, 
                     f'{exp_dir}/predictions.png')
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'stats': stats,
        'config': config
    }, f'{exp_dir}/final_model.pth')
    
    print(f"\nModels saved to {exp_dir}")
    print(f"Run 'tensorboard --logdir {exp_dir}' to view training curves")
    
    writer.close()

if __name__ == "__main__":
    main()
