# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
from tqdm import tqdm
from data import get_data_loaders
from models import VisionTransformer
from eval import evaluate
from masks import *
from utils import visualize_and_save_attention_masks, save_accuracies
import wandb

def main():
    # Command-line arguments
    parser = argparse.ArgumentParser(description='Vision Transformer Training with Local Attention Masks')
    parser.add_argument('--mask_type', type=str, default='full', choices=['full', '1d_cardinal', '1d_all_neighbors', '2d_cardinal', '2d_all_neighbors'], help='Type of attention mask to use')
    parser.add_argument('--expansion_step', type=int, default=1, help='Expansion step for local attention masks')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'fashion_mnist'], help='Dataset to use')
    parser.add_argument('--wandb_project', type=str, default='ViT-Attention-Masks', 
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, 
                       help='Weights & Biases entity name')
    args = parser.parse_args()

    # Initialize wandb
    run_name = f"{args.dataset}_{args.mask_type}"
    if args.mask_type != 'full':
        run_name += f"_expansion_{args.expansion_step}"
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "dataset": args.dataset,
            "mask_type": args.mask_type,
            "expansion_step": args.expansion_step,
            "epochs": args.epochs,
        }
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Hyperparameters
    img_size = 224
    patch_size = 16
    batch_size = 64
    embed_dim = 768
    num_heads = 12
    depth = 12
    mlp_ratio = 4.0
    dropout = 0.1
    num_epochs = args.epochs
    expansion_step = args.expansion_step

    # Prepare data
    train_loader, test_loader = get_data_loaders(
        batch_size=batch_size, img_size=img_size, dataset=args.dataset)

    # Compute number of patches
    num_patches_height = img_size // patch_size
    num_patches_width = img_size // patch_size
    N = num_patches_height * num_patches_width

    # Generate attention masks based on mask_type
    attention_mask = None
    mask_title = 'Full Attention'
    if args.mask_type != 'full':
        if args.mask_type == '1d_cardinal':
            num_neighbors = compute_num_neighbors_cardinal(expansion_step)
            attention_mask = torch.tensor(
                generate_1d_attention_mask(num_neighbors, N))
            mask_title = '1D Cardinal Mask'
        elif args.mask_type == '1d_all_neighbors':
            num_neighbors = compute_num_neighbors_all_neighbors(expansion_step)
            attention_mask = torch.tensor(
                generate_1d_attention_mask(num_neighbors, N))
            mask_title = '1D All Neighbors Mask'
        elif args.mask_type == '2d_cardinal':
            attention_mask = torch.tensor(
                generate_2d_attention_mask_cardinal(
                    expansion_step, num_patches_height, num_patches_width))
            mask_title = '2D Cardinal Mask'
        elif args.mask_type == '2d_all_neighbors':
            attention_mask = torch.tensor(
                generate_2d_attention_mask_all_neighbors(
                    expansion_step, num_patches_height, num_patches_width))
            mask_title = '2D All Neighbors Mask'

        # Save the attention mask plot
        visualize_and_save_attention_masks(
            [attention_mask], [mask_title], args.output_dir, expansion_step)
    else:
        mask_title = 'Full Attention'

    # Move attention mask to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Create model with correct number of input channels
    in_channels = 1 if args.dataset == 'fashion_mnist' else 3
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
        depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
        dropout=dropout).to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.03 * total_steps)  # 3% of total steps
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps)

    # Learning rate warmup
    def adjust_learning_rate(optimizer, step, warmup_steps):
        if step < warmup_steps:
            lr_scale = float(step) / float(max(1, warmup_steps))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * 3e-4
        else:
            scheduler.step()

    criterion = nn.CrossEntropyLoss()

    # Training and evaluation
    train_accuracies = []
    eval_accuracies = []
    eval_losses = []
    train_losses = []

    print(f"Training ViT with {mask_title}")
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        start_time = time.time()

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch} [Train]') as pbar:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                adjust_learning_rate(optimizer, global_step, warmup_steps)
                global_step += 1

                optimizer.zero_grad()
                outputs = model(inputs, attn_mask=attention_mask)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                total += targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

                pbar.update(1)

        epoch_time = time.time() - start_time
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        speed = total / epoch_time

        print(f'Epoch {epoch} Training: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Speed: {speed:.2f} samples/sec')
        train_accuracies.append(epoch_acc)
        train_losses.append(epoch_loss)

        # Log training metrics
        wandb.log({
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
        }, step=epoch)

        # Evaluation
        eval_acc, eval_loss, eval_speed = evaluate(model, device, test_loader, criterion, epoch, attn_mask=attention_mask)
        eval_accuracies.append(eval_acc)
        eval_losses.append(eval_loss)
        
        # Log evaluation metrics
        wandb.log({
            "eval_accuracy": eval_acc,
            "eval_loss": eval_loss,
        }, step=epoch)

    # Save accuracies
    save_accuracies(train_accuracies, eval_accuracies, train_losses, eval_losses, args.output_dir)
    print(f'Accuracies saved to {args.output_dir}')

    # Close wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
