# utils.py

import os
import matplotlib.pyplot as plt
import torch

def visualize_and_save_attention_masks(attention_masks, titles, output_dir, expansion_step):
    os.makedirs(output_dir, exist_ok=True)

    for idx, (mask, title) in enumerate(zip(attention_masks, titles)):
        # Create individual plot for each mask
        plt.figure(figsize=(8, 6))
        im = plt.imshow(mask.cpu().numpy(), cmap='viridis')
        plt.title(f'{title} (Expansion Step: {expansion_step})')
        plt.xlabel('Patch Index')
        plt.ylabel('Patch Index')
        plt.colorbar(im)
        
        # Save individual mask plot
        plt.savefig(os.path.join(output_dir, f'{title.replace(" ", "_").lower()}.png'))
        plt.close()

def save_accuracies(train_accuracies, eval_accuracies, train_losses, eval_losses, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_accuracies, os.path.join(output_dir, 'train_accuracies.pt'))
    torch.save(eval_accuracies, os.path.join(output_dir, 'eval_accuracies.pt'))
    torch.save(train_losses, os.path.join(output_dir, 'train_losses.pt'))
    torch.save(eval_losses, os.path.join(output_dir, 'eval_losses.pt'))
