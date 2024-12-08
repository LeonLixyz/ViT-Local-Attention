# eval.py

import torch
import time
from tqdm import tqdm
import os

def evaluate(model, device, test_loader, criterion, epoch, attn_mask=None):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    start_time = time.time()

    with tqdm(total=len(test_loader), desc=f'Epoch {epoch} [Eval]') as pbar:
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, attn_mask=attn_mask)
                loss = criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                total += targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

                pbar.update(1)

    eval_time = time.time() - start_time
    eval_loss = running_loss / total
    eval_acc = 100. * correct / total
    speed = total / eval_time

    print(f'Epoch {epoch} Evaluation: Loss: {eval_loss:.4f}, Acc: {eval_acc:.2f}%, Speed: {speed:.2f} samples/sec')
    return eval_acc, eval_loss, speed
