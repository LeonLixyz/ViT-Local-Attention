import os
import torch
import matplotlib.pyplot as plt
import argparse

def load_accuracies(experiments_dir):
    experiments = []
    train_accuracies = {}
    eval_accuracies = {}
    train_losses = {}
    eval_losses = {}

    for exp_name in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_name)
        if os.path.isdir(exp_path):
            train_acc_path = os.path.join(exp_path, 'train_accuracies.pt')
            eval_acc_path = os.path.join(exp_path, 'eval_accuracies.pt')
            train_loss_path = os.path.join(exp_path, 'train_losses.pt')
            eval_loss_path = os.path.join(exp_path, 'eval_losses.pt')

            if all(os.path.isfile(p) for p in [train_acc_path, eval_acc_path, train_loss_path, eval_loss_path]):
                # Load accuracies and losses
                train_accuracies[exp_name] = torch.load(train_acc_path)
                eval_accuracies[exp_name] = torch.load(eval_acc_path)
                train_losses[exp_name] = torch.load(train_loss_path)
                eval_losses[exp_name] = torch.load(eval_loss_path)
                experiments.append(exp_name)
            else:
                print(f"Skipping {exp_name}: Some metric files not found.")
    return experiments, train_accuracies, eval_accuracies, train_losses, eval_losses

def plot_accuracies(experiments, train_accuracies, eval_accuracies, train_losses, eval_losses, output_dir):
    plt.rcParams['figure.dpi'] = 300

    # Sort experiments by final eval accuracy
    sorted_experiments = sorted(experiments, 
                              key=lambda x: eval_accuracies[x][-1], 
                              reverse=True)
    
    # Create legend labels with final accuracies for eval plots only
    eval_legend_labels = {exp: f"{exp} (final acc: {eval_accuracies[exp][-1]:.2f}%)"
                         for exp in experiments}

    # Plot training accuracy
    plt.figure(figsize=(12, 6))
    for exp_name in sorted_experiments:
        epochs = range(1, len(train_accuracies[exp_name]) + 1)
        plt.plot(epochs, train_accuracies[exp_name], label=exp_name)  # Original label
    
    plt.title('Training Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    train_acc_path = os.path.join(output_dir, 'training_accuracy.png')
    plt.savefig(train_acc_path, bbox_inches='tight')
    plt.close()
    print(f'Training accuracy plot saved to {train_acc_path}')

    # Plot training loss
    plt.figure(figsize=(12, 6))
    for exp_name in sorted_experiments:
        epochs = range(1, len(train_losses[exp_name]) + 1)
        plt.plot(epochs, train_losses[exp_name], label=exp_name)  # Original label
    
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    train_loss_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(train_loss_path, bbox_inches='tight')
    plt.close()
    print(f'Training loss plot saved to {train_loss_path}')

    # Plot evaluation accuracy
    plt.figure(figsize=(12, 6))
    for exp_name in sorted_experiments:
        epochs = range(1, len(eval_accuracies[exp_name]) + 1)
        plt.plot(epochs, eval_accuracies[exp_name], label=eval_legend_labels[exp_name])  # Label with accuracy
    
    plt.title('Evaluation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    eval_acc_path = os.path.join(output_dir, 'evaluation_accuracy.png')
    plt.savefig(eval_acc_path, bbox_inches='tight')
    plt.close()
    print(f'Evaluation accuracy plot saved to {eval_acc_path}')

    # Plot evaluation loss
    plt.figure(figsize=(12, 6))
    for exp_name in sorted_experiments:
        epochs = range(1, len(eval_losses[exp_name]) + 1)
        plt.plot(epochs, eval_losses[exp_name], label=eval_legend_labels[exp_name])  # Label with accuracy
    
    plt.title('Evaluation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    eval_loss_path = os.path.join(output_dir, 'evaluation_loss.png')
    plt.savefig(eval_loss_path, bbox_inches='tight')
    plt.close()
    print(f'Evaluation loss plot saved to {eval_loss_path}')

def main():
    parser = argparse.ArgumentParser(description='Plot training and evaluation accuracies from experiments.')
    parser.add_argument('--experiments_dir', type=str, default='experiments', help='Directory containing experiment subdirectories')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save the plots')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load accuracies from all experiments
    experiments, train_accuracies, eval_accuracies, train_losses, eval_losses = load_accuracies(args.experiments_dir)

    if not experiments:
        print("No experiments found with accuracy and loss files.")
        return

    # Plot and save accuracies
    plot_accuracies(experiments, train_accuracies, eval_accuracies, train_losses, eval_losses, args.output_dir)

if __name__ == '__main__':
    main()
