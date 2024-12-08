import subprocess
import os
import argparse
import threading

def main():
    parser = argparse.ArgumentParser(description='Run experiments with different mask types and expansion steps.')
    parser.add_argument('--gpus', type=int, default=8, help='Total number of GPUs available')
    parser.add_argument('--expansion_steps', nargs='+', type=int, default=[1, 2, 3], help='List of expansion steps to use')
    parser.add_argument('--output_base_dir', type=str, default='experiments-original', help='Base directory to save outputs')
    parser.add_argument('--wandb_project', type=str, default='ViT-Attention-Masks_cifar10-original', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity name')
    args = parser.parse_args()

    # Configuration for different datasets
    dataset_configs = {
        'cifar10': {
            'epochs': 50,
            'output_dir': os.path.join(args.output_base_dir, 'cifar10')
        }
    }

    mask_types = ['full', '1d_cardinal', '1d_all_neighbors', '2d_cardinal', '2d_all_neighbors']
    expansion_steps = args.expansion_steps
    total_gpus = args.gpus

    # Semaphore to limit the number of concurrent processes
    semaphore = threading.Semaphore(total_gpus)

    # Initialize GPU ID
    gpu_id = 0

    # Function to run a single experiment
    def run_experiment(cmd, env):
        with semaphore:
            process = subprocess.Popen(cmd, env=env)
            process.wait()

    # Run experiments for each dataset
    for dataset, config in dataset_configs.items():
        epochs = config['epochs']
        output_base_dir = config['output_dir']
        
        # Create the base output directory if it doesn't exist
        os.makedirs(output_base_dir, exist_ok=True)
        
        print(f"\nStarting experiments for {dataset} with {epochs} epochs")

        for mask_type in mask_types:
            if mask_type == 'full':
                expansion_step = 0
                output_dir = os.path.join(output_base_dir, mask_type)
                os.makedirs(output_dir, exist_ok=True)
                print(f"Running experiment with dataset={dataset}, mask_type={mask_type} on GPU {gpu_id}")

                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

                cmd = [
                    'python', 'main.py',
                    '--mask_type', mask_type,
                    '--expansion_step', str(expansion_step),
                    '--output_dir', output_dir,
                    '--epochs', str(epochs),
                    '--dataset', dataset,
                    '--wandb_project', args.wandb_project,
                ]
                if args.wandb_entity:
                    cmd.extend(['--wandb_entity', args.wandb_entity])

                threading.Thread(target=run_experiment, args=(cmd, env)).start()

                # Update GPU ID
                gpu_id = (gpu_id + 1) % total_gpus

            else:
                for expansion_step in expansion_steps:
                    output_dir = os.path.join(output_base_dir, f'{mask_type}_expansion_{expansion_step}')
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"Running experiment with dataset={dataset}, mask_type={mask_type}, expansion_step={expansion_step} on GPU {gpu_id}")

                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

                    cmd = [
                        'python', 'main.py',
                        '--mask_type', mask_type,
                        '--expansion_step', str(expansion_step),
                        '--output_dir', output_dir,
                        '--epochs', str(epochs),
                        '--dataset', dataset,
                        '--wandb_project', args.wandb_project,
                    ]
                    if args.wandb_entity:
                        cmd.extend(['--wandb_entity', args.wandb_entity])

                    threading.Thread(target=run_experiment, args=(cmd, env)).start()

                    # Update GPU ID
                    gpu_id = (gpu_id + 1) % total_gpus

    print("All experiments have been launched.")

if __name__ == '__main__':
    main()
