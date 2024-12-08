# Local Attention Vision Transformers (ViTs)

This repository explores and benchmarks two local attention variants in Vision Transformers (ViTs). The goal is to implement and compare these local attention mechanisms in terms of speed and accuracy while evaluating their performance against the standard ViT.

## Main Experiments

- Implement and compare two local-attention mechanisms:
    - **1D Sliding Window Attention**: Operates on a flattened sequence of patches using a fixed-size sliding window.
    - **2D Sliding Window Attention**: Expands attention locally in a 2D patch grid.
- Use a standard ViT as a baseline for performance evaluation.

## Attention Mechanisms

### 1D Sliding Window Attention
Uses a fixed-size sliding window to attend to a sequence of patches. The window size determines the number of neighbors a patch attends to in the flattened 1D sequence.

### 2D Sliding Window Attention
Leverages spatial relationships in the patch grid using two types of expansions:
1. **Cardinal Expansion**:
   - Attends to patches directly above, below, left, and right of the current patch within a specified expansion step.
   - Number of neighbors grows linearly with the expansion step.
2. **All Neighbors Expansion**:
   - Attends to all patches within a square region centered on the current patch.
   - Number of neighbors grows quadratically with the expansion step.

### Masks Implementation
The attention masks for each variant are implemented in `masks.py`:
- **Cardinal Expansion**: `generate_2d_attention_mask_cardinal`
- **All Neighbors Expansion**: `generate_2d_attention_mask_all_neighbors`
- **1D Attention**: `generate_1d_attention_mask`

## Model Details

We use a standard ViT architecture with default parameters:
- Input image size: 224x224
- Patch size: 16x16
- Hidden dimension: 768
- Number of heads: 12 
- Number of layers: 12
- MLP ratio: 4
- Dropout: 0.1

The only modification is in the attention mechanism, where we apply different local attention masks while keeping the core architecture unchanged.

## Directory Structure

- **experiments/**
  - **cifar10/** - CIFAR-10 experiment results, including masks and accuracy/loss 
    - **1d_all_neighbors_expansion_*** - 1D local attention results
    - **2d_all_neighbors_expansion_*** - 2D all neighbors attention results 
    - **2d_cardinal_expansion_*** - 2D cardinal attention results
    - **full/** - Standard ViT results
- **main.py** - Entry point for running experiments
- **data.py** - Data preprocessing utilities
- **eval.py** - Evaluation scripts
- **run_experiments.py** - Experiment orchestration
- **plot_result.py** - Plotting and visualization of results
- **masks.py** - Local attention mask generation
- **models.py** - Vision Transformer model definitions
- **utils.py** - Utility functions

## Usage

### Run experiments
You can launch experiments using the `main.py` or `run_experiments.py` script. 

For launching a single experiment, use `main.py`:
```bash
python main.py --mask_type 2d_cardinal --expansion_step 2 --dataset cifar10 --epochs 50
```

If launching all experiments, use `run_experiments.py`:
```bash
python run_experiments.py --gpus 8 --expansion_steps '[1, 2, 3]'
```

### Visualize results
Use `plot_result.py` to generate plots for accuracy and loss:
```bash
python plot_result.py --results_path experiments/cifar10
```
