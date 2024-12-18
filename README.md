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

### Masks Visualization

Different attention patterns explored in our experiments:

![Attention Masks](masks/attention_masks.png)

Visualization of attention masks with different neighborhood patterns and expansion factors. From left to right: expansion steps k=1, k=2, and k=3. From top to bottom: 1D all neighbors, 1D cardinal neighbors, 2D all neighbors, and 2D cardinal neighbors. White pixels indicate allowed attention connections between patches.

## Experiment Details

### Model Details

We use a standard ViT architecture with default parameters:
- Input image size: 224x224
- Patch size: 16x16
- Hidden dimension: 768
- Number of heads: 12 
- Number of layers: 12
- MLP ratio: 4
- Dropout: 0.1

The only modification is in the attention mechanism, where we apply different local attention masks while keeping the core architecture unchanged.

### Training Details

We used cosine annealing learning rate scheduling with a peak learning rate of 0.0003 and 3% warmup steps. We trained
all models for 50 epochs with batch size 64.


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

## Results

We benchmarked the performance of different attention patterns on CIFAR-10. We obtained all results with the training
configuration mentioned in the [Experiment Details](#experiment-details) section.

| Model Configuration | Test Accuracy |
|-------------------|---------------|
| 2D Cardinal (k=1) | 82.02% |
| 2D Cardinal (k=2) | 81.78% |
| 1D Cardinal (k=1) | 81.44% |
| 2D Cardinal (k=3) | 81.26% |
| 2D All Neighbors (k=1) | 81.21% |
| 1D All Neighbors (k=1) | 81.15% |
| 1D Cardinal (k=2) | 80.99% |
| 1D Cardinal (k=3) | 80.86% |
| 1D All Neighbors (k=2) | 80.32% |
| 2D All Neighbors (k=2) | 80.25% |
| 2D All Neighbors (k=3) | 79.08% |
| 1D All Neighbors (k=3) | 78.53% |
| Full Attention | 77.43% |

The evaluation accuracy plot:
![Evaluation Accuracy](plots/evaluation_accuracy.png)

