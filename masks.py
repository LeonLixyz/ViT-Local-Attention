# masks.py

import numpy as np
import torch

def get_patch_positions(num_patches_height, num_patches_width):
    N = num_patches_height * num_patches_width
    positions = [(i // num_patches_width, i % num_patches_width) for i in range(N)]
    return positions

def generate_2d_attention_mask_cardinal(expansion_step, num_patches_height, num_patches_width):
    N = num_patches_height * num_patches_width
    attention_mask = np.zeros((N, N), dtype=np.float32)
    positions = get_patch_positions(num_patches_height, num_patches_width)

    for idx, (i_row, i_col) in enumerate(positions):
        local_indices = [idx]  # Include the current patch

        for delta in range(1, expansion_step + 1):
            # Up
            if i_row - delta >= 0:
                neighbor_idx = (i_row - delta) * num_patches_width + i_col
                local_indices.append(neighbor_idx)
            # Down
            if i_row + delta < num_patches_height:
                neighbor_idx = (i_row + delta) * num_patches_width + i_col
                local_indices.append(neighbor_idx)
            # Left
            if i_col - delta >= 0:
                neighbor_idx = i_row * num_patches_width + (i_col - delta)
                local_indices.append(neighbor_idx)
            # Right
            if i_col + delta < num_patches_width:
                neighbor_idx = i_row * num_patches_width + (i_col + delta)
                local_indices.append(neighbor_idx)

        attention_mask[idx, local_indices] = 1.0

    return attention_mask

def generate_2d_attention_mask_all_neighbors(expansion_step, num_patches_height, num_patches_width):
    N = num_patches_height * num_patches_width
    attention_mask = np.zeros((N, N), dtype=np.float32)
    positions = get_patch_positions(num_patches_height, num_patches_width)

    for idx, (i_row, i_col) in enumerate(positions):
        local_indices = []
        for r in range(i_row - expansion_step, i_row + expansion_step + 1):
            for c in range(i_col - expansion_step, i_col + expansion_step + 1):
                if 0 <= r < num_patches_height and 0 <= c < num_patches_width:
                    neighbor_idx = r * num_patches_width + c
                    local_indices.append(neighbor_idx)
        attention_mask[idx, local_indices] = 1.0

    return attention_mask

def generate_1d_attention_mask(num_neighbors, N):
    attention_mask = np.zeros((N, N), dtype=np.float32)
    half_window = (num_neighbors - 1) // 2

    for i in range(N):
        start = max(i - half_window, 0)
        end = min(i + half_window + 1, N)
        attention_mask[i, start:end] = 1.0

    return attention_mask

def compute_num_neighbors_cardinal(expansion_step):
    num_neighbors = 1 + 4 * expansion_step
    return num_neighbors

def compute_num_neighbors_all_neighbors(expansion_step):
    window_size = 2 * expansion_step + 1
    num_neighbors = window_size * window_size
    return num_neighbors
