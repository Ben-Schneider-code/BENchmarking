import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_heatmap(attn: torch.Tensor, save_path="./fig.png"):
    # Convert mean attention tensor to numpy array
    attention_data = attn.numpy()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap with log scaling
    heatmap = ax.imshow(attention_data, cmap='viridis', aspect='auto', norm=LogNorm())

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Attention Weight (Log Scale)')

    # Set labels and title
    ax.set_xlabel('Source Tokens')
    ax.set_ylabel('Target Tokens')
    ax.set_title('First Layer Mean Attention Heatmap (Log Scaled)')

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()