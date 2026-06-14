import torch
import matplotlib.pyplot as plt
import os
import sys

def visualize_prosody(file_path="prosody_embs.pt"):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run inference first.")
        sys.exit(1)

    # Load embeddings [1, 32, dim]
    try:
        embs = torch.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

    # Visualize as a heatmap [32, dim]
    plt.figure(figsize=(10, 6))
    plt.imshow(embs[0].cpu().numpy(), aspect='auto', cmap='viridis')
    plt.title(f"Prosody Token Embeddings ({file_path})")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Prosody Token Index")
    plt.colorbar()
    plt.tight_layout()
    
    output_png = "prosody_viz.png"
    plt.savefig(output_png)
    print(f"Visualization saved to {output_png}")
    plt.show()

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "prosody_embs.pt"
    visualize_prosody(file_path)
