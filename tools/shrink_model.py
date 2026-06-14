import torch
import os
import argparse

def shrink_checkpoint(input_path, output_path, dtype='fp16'):
    print(f"Loading checkpoint from {input_path}...")
    ckpt = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # If it's a full checkpoint (dict with 'model' key), extract the state dict
    if isinstance(ckpt, dict) and 'model' in ckpt:
        print("Extracting 'model' from full checkpoint...")
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    initial_size = len(state_dict)
    
    # 1. Elements to remove for inference only
    redundant_prefixes = [
        'phoneme_head.',      # Redundant in ExplicitPlannerModel
        'prosody_bottleneck.' # Training only (Mimi latent extraction)
    ]
    
    keys_to_remove = []
    for key in state_dict.keys():
        for prefix in redundant_prefixes:
            if key.startswith(prefix):
                keys_to_remove.append(key)
                break
                
    for key in keys_to_remove:
        del state_dict[key]
        
    print(f"Removed {len(keys_to_remove)} redundant parameters.")
    print(f"Remaining parameters: {len(state_dict)} (from {initial_size})")

    # 2. Convert to target dtype for massive storage savings
    if dtype != 'fp32':
        print(f"Converting weights to {dtype}...")
        for key in state_dict.keys():
            if state_dict[key].is_floating_point():
                if dtype == 'fp16':
                    state_dict[key] = state_dict[key].half()
                elif dtype == 'bf16':
                    state_dict[key] = state_dict[key].to(torch.bfloat16)

    # 3. Save slim version
    print(f"Saving slim checkpoint to {output_path}...")
    torch.save(state_dict, output_path)
    
    old_size = os.path.getsize(input_path) / (1024*1024)
    new_size = os.path.getsize(output_path) / (1024*1024)
    print(f"\nDone!")
    print(f"Original size: {old_size:.2f} MB")
    print(f"Slim size:     {new_size:.2f} MB")
    print(f"Reduction:     {100 * (1 - new_size/old_size):.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="checkpoints/seedvox__latest.pt")
    parser.add_argument("--output", default="checkpoints/seedvox_latest_slim.pt")
    parser.add_argument("--dtype", choices=['fp32', 'fp16', 'bf16'], default='fp16', help="Target data type")
    args = parser.parse_args()
    
    shrink_checkpoint(args.input, args.output, args.dtype)
