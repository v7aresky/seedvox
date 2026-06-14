import os
import json
import argparse
import torch
from explicit_pros_phon_planner.trainer import ExplicitTrainer

def main():
    parser = argparse.ArgumentParser(description="Train the SeedVox Explicit Phoneme & Prosody Planner")
    parser.add_argument("--config", default="configs/default.json", help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--g2p", default="g2p_en", help="G2P backend for target generation")
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        return

    with open(args.config, "r") as f:
        cfg = json.load(f)
        
    # Ensure necessary config fields for the new planner are present
    # These can be overridden or added here if not in the JSON
    if 'model' not in cfg: cfg['model'] = {}
    cfg['model']['use_phonetic'] = True
    
    # Initialize and run the Explicit Trainer
    # This trainer uses the ExplicitPlannerModel and handles GT phoneme generation
    trainer = ExplicitTrainer(
        config=cfg, 
        device=torch.device(args.device), 
        resume_path=args.resume, 
        g2p_backend=args.g2p
    )
    
    print("Starting training with Explicit Phonetic Planner...")
    trainer.train()

if __name__ == "__main__":
    main()
