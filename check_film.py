import torch
import json
from explicit_pros_phon_planner.model import ExplicitPlannerModel
from seedvox.utils.tokenizer import CharTokenizer

# Load config
with open("configs/default.json", "r") as f:
    cfg = json.load(f)

# Instantiate tokenizer and model
tokenizer = CharTokenizer()
model = ExplicitPlannerModel(cfg, tokenizer.vocab_size, phoneme_vocab_size=128)

# Print all modules to verify FiLM
print("Checking for FiLM layers in model:")
found = False
for name, module in model.named_modules():
    if "film" in name:
        print(f"Found: {name}")
        found = True

if not found:
    print("No FiLM layers found!")
