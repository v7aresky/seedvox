# SeedVox

SeedVox is a hybrid speech synthesis model leveraging JEPA-based prosody planning and an AR token generator.

## Quick Start (Inference)

Run inference using the latest optimized checkpoint:

```bash
python -m explicit_pros_phon_planner.infer \
    --text "Your text here" \
    --checkpoint ./checkpoints/seedvox_latest_slim_bf16.pt \
    --dtype bf16 \
    --play \
    --log_metrics \
    --device cuda
```

## Features
- **Deterministic Synthesis**: Use `--seed <INT>` for reproducible results.
- **Optimized Inference**: Supports `torch.compile` with persistent caching (`--compile`).
- **High-Resolution Visualization**: Built-in terminal-based waveform renderer.

## Conditioning Flow

The model uses a multi-modal conditioning strategy for acoustic token generation, combining semantic text features with explicit phonetic planning and speaker/prosody latents, followed by audio synthesis.

```mermaid
graph TD

    Text[Input Text] -->|BPE/Char Encoder| TE[Enriched Text Features]
    TE -->|Transformer Phonetic Planner| PH[Phoneme IDs]
    TE -->|JEPA Prosody Planner| PRS[Prosody Embeddings]

    %% Phoneme Flow
    PH -->|Embedding| PE[Phoneme Embeddings]
    PE & Speaker -->|FiLM Adapter| FPH[Modulated Phoneme Embeddings]

    


    %% Prosody Flow
    PRS & Speaker -->|FiLM Adapter| FPRS[Modulated Prosody Embeddings]

    %% Context Construction
    TE --> C([Augmented Context])
    FPH --> C
    FPRS --> C
    Speaker --> C

    C -->|Transformer Acoustic Decoder| Acoustic[Acoustic Tokens]
    Acoustic -->|Mimi Decoder| Audio_Out[Audio Waveform]
```

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
