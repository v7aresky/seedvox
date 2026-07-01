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

## Architecture

The model follows a **fusion-based** architecture: text and phoneme features are fused via cross-attention before being passed to the acoustic decoder, avoiding the tri-alignment problem of separate modality blocks.

```mermaid
graph TD
    %% Inputs
    Text["Input Text"]
    Audio["Reference Audio"]

    %% Text Encoding
    Text -->|Char Tokenizer| CT["Char IDs"]
    Text -->|BPE Tokenizer| BT["BPE IDs"]
    CT -->|"Text Encoder"| TE["Text Features<br/>(B, T_text, dim)"]
    BT -->|"BPE Encoder + Expand"| BE["BPE Features<br/>(B, T_text, dim)"]
    BE -->|"Gated Add"| TE

    %% Audio Encoding
    Audio -->|"Mimi Encoder"| AT["Audio Tokens<br/>(B, 16, T_audio)"]
    AT -->|"Speaker Encoder"| SPK["Speaker Latents<br/>(B, 16, dim)"]
    AT -->|"Prosody Encoder"| PRS["Prosody Embeddings<br/>(B, 32, dim)"]

    %% JEPA Prosody Planner
    TE -->|"JEPA Prosody Planner"| JPRS["Predicted Prosody<br/>(B, 32, dim)"]

    %% Phonetic Planner (AR)
    TE -->|"Phonetic Planner"| PH["Phoneme IDs<br/>(B, T_ph)"]
    PH -->|"Embed + Project"| PHE["Phoneme Features<br/>(B, T_ph, dim)"]

    %% FiLM (Speaker Conditioning)
    SPK -->|"Mean"| SL["Speaker Vector<br/>(B, dim)"]
    SL -->|"FiLM Phn"| FPH["Modulated Phoneme<br/>(B, T_ph, dim)"]
    SL -->|"FiLM Prs"| FPRS["Modulated Prosody<br/>(B, 32, dim)"]
    PHE --> FPH
    JPRS --> FPRS

    %% Linguistic Fusion (Key Innovation)
    TE --> LF["LinguisticFusion<br/>(Cross-Attention)"]
    FPH --> LF
    LF -->|"Gated Residual"| UT["Unified Text<br/>(B, T_text, dim)"]

    %% Context Assembly
    SPK --> CTX["Augmented Context"]
    FPRS --> CTX
    UT --> CTX

    CTX -->|"Acoustic Decoder"| TOK["Acoustic Tokens<br/>(B, 16, T_audio)"]
    TOK -->|"Depformer"| RVQ["RVQ Tokens<br/>(B, 16, T_audio)"]
    RVQ -->|"Mimi Decoder"| WAV["Audio Waveform"]
```

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
