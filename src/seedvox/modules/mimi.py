import os
import torch
import torch.nn as nn
from seedvox.modules.moshi.models.compression import MimiModel
from seedvox.modules.moshi.modules import SEANetEncoder, SEANetDecoder, transformer
from seedvox.modules.moshi.quantization import SplitResidualVectorQuantizer
from safetensors.torch import load_file as load_safetensors

def get_mimi_model(device='cpu', checkpoint_path=None, num_codebooks=16):
    seanet_kwargs = {
        "channels": 1,
        "dimension": 512,
        "causal": True,
        "n_filters": 64,
        "n_residual_layers": 1,
        "activation": "ELU",
        "compress": 2,
        "dilation_base": 2,
        "disable_norm_outer_blocks": 0,
        "kernel_size": 7,
        "residual_kernel_size": 3,
        "last_kernel_size": 3,
        "norm": "none",
        "pad_mode": "constant",
        "ratios": [8, 6, 5, 4],
        "true_skip": True,
    }

    quantizer_kwargs = {
        "dimension": 256,
        "n_q": 32,
        "bins": 2048,
        "input_dimension": seanet_kwargs["dimension"],
        "output_dimension": seanet_kwargs["dimension"],
    }
    transformer_kwargs = {
        "d_model": seanet_kwargs["dimension"],
        "num_heads": 8,
        "num_layers": 8,
        "causal": True,
        "layer_scale": 0.01,
        "context": 250,
        "conv_layout": True,
        "max_period": 10000,
        "gating": "none",
        "norm": "layer_norm",
        "positional_embedding": "rope",
        "dim_feedforward": 2048,
        "input_dimension": seanet_kwargs["dimension"],
        "output_dimensions": [seanet_kwargs["dimension"]],
    }

    encoder = SEANetEncoder(**seanet_kwargs)
    decoder = SEANetDecoder(**seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **quantizer_kwargs,
    )
    
    sample_rate = 24000
    frame_rate = 12.5
    
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        encoder_frame_rate=sample_rate / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
        freeze_quantizer=True,
        freeze_encoder=True
    ).to(device=device)
    
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            print(f"Warning: No checkpoint found at {checkpoint_path}")
            return model

        ext = os.path.splitext(checkpoint_path)[1].lower()
        if ext == ".safetensors":
            state_dict = load_safetensors(checkpoint_path, device=device)
        elif ext in [".pt", ".pth", ".bin"]:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        model.load_state_dict(state_dict, strict=False)
        
    return model
