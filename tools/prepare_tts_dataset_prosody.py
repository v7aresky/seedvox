# prepare_tts_dataset_prosody.py — Add F0 and energy features to train_tokens.pt
#
# Computes per-frame F0 (fundamental frequency) and RMS energy at the same
# frame rate as Mimi speech tokens (12.5 Hz, hop=1920 @ 24kHz).
#
# Usage:
#   # From scratch (manifest → tokens + prosody):
#   python prepare_tts_dataset_prosody.py --manifest dataset/manifest.json \
#       --output dataset/train_tokens.pt --data_dir dataset/wavs
#
#   # Add prosody to an EXISTING train_tokens.pt (re-reads audio from manifest):
#   python prepare_tts_dataset_prosody.py --manifest dataset/manifest.json \
#       --existing dataset/train_tokens.pt --data_dir dataset/wavs
#
# Output format per sample:
#   {
#       'text':         str,
#       'audio_tokens': tensor [1, K, T],    # Mimi tokens
#       'f0':           tensor [T],           # log(F0) in log-Hz, utterance-normalized, unvoiced interpolated
#       'energy':       tensor [T],           # log(RMS energy), utterance-normalized
#       'prosody_wavelet': tensor [T, 12],    # DWT coefficients: 6 F0 scales + 6 energy scales
#   }

import os
import json
import torch
import torchaudio
import argparse
import numpy as np
from tqdm import tqdm
from explicit_pros_phon_planner.utils import PhoneticGenerator

# Mimi constants
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
HOP_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples

# Initialize PhoneticGenerator
phoneme_generator = PhoneticGenerator(backend='g2p_en', phoneme_vocab_size=128)


def extract_f0(waveform, sr=SAMPLE_RATE, hop=HOP_SIZE, f_min=50.0, f_max=600.0):
    """
    Extract log(F0) at Mimi token frame rate.
    
    - Computes F0 at 50 Hz, downsamples to 12.5 Hz via median pooling
    - Converts to log(Hz) 
    - Interpolates unvoiced regions linearly between voiced neighbors
    - Normalizes to zero mean, unit variance per utterance
    
    Args:
        waveform: [1, T_samples] mono audio tensor
    
    Returns:
        f0: [T_tokens] tensor — log(F0) in log-Hz, utterance-normalized, unvoiced interpolated
    """
    fine_hop = 480
    f0_fine = torchaudio.functional.detect_pitch_frequency(
        waveform, sr, frame_time=(fine_hop / sr), 
        freq_low=int(f_min), freq_high=int(f_max)
    ).squeeze(0)  # [T_fine]
    
    # Downsample to token frame rate via median pooling of voiced frames
    ratio = hop // fine_hop
    T_tokens = waveform.shape[1] // hop
    f0_hz = torch.zeros(T_tokens)
    
    for t in range(T_tokens):
        start = t * ratio
        end = min(start + ratio, f0_fine.shape[0])
        if start < f0_fine.shape[0]:
            chunk = f0_fine[start:end]
            voiced = chunk[chunk > f_min]
            if voiced.numel() > chunk.numel() // 2:
                f0_hz[t] = voiced.median()
            else:
                f0_hz[t] = 0.0  # mark as unvoiced
    
    # Convert to log(Hz) — unvoiced frames get 0 temporarily
    voiced_mask = f0_hz > 0
    log_f0 = torch.full_like(f0_hz, 0.0)
    log_f0[voiced_mask] = f0_hz[voiced_mask].log()
    
    # Interpolate unvoiced regions between voiced neighbors
    if voiced_mask.any() and not voiced_mask.all():
        voiced_indices = torch.where(voiced_mask)[0].float()
        voiced_values = log_f0[voiced_mask]
        all_indices = torch.arange(T_tokens).float()
        # Linear interpolation; extrapolate edges with nearest voiced value
        interpolated = torch.from_numpy(
            np.interp(all_indices.numpy(), voiced_indices.numpy(), voiced_values.numpy())
        ).float()
        log_f0 = interpolated
    elif not voiced_mask.any():
        # Entirely unvoiced — fill with a large negative (silence)
        log_f0 = torch.full_like(f0_hz, -1e15)
    
    # Utterance-level normalization: zero mean, unit variance
    mean = log_f0.mean()
    std = log_f0.std().clamp(min=1e-6)
    log_f0 = (log_f0 - mean) / std
    
    return log_f0


def extract_energy(waveform, hop=HOP_SIZE):
    """
    Extract log(RMS energy) at Mimi token frame rate, utterance-normalized.
    
    Args:
        waveform: [1, T_samples] mono audio tensor
    
    Returns:
        energy: [T_tokens] tensor — log(RMS), utterance-normalized (zero mean, unit var)
    """
    wav = waveform.squeeze(0)  # [T_samples]
    T_tokens = wav.shape[0] // hop
    energy = torch.zeros(T_tokens)
    
    for t in range(T_tokens):
        frame = wav[t * hop : (t + 1) * hop]
        energy[t] = frame.pow(2).mean().sqrt()
    
    # Log scale
    log_energy = energy.clamp(min=1e-7).log()
    
    # Utterance-level normalization: zero mean, unit variance
    mean = log_energy.mean()
    std = log_energy.std().clamp(min=1e-6)
    log_energy = (log_energy - mean) / std
    
    return log_energy


DWT_LEVELS = 5
DWT_WAVELET = 'db4'  # Daubechies-4: good for smooth signals like F0/energy


def extract_wavelet_features(f0, energy, n_levels=DWT_LEVELS, wavelet=DWT_WAVELET):
    """
    Compute DWT (Discrete Wavelet Transform) multi-scale decomposition of F0 and energy.
    
    Decomposes each signal into n_levels detail bands + 1 approximation band,
    then upsamples each band back to the original length T for frame-aligned features.
    
    Prosody scale mapping (at 12.5 Hz frame rate):
        Level 1 (~160ms):  micro-prosody (jitter, shimmer)
        Level 2 (~320ms):  syllable-level pitch accent
        Level 3 (~640ms):  word-level stress patterns
        Level 4 (~1280ms): phrase-level intonation
        Level 5 (~2560ms): utterance-level contour
        Approx (>2560ms):  global pitch level / energy baseline
    
    Args:
        f0:     [T] — log(F0), utterance-normalized (from extract_f0)
        energy: [T] — log(energy), utterance-normalized (from extract_energy)
        n_levels: number of DWT decomposition levels
        wavelet: wavelet family name
    
    Returns:
        wavelet_features: [T, 2*(n_levels+1)] — concatenated F0 and energy wavelet bands
                          Channels: [f0_d1, f0_d2, ..., f0_d5, f0_approx,
                                     energy_d1, energy_d2, ..., energy_d5, energy_approx]
    """
    import pywt
    
    T = f0.shape[0]
    
    def _dwt_to_frame_features(signal_np, n_levels, wavelet):
        """Decompose 1D signal and upsample all bands back to original length."""
        # Pad signal to next power of 2 for clean DWT
        pad_len = max(2**n_levels, int(2**np.ceil(np.log2(max(len(signal_np), 2**n_levels)))))
        padded = np.pad(signal_np, (0, pad_len - len(signal_np)), mode='edge')
        
        # DWT decomposition: returns [approx, detail_n, detail_n-1, ..., detail_1]
        coeffs = pywt.wavedec(padded, wavelet, level=n_levels)
        
        # Upsample each coefficient band back to original length T
        bands = []
        # Detail coefficients (high → low frequency): coeffs[1], coeffs[2], ..., coeffs[n_levels]
        for level_idx in range(1, n_levels + 1):
            detail = coeffs[level_idx]
            # Upsample via linear interpolation to T
            upsampled = np.interp(
                np.linspace(0, len(detail) - 1, T),
                np.arange(len(detail)),
                detail
            )
            bands.append(upsampled)
        
        # Approximation coefficients (lowest frequency)
        approx = coeffs[0]
        upsampled_approx = np.interp(
            np.linspace(0, len(approx) - 1, T),
            np.arange(len(approx)),
            approx
        )
        bands.append(upsampled_approx)
        
        return np.stack(bands, axis=-1)  # [T, n_levels+1]
    
    f0_bands = _dwt_to_frame_features(f0.numpy(), n_levels, wavelet)           # [T, 6]
    energy_bands = _dwt_to_frame_features(energy.numpy(), n_levels, wavelet)   # [T, 6]
    
    # Concatenate F0 and energy bands: [T, 12]
    combined = np.concatenate([f0_bands, energy_bands], axis=-1)
    
    # Per-channel normalization (zero mean, unit variance)
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True).clip(min=1e-6)
    combined = (combined - mean) / std
    
    return torch.from_numpy(combined).float()  # [T, 2*(n_levels+1)]


def compute_dataset_stats(data):
    """
    Compute dataset-level statistics for reporting purposes.
    F0 and energy are already per-utterance normalized (zero mean, unit var)
    in the extract functions. This just reports aggregate stats.
    
    Returns dict of stats for saving alongside the dataset.
    """
    all_f0 = []
    all_energy = []
    
    for item in data:
        if 'f0' in item:
            all_f0.append(item['f0'])
        if 'energy' in item:
            all_energy.append(item['energy'])
    
    stats = {'sample_rate': SAMPLE_RATE, 'frame_rate': FRAME_RATE, 'normalization': 'per_utterance'}
    
    if all_f0:
        cat_f0 = torch.cat(all_f0)
        stats['f0_global_mean'] = cat_f0.mean().item()
        stats['f0_global_std'] = cat_f0.std().item()
        print(f"F0 (post-norm):     global mean={stats['f0_global_mean']:.4f}, std={stats['f0_global_std']:.4f}")
    
    if all_energy:
        cat_energy = torch.cat(all_energy)
        stats['energy_global_mean'] = cat_energy.mean().item()
        stats['energy_global_std'] = cat_energy.std().item()
        print(f"Energy (post-norm): global mean={stats['energy_global_mean']:.4f}, std={stats['energy_global_std']:.4f}")
    
    print(f"Dataset: {len(data)} samples, {sum(item['f0'].shape[0] for item in data if 'f0' in item)} total frames")
    return stats


def resolve_audio_path(filepath, data_dir=None):
    """Try to find the audio file, matching prepare_tts_dataset.py behavior."""
    if os.path.exists(filepath):
        return filepath
    if not os.path.isabs(filepath) and data_dir:
        joined = os.path.join(data_dir, filepath)
        if os.path.exists(joined):
            return joined
    if data_dir:
        alt = os.path.join(data_dir, os.path.basename(filepath))
        if os.path.exists(alt):
            return alt
    return None


def load_and_resample(filepath, target_sr=SAMPLE_RATE):
    """Load audio, resample to 24kHz mono."""
    waveform, sr = torchaudio.load(filepath)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform  # [1, T_samples]


def prepare_from_scratch(manifest_path, output_path, data_dir=None, device='cuda', skip_wavelet=False, n_q=16, no_prosody=False):
    """Full pipeline: tokenize + extract prosody (like prepare_tts_dataset.py + prosody)."""
    from seedvox.modules.mimi import get_mimi_model
    
    print(f"Loading Mimi on {device}...")
    mimi = get_mimi_model(device=device, checkpoint_path='pretrained_models/best_mimi.pt', num_codebooks=n_q)
    mimi.eval()
    
    with open(manifest_path, 'r') as f:
        lines = f.readlines()
    
    processed = []
    for line in tqdm(lines, desc="Tokenizing + prosody"):
        item = json.loads(line)
        filepath = resolve_audio_path(item['audio_filepath'], data_dir)
        if filepath is None:
            continue
        
        try:
            waveform = load_and_resample(filepath)
            
            # Skip clips shorter than ~0.5s — too short for Mimi's encoder convolutions
            min_samples = 24000 // 2  # 0.5 seconds at 24kHz
            if waveform.shape[1] < min_samples:
                continue
            
            # Mimi tokenization
            with torch.no_grad():
                audio_tokens = mimi.encode(waveform.unsqueeze(0).to(device))  # [1, K, T]
            
            T_tokens = audio_tokens.shape[2]
            
            sample = {
                'text': item.get('normalized_text', item.get('text', '')),
                'audio_tokens': audio_tokens.cpu(),
            }
            
            # Phoneme generation
            try:
                ph_ids = phoneme_generator.generate_targets(sample['text'])
                sample['ph_ids'] = torch.tensor(ph_ids, dtype=torch.long)
            except Exception as e:
                print(f"Error generating phonemes for {sample['text']}: {e}")
                sample['ph_ids'] = torch.tensor([phoneme_generator.SOS_ID, phoneme_generator.EOS_ID], dtype=torch.long)

            if not no_prosody:
                # Prosody extraction (on CPU)
                f0 = extract_f0(waveform)[:T_tokens]
                energy = extract_energy(waveform)[:T_tokens]
                
                # Pad/trim to exact T_tokens
                if f0.shape[0] < T_tokens:
                    f0 = torch.cat([f0, torch.zeros(T_tokens - f0.shape[0])])
                if energy.shape[0] < T_tokens:
                    energy = torch.cat([energy, torch.zeros(T_tokens - energy.shape[0])])
                
                sample['f0'] = f0
                sample['energy'] = energy
                
                # Wavelet multi-scale decomposition
                if not skip_wavelet:
                    sample['prosody_wavelet'] = extract_wavelet_features(f0, energy)
            processed.append(sample)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    # Report stats
    stats = compute_dataset_stats(processed)
    
    # Save with stats
    output = {
        'data': processed,
        'prosody_stats': stats,
    }
    torch.save(output, output_path)
    print(f"Saved {len(processed)} samples with prosody to {output_path}")


def augment_existing(manifest_path, existing_path, data_dir=None, output_path=None, skip_wavelet=False):
    """Add F0/energy to an existing train_tokens.pt by re-reading audio from manifest."""
    print(f"Loading existing tokens from {existing_path}...")
    raw = torch.load(existing_path, weights_only=False)
    
    # Handle both old format (list) and new format (dict with 'data')
    if isinstance(raw, list):
        existing_data = raw
    elif isinstance(raw, dict) and 'data' in raw:
        existing_data = raw['data']
    else:
        print("Error: Unknown format in existing .pt file")
        return
    
    # Build text→index map for matching
    text_to_idx = {}
    for i, item in enumerate(existing_data):
        text_to_idx[item['text']] = i
    
    print(f"Existing dataset: {len(existing_data)} samples")
    
    with open(manifest_path, 'r') as f:
        lines = f.readlines()
    
    matched, skipped = 0, 0
    for line in tqdm(lines, desc="Extracting prosody"):
        item = json.loads(line)
        text = item.get('normalized_text', item.get('text', ''))
        
        if text not in text_to_idx:
            skipped += 1
            continue
        
        idx = text_to_idx[text]
        
        # Skip if already has prosody
        if 'f0' in existing_data[idx] and 'energy' in existing_data[idx]:
            matched += 1
            continue
        
        filepath = resolve_audio_path(item['audio_filepath'], data_dir)
        if filepath is None:
            skipped += 1
            continue
        
        try:
            waveform = load_and_resample(filepath)
            T_tokens = existing_data[idx]['audio_tokens'].shape[2]
            
            f0 = extract_f0(waveform)[:T_tokens]
            energy = extract_energy(waveform)[:T_tokens]
            
            # Pad to exact T_tokens
            if f0.shape[0] < T_tokens:
                f0 = torch.cat([f0, torch.zeros(T_tokens - f0.shape[0])])
            if energy.shape[0] < T_tokens:
                energy = torch.cat([energy, torch.zeros(T_tokens - energy.shape[0])])
            
            existing_data[idx]['f0'] = f0
            existing_data[idx]['energy'] = energy
            
            if not skip_wavelet:
                existing_data[idx]['prosody_wavelet'] = extract_wavelet_features(f0, energy)
            matched += 1
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            skipped += 1
            continue
    
    print(f"Matched: {matched}, Skipped: {skipped}")
    
    # Check how many have prosody
    with_prosody = sum(1 for d in existing_data if 'f0' in d and 'energy' in d)
    print(f"Samples with prosody: {with_prosody}/{len(existing_data)}")
    
    if with_prosody == 0:
        print("Warning: No prosody features were added!")
        return
    
    # Report stats
    prosody_items = [d for d in existing_data if 'f0' in d and 'energy' in d]
    stats = compute_dataset_stats(prosody_items)
    
    # Save
    save_path = output_path or existing_path
    output = {
        'data': existing_data,
        'prosody_stats': stats,
    }
    torch.save(output, save_path)
    print(f"Saved augmented dataset ({with_prosody} with prosody) to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract F0 and energy prosody features aligned to Mimi token frames."
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to JSON manifest (same as prepare_tts_dataset.py)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for new .pt file (required for --from-scratch)")
    parser.add_argument("--existing", type=str, default=None,
                        help="Path to existing train_tokens.pt to augment with prosody")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Base directory for audio files")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for Mimi tokenization (only used with --output)")
    parser.add_argument("--no-wavelet", action="store_true",
                        help="Skip wavelet feature extraction (faster, F0+energy only)")
    parser.add_argument("--n_q", type=int, default=16,
                        help="Number of codebooks to use (default: 16)")
    parser.add_argument("--no-prosody", action="store_true",
                        help="Skip all prosody extraction (text + tokens only)")
    
    args = parser.parse_args()
    
    if args.existing:
        augment_existing(args.manifest, args.existing, args.data_dir, args.output, skip_wavelet=args.no_wavelet)
    elif args.output:
        prepare_from_scratch(args.manifest, args.output, args.data_dir, args.device, skip_wavelet=args.no_wavelet, n_q=args.n_q, no_prosody=args.no_prosody)
    else:
        parser.error("Provide either --existing (to augment) or --output (from scratch)")
