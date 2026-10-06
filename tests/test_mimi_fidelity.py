"""
Mimi vocoding fidelity check for SeedVox.
Tests multiple signal paths through Mimi to isolate quality issues.

Usage:
    python tests/test_mimi_fidelity.py --wav /path/to/ref.wav
    python tests/test_mimi_fidelity.py --wav /path/to/ref.wav --checkpoint pretrained_models/best_mimi.pt

Tests:
    1. Token roundtrip:     wav → encode → tokens → decode → wav (standard)
    2. Partial codebooks:   decode with 1, 2, 4, 8 codebooks
    3. Unquantized latent:  wav → encoder+transformer → latent → decoder+transformer → wav (no quantizer)
    4. Quantized latent:    wav → encoder+transformer → quantize → dequantize → latent → decoder+transformer → wav
    5. Decoder-only:        tokens → quantizer.decode → latent → decoder_transformer → decoder → wav (vocoder path)
"""
import torch
import torchaudio
import argparse
import os
import torch.nn.functional as F


def check_fidelity(wav_path, checkpoint='pretrained_models/best_mimi.pt', device='cuda', max_codebooks=16):
    from seedvox.modules.mimi import get_mimi_model

    print(f"Loading Mimi from {checkpoint}...")
    mimi = get_mimi_model(device=device, checkpoint_path=checkpoint)
    mimi.set_num_codebooks(max_codebooks)
    mimi.eval()
    print(f"  num_codebooks: {mimi.num_codebooks}")
    print(f"  total_codebooks: {mimi.quantizer.total_codebooks}")
    print(f"  sample_rate:   {mimi.sample_rate}")
    print(f"  frame_rate:    {mimi.frame_rate}")

    # Load audio
    wav, sr = torchaudio.load(wav_path)
    if sr != 24000:
        wav = torchaudio.transforms.Resample(sr, 24000)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    print(f"\nInput: {wav_path}")
    print(f"  shape: {wav.shape}, duration: {wav.shape[1]/24000:.2f}s")

    wav_gpu = wav.unsqueeze(0).to(device)  # [1, 1, T]

    with torch.no_grad():
        # Encode
        tokens = mimi.encode(wav_gpu)  # [1, K, T_tokens]
        print(f"\nEncoded tokens: {tokens.shape}")
        print(f"  codebooks: {tokens.shape[1]}, frames: {tokens.shape[2]}")
        print(f"  token range: [{tokens.min().item()}, {tokens.max().item()}]")

        # Check codebook utilization
        K = tokens.shape[1]
        for k in range(K):
            unique = tokens[0, k].unique().numel()
            print(f"  codebook {k}: {unique}/2048 codes used ({unique/2048*100:.1f}%)")

        # Decode (full codebooks)
        recon = mimi.decode(tokens)
        print(f"\nReconstructed: {recon.shape}, range: [{recon.min().item():.3f}, {recon.max().item():.3f}]")

        # Also decode with only first N codebooks
        test_codebooks = [1, 2, 4, 8]
        if K >= 12:
            test_codebooks.append(12)
        if K >= 16:
            test_codebooks.append(16)

        print(f"\n  {'n_q':>4s}  {'SNR':>8s}  {'File'}")
        print(f"  {'----':>4s}  {'--------':>8s}  {'----'}")
        for n_q_test in test_codebooks:
            if n_q_test <= K:
                partial_tokens = tokens.clone()
                partial_tokens[:, n_q_test:, :] = 0
                mimi.set_num_codebooks(n_q_test)
                partial_recon = mimi.decode(partial_tokens[:, :n_q_test, :])
                mimi.set_num_codebooks(max_codebooks)

                pr_len = min(wav.shape[1], partial_recon.shape[2])
                pr_cpu = partial_recon[0, 0, :pr_len].cpu()
                pr_orig = wav[0, :pr_len]
                pr_snr = 10 * torch.log10(pr_orig.pow(2).mean() / (pr_cpu - pr_orig).pow(2).mean().clamp(min=1e-10))

                out_name = f"mimi_recon_q{n_q_test}.wav"
                torchaudio.save(out_name, partial_recon[0].cpu().clamp(-1, 1), 24000)
                print(f"  {n_q_test:>4d}  {pr_snr.item():>7.1f} dB  {out_name}")

    # Save full reconstruction and original
    torchaudio.save("mimi_recon_full.wav", recon[0].cpu().clamp(-1, 1), 24000)
    orig_len = min(wav.shape[1], recon.shape[2])
    torchaudio.save("mimi_original.wav", wav[:, :orig_len], 24000)

    recon_cpu = recon[0, 0, :orig_len].cpu()
    orig_cpu = wav[0, :orig_len]

    noise = recon_cpu - orig_cpu
    snr = 10 * torch.log10(orig_cpu.pow(2).mean() / noise.pow(2).mean().clamp(min=1e-10))
    print(f"\n=== Quality Metrics ===")
    print(f"  SNR:           {snr.item():.1f} dB")

    n_fft = 1024
    orig_spec = torch.stft(orig_cpu, n_fft, hop_length=256, win_length=1024, return_complex=True, window=torch.hann_window(1024)).abs()
    recon_spec = torch.stft(recon_cpu, n_fft, hop_length=256, win_length=1024, return_complex=True, window=torch.hann_window(1024)).abs()
    spec_conv = torch.norm(orig_spec - recon_spec, p='fro') / torch.norm(orig_spec, p='fro').clamp(min=1e-10)
    print(f"  Spectral conv: {spec_conv.item():.4f} (lower=better, <0.3 is good)")

    log_spec_dist = (torch.log(orig_spec.clamp(min=1e-7)) - torch.log(recon_spec.clamp(min=1e-7))).pow(2).mean().sqrt()
    print(f"  Log spec dist: {log_spec_dist.item():.4f} (lower=better)")

    # ==================================================================
    # TEST 3: Quantized latent from encoder (training path)
    # ==================================================================
    print(f"\n{'='*60}")
    print("TEST 3: Exact Training Forward Path (replicates compression.py forward)")
    print(f"{'='*60}")

    with torch.no_grad():
        with mimi._context_for_encoder_decoder:
            emb = mimi.encoder(wav_gpu)
        if mimi.encoder_transformer is not None:
            (emb,) = mimi.encoder_transformer(emb)
        print(f"  Step 1 - Encoder output:       {emb.shape}")

        emb = mimi._to_framerate(emb)
        print(f"  Step 2 - After _to_framerate:  {emb.shape}")
        encoder_latent = emb.clone()

        q_res = mimi.quantizer(emb, mimi.frame_rate)
        emb = q_res.x
        quantized_latent = emb.clone()
        print(f"  Step 3 - Quantized (q_res.x):  {emb.shape}")
        print(f"    Pre-VQ range:  [{encoder_latent.min().item():.3f}, {encoder_latent.max().item():.3f}], std={encoder_latent.std().item():.4f}")
        print(f"    Post-VQ range: [{emb.min().item():.3f}, {emb.max().item():.3f}], std={emb.std().item():.4f}")
        print(f"    Quantization RMSE: {(encoder_latent - emb).pow(2).mean().sqrt().item():.4f}")

        emb = mimi._to_encoder_framerate(emb)
        print(f"  Step 4 - After _to_encoder_fr:  {emb.shape}")

        if mimi.decoder_transformer is not None:
            (emb,) = mimi.decoder_transformer(emb)
        print(f"  Step 5 - After dec_transformer: {emb.shape}")

        with mimi._context_for_encoder_decoder:
            q_train_recon = mimi.decoder(emb)
        print(f"  Step 6 - Decoder output:        {q_train_recon.shape}")

        out_name = "mimi_recon_training_path.wav"
        q_train_recon = q_train_recon[..., :wav_gpu.shape[-1]]
        torchaudio.save(out_name, q_train_recon[0].cpu().clamp(-1, 1), 24000)
        print(f"  Saved {out_name}")

        tp_len = min(wav.shape[1], q_train_recon.shape[2])
        tp_recon_cpu = q_train_recon[0, 0, :tp_len].cpu()
        tp_orig_cpu = wav[0, :tp_len]
        tp_snr = 10 * torch.log10(tp_orig_cpu.pow(2).mean() / (tp_recon_cpu - tp_orig_cpu).pow(2).mean().clamp(min=1e-10))
        print(f"  SNR: {tp_snr.item():.1f} dB (training quality ceiling)")

    # ==================================================================
    # TEST 4: Latent from token decode (TTS inference path)
    # ==================================================================
    print(f"\n{'='*60}")
    print("TEST 4: Latent from Token Decode (quantizer.decode — TTS path)")
    print(f"{'='*60}")

    with torch.no_grad():
        latent_from_tokens = mimi.decode_latent(tokens)
        print(f"  Latent from tokens: {latent_from_tokens.shape}")
        print(f"  Range: [{latent_from_tokens.min().item():.3f}, {latent_from_tokens.max().item():.3f}]")

        T_min = min(quantized_latent.shape[2], latent_from_tokens.shape[2])
        lat_diff = (quantized_latent[:, :, :T_min] - latent_from_tokens[:, :, :T_min]).pow(2).mean().sqrt()
        print(f"  Diff vs training-path latent: {lat_diff.item():.6f} (should be ~0)")

    # ==================================================================
    # TEST 5: Noise robustness of latent decode
    # ==================================================================
    print(f"\n{'='*60}")
    print("TEST 5: Latent Noise Robustness (simulates prediction error)")
    print(f"{'='*60}")

    with torch.no_grad():
        latent_std = latent_from_tokens.std().item()
        print(f"  Latent std: {latent_std:.4f}\n")

        for noise_level in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
            noisy_latent = latent_from_tokens + noise_level * latent_std * torch.randn_like(latent_from_tokens)

            emb = mimi._to_encoder_framerate(noisy_latent)
            if mimi.decoder_transformer is not None:
                (emb,) = mimi.decoder_transformer(emb)
            with mimi._context_for_encoder_decoder:
                noisy_recon = mimi.decoder(emb)

            nl_len = min(wav.shape[1], noisy_recon.shape[2])
            nl_recon_cpu = noisy_recon[0, 0, :nl_len].cpu()
            nl_orig_cpu = wav[0, :nl_len]
            nl_snr = 10 * torch.log10(nl_orig_cpu.pow(2).mean() / (nl_recon_cpu - nl_orig_cpu).pow(2).mean().clamp(min=1e-10))

            rel_noise = noise_level * 100
            out_name = f"mimi_recon_noise_{rel_noise:.0f}pct.wav"
            torchaudio.save(out_name, noisy_recon[0].cpu().clamp(-1, 1), 24000)
            print(f"  noise={rel_noise:5.1f}% of std -> SNR: {nl_snr.item():6.1f} dB  ({out_name})")

        print()
        print("  If SNR degrades gracefully: continuous latent prediction is viable")
        print("  If SNR collapses at small noise: decoder is fragile, stick to discrete tokens")

    # ==================================================================
    # TEST 6: Per-codebook latent contribution
    # ==================================================================
    print(f"\n{'='*60}")
    print("TEST 6: Per-Codebook Latent Contribution")
    print(f"{'='*60}")

    with torch.no_grad():
        full_latent = latent_from_tokens

        for k in range(min(tokens.shape[1], 8)):
            single_tokens = torch.zeros_like(tokens)
            single_tokens[:, k, :] = tokens[:, k, :]
            single_latent = mimi.decode_latent(single_tokens)

            contrib_pct = single_latent.pow(2).sum() / full_latent.pow(2).sum() * 100
            print(f"  Codebook {k}: energy contribution = {contrib_pct.item():.1f}%, "
                  f"latent std = {single_latent.std().item():.4f}")

    # ==================================================================
    # TEST 7: Soft Embedding Reconstruction
    # ==================================================================
    print(f"\n{'='*60}")
    print("TEST 7: Soft Embedding Decode (simulates embedding loss path)")
    print(f"{'='*60}")

    with torch.no_grad():
        cb_weights = []
        rvq_first = mimi.quantizer.rvq_first
        rvq_rest = mimi.quantizer.rvq_rest

        cb0 = rvq_first.vq.layers[0].embedding
        cb0_projected = rvq_first.output_proj(cb0.T.unsqueeze(0)).squeeze(0).T
        cb_weights.append(cb0_projected)

        for layer in rvq_rest.vq.layers:
            cb_k = layer.embedding
            cb_k_projected = rvq_rest.output_proj(cb_k.T.unsqueeze(0)).squeeze(0).T
            cb_weights.append(cb_k_projected)

        n_q_test = min(8, len(cb_weights))
        T_tokens_test = tokens.shape[2]
        latent_dim = cb_weights[0].shape[1]
        print(f"  Codebook dim: {cb_weights[0].shape[0]} entries x {latent_dim}d (after output_proj)")

        # a) One-hot soft embedding
        print("  a) One-hot soft embedding (verification):")
        soft_latent_onehot = torch.zeros(1, latent_dim, T_tokens_test, device=device)
        for k in range(n_q_test):
            cb = cb_weights[k]
            one_hot = F.one_hot(tokens[0, k, :].long(), num_classes=cb.shape[0]).float()
            soft_latent_onehot += (one_hot @ cb).transpose(0, 1).unsqueeze(0)

        diff_vs_hard = (soft_latent_onehot - latent_from_tokens).pow(2).mean().sqrt()
        print(f"     Diff vs hard lookup: {diff_vs_hard.item():.6f} (should be ~0)")

        # b) Temperature-softened embeddings
        print("\n  b) Temperature-softened embeddings:")
        for temp in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            soft_latent = torch.zeros(1, latent_dim, T_tokens_test, device=device)
            for k in range(n_q_test):
                cb = cb_weights[k]
                one_hot = F.one_hot(tokens[0, k, :].long(), num_classes=cb.shape[0]).float()
                logits = one_hot * 20.0
                probs = torch.softmax(logits / temp, dim=-1)
                soft_emb = probs @ cb
                soft_latent += soft_emb.transpose(0, 1).unsqueeze(0)

            emb = mimi._to_encoder_framerate(soft_latent)
            if mimi.decoder_transformer is not None:
                (emb,) = mimi.decoder_transformer(emb)
            with mimi._context_for_encoder_decoder:
                soft_recon = mimi.decoder(emb)

            sl_len = min(wav.shape[1], soft_recon.shape[2])
            sl_recon_cpu = soft_recon[0, 0, :sl_len].cpu()
            sl_orig_cpu = wav[0, :sl_len]
            sl_snr = 10 * torch.log10(sl_orig_cpu.pow(2).mean() / (sl_recon_cpu - sl_orig_cpu).pow(2).mean().clamp(min=1e-10))

            latent_rmse = (soft_latent - latent_from_tokens).pow(2).mean().sqrt()
            out_name = f"mimi_recon_soft_temp_{temp:.2f}.wav"
            torchaudio.save(out_name, soft_recon[0].cpu().clamp(-1, 1), 24000)
            print(f"     temp={temp:5.2f} -> latent RMSE: {latent_rmse.item():.4f}, SNR: {sl_snr.item():6.1f} dB  ({out_name})")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'='*60}")
    print("SUMMARY — SNR by path (higher = better)")
    print(f"{'='*60}")
    print(f"  Training path (enc->VQ->dec):   {tp_snr.item():6.1f} dB  <- training quality ceiling")
    print(f"  Token roundtrip (enc->tok->dec): {snr.item():6.1f} dB  <- standard encode->decode")
    print()
    print(f"  Quantization RMSE:    {(encoder_latent - quantized_latent).pow(2).mean().sqrt().item():.4f}")
    print(f"  Token<->training diff:  {lat_diff.item():.6f}")
    print()
    print("Diagnosis:")
    print(f"  If training path >> token roundtrip: encode->decode path has issues")
    print(f"  If training path ~= token roundtrip: codec is consistent (good)")
    print(f"  If noise test degrades gracefully: latent prediction TTS is viable")
    print(f"  If noise test collapses early: stick to discrete token prediction")
    print(f"  If all are low: encoder/decoder capacity is the limit")
    print()
    print("=== Files saved ===")
    print(f"  mimi_original.wav              — original (trimmed)")
    print(f"  mimi_recon_full.wav            — token roundtrip")
    print(f"  mimi_recon_q*.wav              — partial codebook reconstructions")
    print(f"  mimi_recon_training_path.wav   — encoder->VQ->decoder (training path)")
    print(f"  mimi_recon_noise_*.wav         — latent + noise at various levels")
    print(f"  mimi_recon_soft_temp_*.wav     — soft embedding at various temperatures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Mimi encode->decode fidelity")
    parser.add_argument("--wav", type=str, required=True, help="Path to reference wav file")
    parser.add_argument("--checkpoint", type=str, default="pretrained_models/best_mimi.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_codebooks", type=int, default=16)
    args = parser.parse_args()
    check_fidelity(args.wav, args.checkpoint, args.device, max_codebooks=args.max_codebooks)
