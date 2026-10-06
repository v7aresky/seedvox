"""Feasibility: can the mimi codec + frozen estimator represent an s-scaled
contour (log_f0c * s around the mean)?

Synthesizes audio whose absolute log-F0 trace is exactly mean + s*(logF0-mean)
(harmonic source + original amplitude envelope + dataset voicing gate), then
checks whether est() reads back s*exc and whether mimi encode->decode preserves it.

Run: python explicit_pros_phon_planner/feasibility_contour_scale.py
"""
import sys, os, torch, torch.nn.functional as F
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'src')
from seedvox.modules.mimi import get_mimi_model
from explicit_pros_phon_planner.f0_estimator import F0Estimator

def _load():
    dev = 'cuda'
    torch.set_grad_enabled(False)
    mimi = get_mimi_model(device=dev, checkpoint_path='pretrained_models/best_mimi.pt').eval()
    est = F0Estimator().eval().to(dev)
    est.load_state_dict(torch.load('checkpoints/f0_estimator_best.pt', map_location='cpu', weights_only=False)['model'])
    return mimi, est, dev


SR = 24000
HOP = 1920


def _load():
    dev = 'cuda'
    torch.set_grad_enabled(False)
    mimi = get_mimi_model(device=dev, checkpoint_path='pretrained_models/best_mimi.pt').eval()
    est = F0Estimator().eval().to(dev)
    est.load_state_dict(torch.load('checkpoints/f0_estimator_best.pt', map_location='cpu', weights_only=False)['model'])
    return mimi, est, dev


def frame_rms_env(wav, T):
    e = torch.zeros(T, device=wav.device)
    for t in range(T):
        e[t] = wav[t * HOP:(t + 1) * HOP].square().mean().sqrt()
    e = e.clamp(min=1e-4)
    k = torch.ones(7, device=wav.device) / 7
    return F.conv1d(e[None, None], k[None, None], padding=3)[0, 0]


def synth_contour(wav, lf, vo, T, s, K=18):
    m = (lf * vo).sum() / vo.sum().clamp(min=1)
    tgt_abs = m + s * (lf - m)          # absolute log-F0 target trace
    f_hz = tgt_abs.exp()                # [T] Hz
    e = frame_rms_env(wav, T)
    e = torch.stack([e] * (HOP // 2 + 1), dim=1).reshape(-1)[:T * HOP // 2 + 1]
    env = F.interpolate(e[None, None], size=T * HOP, mode='linear')[0, 0]
    v = torch.stack([vo] * (HOP // 2 + 1), dim=1).reshape(-1)[:T * HOP // 2 + 1]
    v = F.interpolate(v[None, None], size=T * HOP, mode='linear')[0, 0]
    phase = torch.zeros(T * HOP, device=wav.device)
    f_per = F.interpolate(tgt_abs.exp()[None, None], size=T * HOP, mode='linear')[0, 0]
    phase[1:] = torch.cumsum(f_per[:-1] / SR * 2 * np.pi, 0)
    har = torch.zeros_like(phase)
    for k in range(1, K + 1):
        har.add_(torch.sin(k * phase) / k)
    noise = torch.randn_like(phase) * 0.35
    x = env * (v * har + (1 - v) * noise)
    x = x * (wav.square().mean().sqrt() / (x.square().mean().sqrt() + 1e-8))
    return x


def exc_of(wav, est):
    lf = est(wav)[0][0]
    vo = est(wav)[1].sigmoid()[0]
    v = (vo > 0.5).float()
    if v.sum() < 4:
        return float('nan'), float('nan')
    lf_c = lf - (lf * v).sum() / v.sum().clamp(min=1)
    e = (lf_c * v).square().sum().div(v.sum().clamp(min=1)).sqrt().item()
    return e, v.sum().item()


if __name__ == '__main__':
    mimi, est, dev = _load()
    d = torch.load('../autovoc/dataset/train_tokens_prosody_zp_prosody_codec.pt', map_location='cpu', weights_only=False)
    items = d['data'] if isinstance(d, dict) else d
    items = [it for it in items if 'audio_tokens' in it and it['audio_tokens'].shape[-1] <= 220]

    for it in items[:4]:
        tok = it['audio_tokens'].to(dev)
        vo = it['voicing'].float().to(dev)
        T = tok.shape[-1]
        N = T * HOP
        wav0 = mimi.decode(tok)[:, :, :N]
        lf0, vlog = est(wav0)
        vgt = (vo > 0.5).float()
        m = (lf0[0] * vgt).sum() / vgt.sum().clamp(min=1)
        exc0 = (lf0[0] - m)[vgt.bool()].square().mean().sqrt().item()
        print(f'T={T}  est-exc of decoded audio = {exc0:.4f}')
        for s in [0.6, 0.8, 1.0, 1.2, 1.5, 1.8]:
            x = synth_contour(wav0[0, 0], lf0[0], vo, T, s)
            e_s, _ = exc_of(x[None, None], est)
            try:
                tok2 = mimi.encode(x[None, None])
                if tok2.dim() == 4:
                    tok2 = tok2[0]
                tok2 = tok2[:, :, :T] if tok2.shape[-1] >= T else tok2
                wav2 = mimi.decode(tok2)[:, :, :N]
                e_rt, _ = exc_of(wav2, est)
                r_rt = e_rt / max(exc0, 1e-4)
            except Exception as ex:
                e_rt = float('nan'); r_rt = float('nan')
            print(f'   s={s:.1f}: synth-exc={e_s:.4f} (resp {e_s/max(exc0*s,1e-4):.3f})  '
                  f'roundtrip-exc={e_rt:.4f} (resp {r_rt/s:.3f})  exc_ratio/1 = {r_rt:.3f}')
        print()
