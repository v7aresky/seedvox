"""Natural contour-scaling pitch shift for exemplar targets.

Real contour scaling: target absolute log-F0 = mean + s*(lf-mean), equivalent to
a TIME-VARYING pitch shift of +12*(s-1)*(lf-mean)/ln2 semitones per frame. This
shifts real speech (timbre/content preserved) instead of synthesizing harmonics,
so the re-encoded exemplar tokens stay close to the originals (small,
content-preserving edits the operator can actually learn to generalize).

Implementation: per-hop librosa.pitch_shift with Hann overlap-add (PSOLA-style).
"""
import sys
import os
import numpy as np
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

SR = 24000
HOP = 1920


def shift_semitones(lf, vo, s, max_st=7.0, smooth=7):
    """Per-frame semitone shifts realizing contour scaling s. lf [T] natural-log
    F0, vo [T] voiced mask. Returns [T] numpy."""
    lf = np.asarray(lf, dtype=np.float64)
    vo = np.asarray(vo, dtype=np.float64)
    m = (lf * vo).sum() / max(vo.sum(), 1e-6)
    st = 12.0 * (s - 1.0) * (lf - m) / np.log(2.0)
    st = np.clip(st, -max_st, max_st)
    k = np.ones(smooth) / smooth
    return np.convolve(st, k, mode='same')


def timev_pitch_shift(wav, lf, vo, s, hop=3840, win_ratio=4, max_st=7.0):
    """wav: 1D audio (24 kHz). lf [T], vo [T]. Returns shifted 1D numpy."""
    wav = np.asarray(wav).squeeze().astype(np.float64)
    N = wav.shape[0]
    st = shift_semitones(lf, vo, s, max_st=max_st)
    win = hop * win_ratio
    out = np.zeros(N)
    wsum = np.zeros(N)
    hw = np.hanning(win)
    for h in range(0, N, hop):
        seg = wav[h:h + win]
        if seg.shape[0] < win:
            break
        fi = int(round((h + hop // 2) / HOP))
        fi = min(fi, len(st) - 1)
        seg_s = librosa.effects.pitch_shift(seg, sr=SR, n_steps=float(st[fi]),
                                            bins_per_octave=24)
        out[h:h + win] += seg_s * hw
        wsum[h:h + win] += hw
    m = wsum > 1e-8
    out[m] /= wsum[m]
    return out
