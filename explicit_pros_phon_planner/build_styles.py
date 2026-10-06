import os, sys, torch, numpy as np
sys.path.insert(0, 'src')
from sklearn.cluster import KMeans
from seedvox.prosody_codec import ProsodyCodec

CORPORA = [
    "../autovoc/dataset/train_tokens_libri_prosody_16q_prosody_codec.pt",
    "../autovoc/train_tokens_globe_prosody_codec.pt",
    "../autovoc/dataset/hifitts/train_tokens_hifitts_prosody_codec.pt",
    "../autovoc/dataset/train_tokens_prosody_16q_prosody_codec.pt",
    "../autovoc/dataset/train_tokens_prosody_zp_prosody_codec.pt",
    "../autovoc/dataset/train_tokens_dailytalk_prosody_16q_prosody_codec.pt",
]
CAP_PER_CORPUS = 12000
K = 16
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

dev = 'cuda'
codec = ProsodyCodec(dim=512, num_blocks=32).eval().to(dev)
ck = torch.load('checkpoints/prosody_codec.pt', map_location='cpu', weights_only=False)
codec.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck, strict=False)
codec.eval()
for p in codec.parameters():
    p.requires_grad = False

torch.set_grad_enabled(False)


def pooled(feats):
    z = codec.encode(feats)            # [1, 32, 512]
    return torch.cat([z.mean(1), z.std(1)], dim=-1)   # [1, 1024]


samples = []      # pooled reps
keys = []         # (wav_path, corpus_idx)
per_corpus = []
for ci, path in enumerate(CORPORA):
    if not os.path.exists(path):
        print(f"skip missing {path}")
        continue
    d = torch.load(path, map_location='cpu', weights_only=False)
    data = d['data'] if isinstance(d, dict) and 'data' in d else d
    n = 0
    pcs = []
    for it in data:
        if n >= CAP_PER_CORPUS:
            break
        if 'log_f0_center' not in it or 'e_center' not in it or 'voicing' not in it:
            continue
        f0 = it['log_f0_center'].float()
        e_ = it['e_center'].float()
        vo = it['voicing'].float()
        feats = torch.stack([f0, e_, vo], dim=-1)[None].to(dev)
        samples.append(pooled(feats)[0].cpu())
        keys.append((it.get('wav_path', f'{path}#{n}'), ci))
        n += 1
    per_corpus.append(n)
    print(f"corpus {ci}: {n} sampled", flush=True)

X = torch.stack(samples).numpy()        # [N, 1024]
keys_np = keys
print(f"total {len(X)} reps, dim {X.shape[1]}, per-corpus {per_corpus}")

# Whiten
mu = X.mean(0, keepdims=True)
sd = X.std(0, keepdims=True) + 1e-6
Xw = (X - mu) / sd

# K-means on a capped sample for the fit, then predict all
fit_n = min(40000, len(Xw))
rng = np.random.RandomState(SEED)
idx = rng.choice(len(Xw), fit_n, replace=False)
km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
km.fit(Xw[idx])
labels = km.predict(Xw)

counts = np.bincount(labels, minlength=K)
print("style counts:", counts.tolist())

assignments = {}
for (wp, _), lab in zip(keys_np, labels):
    assignments[wp] = int(lab)

torch.save({
    'k': K,
    'centers_whitened': torch.from_numpy(km.cluster_centers_).float(),
    'whiten_mean': torch.from_numpy(mu[0]).float(),
    'whiten_std': torch.from_numpy(sd[0]).float(),
    'assignments': assignments,
    'per_corpus': per_corpus,
    'seed': SEED,
}, 'checkpoints/style_vocab.pt')
print(f"saved checkpoints/style_vocab.pt ({len(assignments)} assignments)")
