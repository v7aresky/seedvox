import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


PROMPTS = [
    "Look at synthetic speech today, we're slapping massive brute force autoregressive decoders onto audio context, crossing our fingers and hoping the network guesses intonation while computing pronunciation.",
    "But nature isn't that sloppy! Emotion is a global state of mind, an overarching boundary condition set before we even open our mouths!",
    "By coupling an autoregressive phonetic planner with a JEPA world model to project global expressiveness into a unified latent space we achieve a massive breakthrough.",
    "Students and scholars can now train and run inference for their own speech systems on a single consumer gaming laptop without needing massive compute or giant datasets.",
    "The Belgians approach the game like classical mechanics; it's all about structure, geometry, and predictable trajectories.",
    "They pass the ball, forming beautiful triangles on the grass, like a crystal lattice structure growing across the field.",
    "Enter SeedVox! An open-source speech engine that brings mathematical and structural discipline back to neural audio by separating the phonetic what from the prosodic how.",
    "Best of all, because phonemes are explicitly planned upfront with sub four hundred millisecond planning latencies, the representation is fully intervenable for hand-correcting tricky pronunciations on the fly.",
    "We aren't just scaling up models. We are democratizing the future of speech research for everyone everywhere!",
    "Let's look at this match: the Belgian Red Devils versus the Lions of Teranga from Senegal. Two completely different styles of moving matter through space!",
    "It looks just like a crystal lattice structure growing across the field!",
    "Kevin De Braaneh sits in the midfield like a master experimenter. He looks at the field, calculates the vectors, accounts for the wind resistance, and whack!",
    "He sends a beautiful, spinning pass right into the path of an attacker.",
    "But then, the ball meets the Senegalese defense. And this is where the physics gets really interesting!",
    "The Senegalese players don't care about static geometry. They operate on pure kinetic energy! They have this incredible, explosive speed.",
    "To the Belgians, it must have felt like quantum mechanics.",
    "You think the Senegalese defender is over here, but by the time you look down, he's already over there, occupying three places at once through sheer velocity!",
    "Saadio Maaneh gets the ball on the wing, and he doesn't just run, he accelerates like a particle in a cyclotron!",
    "The ball was passed, passed, and passed again by the patient Belgians.",
    "In 1984, exactly 2,375 people gathered at 11:45 PM and waited.",
    "Kevin De Braaneh passes, Saadio Maaneh accelerates, and the crowd roars.",
    "Nooo, that was never going to work...",
    "bip, bip, bip, bip, bip, bip, bip, bip, bip.",
    "The Senegalese defenders occupy three places at once through sheer velocity, while the Belgians calculate the wind resistance and form a crystal lattice across the field.",
    "No, that was never going to work.",
    "Is that really what happened? I cannot believe it!",
    "This is the greatest day of my entire life!",
]


def tokenize(text):
    words = re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()
    return words


def word_wer(hyp, ref):
    hyp, ref = tokenize(hyp), tokenize(ref)
    if not ref:
        return 0.0
    n, m = len(hyp), len(ref)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if hyp[i - 1] == ref[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    return dp[n][m] / m


def transcribe(wav, model):
    segs, _ = model.transcribe(str(wav), beam_size=5, language="en")
    return " ".join(s.text for s in segs).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/light_fusion_r6_looped2_nostyle.json")
    ap.add_argument("--ref_wav", default="/home/vpollet/proj/autovoc/dataset/wavs/LJ007-0039.wav")
    ap.add_argument("--prompts_file", default=None, help="Optional file of sentences (one per line); overrides built-in set")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--whisper_model", default="small")
    ap.add_argument("--num", type=int, default=None, help="Only first N prompts")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.prompts_file:
        prompts = [
            ln.strip() for ln in open(args.prompts_file, encoding="utf-8")
            if ln.strip()
        ]
    else:
        prompts = PROMPTS[: args.num] if args.num else PROMPTS

    env = dict(os.environ)
    for i, text in enumerate(prompts):
        wav = out / f"synth_{i:02d}.wav"
        if not wav.exists():
            cmd = [
                sys.executable, "-m", "explicit_pros_phon_planner.infer",
                "--text", text,
                "--config", args.config,
                "--checkpoint", args.checkpoint,
                "--ref_wav_speaker", args.ref_wav,
                "--ref_wav_prosody", args.ref_wav,
                "--temp", str(args.temp),
                "--seed", str(args.seed),
                "--use_linguistic_fusion",
                "--g2p", "espeak",
                "--output", str(wav),
            ]
            print(f"[synth] {wav.name} ...", flush=True)
            subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)

    from faster_whisper import WhisperModel
    model = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")

    rows = []
    for i, text in enumerate(prompts):
        hyp = transcribe(out / f"synth_{i:02d}.wav", model)
        wer = word_wer(hyp, text)
        rows.append((i, wer, text[:60], hyp[:60]))
        print(f"WER {wer*100:5.1f}%  ref: {text[:60]!r}")
        print(f"             hyp: {hyp[:80]!r}", flush=True)

    avg = sum(r[1] for r in rows) / len(rows)
    print("=" * 70)
    print(f"AVG WER {avg*100:.1f}%  ({len(rows)} prompts, whisper-{args.whisper_model}, seed {args.seed}, temp {args.temp})")


if __name__ == "__main__":
    main()