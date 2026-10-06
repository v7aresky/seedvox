#!/usr/bin/env python3
"""
track_training.py — epoch-averaged loss trend from a Fusion trainer log.

tqdm refreshes re-print each step several times with \r separators, so we
dedupe by (epoch, step) keeping the LAST occurrence, then aggregate per epoch.
Per-step values are too noisy (length-grouped sampling), epoch means are the
signal: watch `ar` — a sustained downward trend means the decoder is learning
to follow the teacher-forced prosody latent.

Usage:
  python scripts/track_training.py [logfile] [--last N] [--watch ar]
"""

import re
import sys
from collections import defaultdict

STEP_RE = re.compile(
    r"Epoch (\d+) \(Fusion\):.*?(\d+)/(\d+) \[[^\]]*?(\d+\.\d+)it/s, "
    r"ar=([0-9.]+), jepa=([0-9.]+), ph=([0-9.]+), total=([0-9.]+)"
)


def parse(path):
    # universal newlines splits on \r as well, undoing tqdm's refresh spam
    rows = defaultdict(dict)  # epoch -> {step: (ar, jepa, ph, total, its)}
    order = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            ep, step, tot, its, ar, jepa, ph, total = (
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                float(m.group(4)), float(m.group(5)), float(m.group(6)),
                float(m.group(7)), float(m.group(8)),
            )
            if ep not in rows:
                order.append(ep)
            rows[ep][step] = (ar, jepa, ph, total, its)
    return order, rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "training_newcode.log"
    last = None
    watch = "ar"
    for a in sys.argv[2:]:
        if a == "--last":
            last = int(sys.argv[sys.argv.index(a) + 1])
        elif a == "--watch":
            watch = sys.argv[sys.argv.index(a) + 1]

    order, rows = parse(path)
    if not order:
        print(f"No matching log lines in {path}")
        return

    names = {"ar": "ar", "jepa": "jepa", "ph": "ph", "total": "total", "its": "it/s"}
    if watch not in names:
        watch = "ar"
    idx = {"ar": 0, "jepa": 1, "ph": 2, "total": 3, "its": 4}[watch]

    order = order[-last:] if last else order
    prev = None
    print(f"{'epoch':>5} {'steps':>6} {'ar':>7} {'jepa':>7} {'ph':>7} {'total':>7} "
          f"{'it/s':>6}   delta {watch} (vs prev epoch)")
    for ep in order:
        steps = rows[ep]
        n = len(steps)
        vals = {k: [r[k] for r in steps.values()] for k in range(5)}
        means = {k: sum(v) / len(v) for k, v in vals.items() if v}
        delta = ""
        if prev is not None:
            delta = f"{means[idx] - prev:+.4f}"
        print(f"{ep:>5} {n:>6} {means[0]:>7.3f} {means[1]:>7.3f} {means[2]:>7.3f} "
              f"{means[3]:>7.3f} {means[4]:>6.2f}   {delta}")
        prev = means[idx]

    if prev is not None:
        print()
        print(f"Most recent epoch {order[-1]}: mean {watch} = {prev:.4f}")
        print(f"(vs epoch {order[0]}: {rows[order[0]][next(iter(rows[order[0]]))][idx]:.4f} at step 0)")


if __name__ == "__main__":
    main()
