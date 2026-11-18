"""
smoke_pipeline.py
─────────────────────────────────────────────────────────────────────────────
End-to-end pipeline smoke test on tiny synthetic data: builds a small tokenizer
and a packed .bin, then trains BOTH architectures for a few steps through the
real Hydra entry point and runs the eval + sampling scripts on the result.
Catches wiring bugs (configs, dataloaders, checkpoint round-trip) in under a
minute, with no dataset download and no real training.

  python tests/smoke_pipeline.py
  python tests/smoke_pipeline.py --cpu
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gatedloop.tokenizer import TextTokenizer

ART = ROOT / "artifacts_smoke"
RUNS = ROOT / "runs_smoke"

WORDS = ("the cat sat on a mat and ran to the big red box near the small blue dog "
         "once upon a time a little girl found a tiny key under the old green tree").split()


def build_corpus() -> TextTokenizer:
    ART.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    docs = [" ".join(rng.choice(WORDS, size=int(rng.integers(8, 20)))) + "."
            for _ in range(4000)]
    tok = TextTokenizer.train(iter(docs[:800]), vocab_size=512,
                              save_path=ART / "tokenizer.json", show_progress=False)
    ids: list[int] = []
    for row in tok.encode_batch(docs):
        ids.extend(row)
    arr = np.asarray(ids, dtype=np.uint16)
    split = int(len(arr) * 0.9)
    arr[:split].tofile(ART / "train.bin")
    arr[split:].tofile(ART / "val.bin")
    print(f"  tokenizer vocab={tok.vocab_size} | bins: {arr.size} tokens "
          f"(train {split}, val {arr.size - split})")
    return tok


def run(cmd: list[str], label: str):
    print(f"\n$ {' '.join(cmd[1:])}\n" + "─" * 70)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"{label} FAILED (exit {r.returncode})")


def train(model: str, run_name: str, accelerator: str, extra: list[str]) -> Path:
    run([sys.executable, "-m", "gatedloop.train",
         f"model={model}", f"run_name={run_name}", f"out_dir={RUNS.name}",
         "trainer.max_steps=6", "trainer.val_check_interval=3",
         f"trainer.accelerator={accelerator}", "trainer.devices=1",
         "trainer.precision=32", "trainer.num_sanity_val_steps=0",
         "data.seq_len=64", "dataloader.batch_size=4", "dataloader.num_workers=0",
         f"data.tokenizer_path={ART.name}/tokenizer.json",
         f"data.train_bin={ART.name}/train.bin", f"data.val_bin={ART.name}/val.bin",
         "sample_every=3", "model.max_seq_len=128",
         *extra], f"train {model}")
    ckpts = sorted((RUNS / run_name).glob("version_*/checkpoints/last.ckpt"))
    assert ckpts, f"no checkpoint produced for {model}"
    return ckpts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="run on CPU instead of GPU")
    args = ap.parse_args()
    accel = "cpu" if args.cpu else "gpu"

    build_corpus()

    plain_ckpt = train("plain_tiny", "plain", accel, [])
    looped_ckpt = train("looped_tiny", "looped", accel,
                        ["val_loop_sweep=[1,2,4]", "model.loop_backprop_k=1"])
    print(f"\n  checkpoints: {plain_ckpt}\n               {looped_ckpt}")

    for name, ckpt, loops in [("plain", plain_ckpt, "1"), ("looped", looped_ckpt, "1,2,4")]:
        run([sys.executable, "scripts/eval_ppl.py", "--ckpt", str(ckpt),
             "--bin", f"{ART.name}/val.bin", "--seq-len", "64", "--loops", loops,
             "--batch-size", "4", "--max-batches", "5",
             "--device", "cpu" if args.cpu else "cuda", "--precision", "fp32"],
            f"eval_ppl ({name})")

    run([sys.executable, "scripts/generate.py", "--ckpt", str(looped_ckpt),
         "--tokenizer", f"{ART.name}/tokenizer.json", "--prompt", "the cat",
         "--max-new-tokens", "20", "-n", "2", "--loops", "4",
         "--device", "cpu" if args.cpu else "cuda"], "generate")

    print("\n✅ SMOKE PIPELINE PASSED (both architectures trained, evaluated, sampled)")


if __name__ == "__main__":
    main()
