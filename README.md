# gatedloop-transformer

Code, configurations and run logs for

> **Loop-Count Schedules in Looped Transformers: An Empirical Study**
> Konstantin Kolokolov, 2026. [arXiv:2611.29704](https://arxiv.org/abs/2611.29704) [cs.LG]

Paper: [abstract](https://arxiv.org/abs/2611.29704) · [PDF](https://arxiv.org/pdf/2611.29704).
Trained weights and per-token traces:
[huggingface.co/konstank/gatedloop-transformer](https://huggingface.co/konstank/gatedloop-transformer).

A looped transformer applies one block several times with shared weights. This
repository trains small looped language models (46M–110M parameters, 2B tokens,
one consumer GPU) in which every pass after the first is scaled by two learned
scalar gates, one on the attention update and one on the feed-forward update,
and studies how the **schedule of the loop count R during training** decides
what the loop learns: an extra layer (fixed R, or R sampled from a range that
starts at 2) or a damped iteration (R sampled from a range that includes 1).
It also measures per-token convergence across passes in hidden-state and output
space, and a forward **static-KV** variant that computes keys and values once
on the first pass and reuses them.

Everything in the paper (thirteen runs, two seeds on the four headline runs,
per-token convergence traces) can be regenerated from this repository with the
commands below. Built on PyTorch Lightning and Hydra.

---

## The model

A pre-norm decoder (RMSNorm, RoPE, QK-norm, SwiGLU, tied embeddings). Each
block is applied R times to its input. Pass 1 is a plain block; passes 2…R are
gated:

```
pass 1 (ungated)                       passes 2 … R (gated, same weights)
x = x + Attn(Norm1(x))                 x = x + α_attn · Attn(Norm1(x))
x = x + FFN(Norm2(x))                  x = x + α_ff   · FFN(Norm2(x))
```

`α = σ(θ)`, one `θ` per gate per block, initialised at 0.5 and excluded from
weight decay. R = 1 is exactly the plain model, so a looped block adds two
scalars per block and nothing else (`tests/test_model.py::test_param_parity`).

Three loop-count schedules are compared, all trained with the same data,
optimizer and 2B-token budget:

| schedule | config suffix | R during training | validated at |
|---|---|---|---|
| fixed | *(none)* | R = 2 on every step | R ∈ {1, 2, 3, 4, 6} |
| sampled, lower bound 1 | `_rs` | R ~ U{1, …, 6}, drawn per micro-batch | R ∈ {1, …, 6, 8} |
| sampled, lower bound 2 | `_r234` | R ~ U{2, 3, 4}, drawn per micro-batch | R ∈ {1, …, 6, 8} |

Each schedule exists with fresh KV (keys and values recomputed on every pass)
and with static KV (`_skv`: K and V from pass 1 reused, only Q recomputed).

The learned gates are logged on every training step as
`gate/alpha_attn_l{ℓ}` / `gate/alpha_ff_l{ℓ}`; they are the paper's main
observable.

---

## Install

```bash
git clone https://github.com/kvoyager/gatedloop-transformer.git
cd gatedloop-transformer
pip install -e .          # or: pip install -r requirements.txt
```

Clone the repo for anything beyond training: the data preparation, evaluation
and analysis tools in `scripts/` and the tests in `tests/` are not part of the
installed package. Installing straight from GitHub gives the package and the
`gatedloop-train` command only:

```bash
pip install "gatedloop-transformer @ git+https://github.com/kvoyager/gatedloop-transformer.git"
```

Relative paths in the configs (`./artifacts/...`, `./runs_ladder`) resolve
against the repo root in a clone, and against the directory you launch from
for an installed copy. Set `GATEDLOOP_ROOT` to anchor them elsewhere.

Tested on Python 3.13, torch 2.11 (CUDA 12.8), Lightning 2.6, one RTX 5080
(16 GB). All paper runs fit in 16 GB with the memory presets in
`gatedloop/configs/memory/`. The exact package versions used for the
experiments are listed at the top of `requirements.txt`.

On Windows, `pip install torch` gives a CPU-only build, so install the CUDA
build first:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

---

## Data

The corpus is an 80 / 10 / 10 token mixture of FineWeb-Edu (sample-10BT),
Python from `codeparrot-clean`, and FineMath-4+, described in one file so that
tokenizer, corpus and training shares cannot drift. The prep scripts live in
`scripts/`, so run them from a clone of the repo:

```bash
python scripts/train_tokenizer.py --mix gatedloop/configs/data/mix_fwe_code_math.yaml   # BPE 32k on a proportional sample
python scripts/prepare_mix.py     --data gatedloop/configs/data/mix_fwe_code_math.yaml   # → artifacts/mix/<domain>_{train,val}.bin + manifest.json
```

Each domain is tokenized into its own `.bin` pair; 1M validation tokens per
domain are held out before training. Documents are `<bos>`-separated and packed
into 1024-token sequences. `prepare_mix.py --total-tokens 5e7` builds a small
corpus first to check the pipeline.

---

## Reproducing the paper

`gatedloop/configs/train_mix.yaml` is the single recipe shared by every run:
AdamW (β = 0.9 / 0.95), peak lr 6e-4, 2,000 warmup steps, cosine to 10 %,
weight decay 0.1 on matrices only, bf16, 32,768 tokens per optimizer step,
61,000 steps = 2.0B tokens, seed 42. A run differs from the others only in
`model=<name>`.

### The thirteen runs (seed 42)

```bash
# depth ladder, plain
python -m gatedloop.train model=plain_3
python -m gatedloop.train model=plain_6
python -m gatedloop.train model=plain_12

# 6-block looped: three schedules × fresh / static KV
python -m gatedloop.train model=looped_6
python -m gatedloop.train model=looped_6_skv
python -m gatedloop.train model=looped_6_rs
python -m gatedloop.train model=looped_6_skv_rs
python -m gatedloop.train model=looped_6_r234
python -m gatedloop.train model=looped_6_skv_r234

# 3-block looped: fixed and U{1,…,6} × fresh / static KV
python -m gatedloop.train model=looped_3
python -m gatedloop.train model=looped_3_skv
python -m gatedloop.train model=looped_3_rs
python -m gatedloop.train model=looped_3_skv_rs
```

Each run lands in `runs_ladder/<model>/version_N/` with TensorBoard events,
`metrics.csv` (every step: loss, gates, throughput, peak memory),
`val_metrics.csv` (every 2,000 steps: per-domain loss at every R in the sweep),
`train.log`, a zipped code snapshot, and checkpoints (`best.ckpt`, `last.ckpt`,
`stepNNNNNN.ckpt`). Wall-clock on one RTX 5080 is roughly 1 day for `plain_6`,
2.5 days for `looped_6`, and 3–3.5 days for the sampled-R runs.

### The second seed

The four headline runs were repeated with seed 43 by overriding the seed and
the run name; there are no separate configs for them:

```bash
python -m gatedloop.train model=plain_6       seed=43 run_name=plain_6_s43
python -m gatedloop.train model=looped_6      seed=43 run_name=looped_6_s43
python -m gatedloop.train model=looped_6_r234 seed=43 run_name=looped_6_r234_s43
python -m gatedloop.train model=looped_6_rs   seed=43 run_name=looped_6_rs_s43
```

### Resuming

```bash
python -m gatedloop.train model=looped_6_r234 seed=43 run_name=looped_6_r234_s43 \
    resume=runs_ladder/looped_6_r234_s43/version_0/checkpoints/last.ckpt
```

A resumed run continues in a new `version_N` directory; `val_metrics.csv` of
the two directories concatenate without overlap.

### Convergence traces

During training, `val_convergence` records on the first two validation
batches of each domain, per token, per block and per pass: the relative step
`rel_i = ‖s_i − s_{i−1}‖ / ‖s_{i−1}‖`, the direction change `1 − cos(s_i,
s_{i−1})`, the successive-output KL `KL(p_i ‖ p_{i−1})` read through the final
norm and the tied head, and the per-pass target NLL. Arrays go to
`<run dir>/convergence/step*_<domain>.npz` with shape `(N, T, L, R)`; scalar
summaries go to `val/conv/*`.

Runs trained before the KL was added (`val_convergence.output_kl`) can be
traced offline from a checkpoint without retraining:

```bash
python -m gatedloop.tools.trace_ckpt --ckpt runs_ladder/looped_6/version_0/checkpoints/step060000.ckpt --loops 8
```

### Evaluation

```bash
# per-domain loss at each R for a checkpoint, plus the share-weighted row
python scripts/eval_ppl.py --ckpt runs_ladder/looped_6_r234/version_0/checkpoints/best.ckpt \
    --data gatedloop/configs/data/mix_fwe_code_math.yaml --loops 1,2,3,4,5,6,8

# sample text at a chosen loop count
python scripts/generate.py --ckpt runs_ladder/looped_6_r234/version_0/checkpoints/best.ckpt \
    --prompt "def fibonacci(n):" -n 4 --loops 3
```

For release, checkpoints are stripped to the weights: `<stem>.safetensors`
(model state_dict only, no optimizer state, no pickled code) plus `<stem>.json`
(model config and training hyper-parameters), written next to the `.ckpt`.
Every script above accepts the `.safetensors` file in place of `--ckpt`.

```bash
python -m gatedloop.tools.export_weights runs_ladder/looped_6/version_0/checkpoints/step060000.ckpt
```

The validation loss reported in the paper is `val/loss`: the mean of the three
per-domain losses weighted by the training shares. `val/loss_<domain>_r{R}` is
the loss-versus-R curve per domain.

### Analysis

`runs_ladder/analysis/` holds the markdown analyses and figures the paper's
tables were built from (`runs_ladder_analysis.md`, `static_kv_addendum.md`,
`sampled_R_analysis.md`, `round4_r234_analysis.md`, `seed43_replication.md`,
and the figure sources).

---

## Run control

A run can be paused, resumed and stopped without killing it
(`TrainControlCallback`). In the console, `p` pauses or resumes, `s` or Ctrl-C
finishes the step, saves `last.ckpt` and exits. From another terminal, create
`<run dir>/PAUSE` to hold the loop (delete it to resume) or `<run dir>/STOP`
to save and stop. The throughput window is reset on resume so paused time does
not appear as a `train/tok_per_s` dip; `train/paused_min` records it.

`train.log` mirrors the console output of the run (resolved config, model
banner, validation summaries, samples, warnings) with ANSI codes stripped.

---

## Configuration

Every option is a Hydra override on the command line (`model.loop_count=3`,
`optimizer.lr=4e-4`, …). The ones the paper varies:

| parameter | meaning |
|---|---|
| `model.loop_count` | R at evaluation and, when sampling is off, in training. `1` = plain model. |
| `model.loop_sample`, `loop_min`, `loop_max` | draw R ~ U{min, …, max} independently for each micro-batch during training |
| `model.loop_static_kv` | compute K and V on pass 1 and reuse them; later passes re-project only Q |
| `model.loop_gate`, `loop_gate_init` | the two learned gates per block and their initial value (0.5 in every paper run) |
| `model.loop_target` | `block` (attention and FFN loop together; every paper run) or `attn` (attention only, FFN once after the loop; kept for the loop-body ablation in future work) |
| `model.loop_layers`, `loop_backprop_k`, `loop_step_embed` | which blocks loop; truncated backpropagation depth; per-pass input bias. All at their defaults (all blocks, full gradients, off) in the paper. |
| `val_loop_sweep` | R values evaluated at every validation (`configs/sweep/ladder.yaml` for fixed-R runs, `sampled.yaml` for sampled-R runs; selected by the model config) |
| `val_convergence.n_batches`, `.loops`, `.output_kl` | the convergence trace: batches per domain, passes to trace, whether to add the output-space metrics |
| `memory=default` / `half` | micro-batch × accumulation (8 × 4 or 4 × 8); both are 32,768 tokens per step. Sampled-R runs at R = 4+ need `half` on 16 GB. |

**Truncated BPTT** (`loop_backprop_k`) runs the skipped passes under `no_grad`
and splices their net update back as a constant, so the forward value is
unchanged and the bottom of the stack keeps its gradient path
(`tests/test_model.py::test_truncated_bptt`). Not used in the paper.

---

## Layout

```
gatedloop/
  tokenizer.py            byte-level BPE (<pad>/<bos>/<eos>)
  optim.py                AdamW with decoupled weight-decay groups (gates excluded) + cosine-warmup schedule
  checkpoint.py           rebuild a bare model from a Lightning .ckpt or an exported .safetensors
  paths.py                where relative config paths resolve (clone vs installed copy)
  model/
    layers.py             RMSNorm, SwiGLU, RoPE, causal SDPA attention with QK-norm
    block.py              the looped block: ungated pass 1, gated passes 2…R, static KV, LoopTrace
    lm.py                 TransformerLM, ModelConfig, per-micro-batch R sampling, trace_loop()
  data/
    text.py               packed uint16 memmap dataset + MixtureDataset
    mixture.py            multi-domain spec, Hub streaming, per-domain bin writer
    loader.py             DataLoader construction
  training/
    lit_module.py         LightningModule: loss, schedule, loop sweep at validation
    callbacks.py          convergence trace, gate logging, throughput, samples, run control, CSV, code snapshot
  train.py                Hydra entry point (default config: train; the paper recipe is --config-name train_mix)
  tools/
    trace_ckpt.py         offline convergence trace (incl. output KL) from a checkpoint
    export_weights.py     .ckpt → .safetensors + .json (weights and config only) for release
  configs/
    train_mix.yaml        the shared recipe
    model/                the thirteen paper configs + plain_tiny / looped_tiny for tests
    sweep/  memory/  trainer/  optimizer/  dataloader/  data/
scripts/                  train_tokenizer · prepare_mix · prepare_data · eval_ppl · generate
tests/                    test_model.py · test_data.py · test_callbacks.py · test_paths.py · test_export.py (unit) · smoke_pipeline.py (end-to-end)
runs_ladder/              run directories + analysis/
```

## Tests

```bash
python tests/test_model.py        # parity, causality, RoPE, truncated BPTT, loop trace, gate init
python tests/test_data.py         # mixture bins, MixtureDataset, per-domain validation
python tests/test_callbacks.py    # PAUSE / STOP run control, one text sample per step
python tests/test_paths.py        # path resolution in a clone vs an installed copy
python tests/test_export.py       # .ckpt → .safetensors export reloads to identical outputs
python tests/smoke_pipeline.py    # trains plain_tiny and looped_tiny end-to-end on synthetic data
```

## Citation

```bibtex
@misc{kolokolov2026loopcount,
  title         = {Loop-Count Schedules in Looped Transformers: An Empirical Study},
  author        = {Kolokolov, Konstantin},
  year          = {2026},
  eprint        = {2611.29704},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.29704}
}
```

## License

MIT.
