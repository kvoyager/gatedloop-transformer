"""
test_export.py
─────────────────────────────────────────────────────────────────────────────
Weight export: a Lightning-style checkpoint (with optimizer state and an
embedded code snapshot) → .safetensors + .json, and load_model on the export
gives the same outputs as on the checkpoint. CPU only, < 1 s.

  python tests/test_export.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatedloop.checkpoint import load_model                   # noqa: E402
from gatedloop.model.lm import ModelConfig, TransformerLM     # noqa: E402
from gatedloop.tools.export_weights import export             # noqa: E402

CFG = dict(vocab_size=256, d_model=64, n_heads=4, n_layers=2, ff_dim=128, max_seq_len=64,
           dropout=0.0, loop_count=2, loop_min=2, loop_max=2, loop_static_kv=True)


def _fake_ckpt(path: Path) -> TransformerLM:
    torch.manual_seed(0)
    model = TransformerLM(ModelConfig(**CFG))
    opt = torch.optim.AdamW(model.parameters())
    model(torch.randint(0, 256, (2, 16)), labels=torch.randint(0, 256, (2, 16))).loss.backward()
    opt.step()
    torch.save({
        "epoch": 0, "global_step": 123,
        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
        "optimizer_states": [opt.state_dict()],
        "hyper_parameters": {"model": CFG, "lr": 1e-3, "domains": ["a", "b"]},
        "code_bytes": b"PK\x03\x04 pretend zip",
        "git": {"commit": "abc", "dirty": False},
    }, str(path))
    return model.eval()


def test_export_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "step000123.ckpt"
        ref = _fake_ckpt(ckpt)
        st, cfg = export(ckpt)
        assert st.name == "step000123.safetensors" and cfg.name == "step000123.json"

        meta = json.loads(cfg.read_text())
        assert meta["model"] == CFG and meta["global_step"] == 123
        assert meta["training"] == {"lr": 1e-3, "domains": ["a", "b"]}
        raw = st.read_bytes()
        assert b"pretend zip" not in raw and b"abc" not in cfg.read_bytes()

        model, hparams = load_model(st)
        assert model.lm_head.weight is model.embed.weight, "tied embeddings lost"
        assert hparams["model"] == CFG and hparams["lr"] == 1e-3
        ids = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            for loops in (2, 4):
                assert torch.equal(model.backbone(ids, loops), ref.backbone(ids, loops))
            assert torch.equal(model(ids).logits, ref(ids).logits)
    print("  export has weights + config only, and reloads to identical outputs — OK")


def main():
    for fn in [test_export_roundtrip]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n✅ all export tests passed")


if __name__ == "__main__":
    main()
