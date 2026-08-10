"""Train the single-layer HEPT Transformer on tracking-600 and extract its attention
block into a checkpoint loadable by `PseudoHEPT` (see convertSimpleHEPT.py).

`tracking_trans_hept.yaml` already configures a `Transformer(attn_type="hept", n_layers=1,
h_dim=16, ...)` for the tracking-600 dataset. Since n_layers=1, `Transformer.attns[0]` is a
`HEPT` instance with exactly the submodule names `PseudoHEPT.attn` expects (w_q, w_k, w_v,
attn, norm1, norm2, ff, w_rpe) -- so after training we just copy that submodule's weights
out for use in the HLS conversion demo.

This reuses the existing, tested training loop in HEPT/src/tracking_trainer.py (InfoNCE
loss, optimizer/scheduler, accuracy/precision/recall@k metrics) unchanged, only overriding
`data_dir` (to point at the real, already-processed tracking-600 data under `data/tracking`,
rather than the yaml's `../data/` which resolves to an empty scaffold under HEPT/data) and
`num_epochs` (the yaml's default of 2000, at batch_size=1 over 8000 train graphs, is ~16M
optimizer steps -- impractically long for a first run).

Usage:
    python train_simple_hept.py [--epochs 20] [--seed 42] [--device cuda:0] [--note k60_rad256_hept]
"""
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "HEPT" / "src"))

from HEPT.src.tracking_trainer import run_one_seed  # noqa: E402


def extract_attn0_checkpoint(best_model_path, out_path):
    """Copy the `attns.0.*` submodule of a trained Transformer state dict into a
    standalone checkpoint whose keys match `HEPT`'s own submodule names (i.e. loadable
    directly into `PseudoHEPT.attn.load_state_dict(...)`)."""
    state_dict = torch.load(best_model_path, map_location="cpu")
    prefix = "attns.0."
    attn0_state = OrderedDict(
        (key[len(prefix) :], value) for key, value in state_dict.items() if key.startswith(prefix)
    )
    torch.save(attn0_state, out_path)
    return attn0_state


def main():
    parser = argparse.ArgumentParser(description="Train the single-layer HEPT Transformer on tracking-600.")
    parser.add_argument("-m", "--model", type=str, default="hept")
    parser.add_argument("--epochs", type=int, default=20, help="Override the yaml's num_epochs (default 2000).")
    parser.add_argument("--seed", type=int, default=None, help="Override the yaml's seed.")
    parser.add_argument("--device", type=str, default=None, help="Override the yaml's device.")
    parser.add_argument("--note", type=str, default=None, help="Override the yaml's log-dir note.")
    args = parser.parse_args()

    config_path = REPO_ROOT / "HEPT" / "src" / "configs" / "tracking" / f"tracking_trans_{args.model}.yaml"
    config = yaml.safe_load(config_path.open("r").read())

    config["data_dir"] = str(REPO_ROOT / "data")
    config["num_epochs"] = args.epochs
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device
    if args.note is not None:
        config["note"] = args.note

    log_dir = run_one_seed(config)

    best_model_path = Path(log_dir) / "best_model.pt"
    attn0_path = Path(log_dir) / "hept_block0.pt"
    extract_attn0_checkpoint(best_model_path, attn0_path)

    print(f"\nTrained Transformer checkpoint : {best_model_path}")
    print(f"Extracted HEPT attn-block ckpt  : {attn0_path}")
    print(f"Use it with: python convertSimpleHEPT.py --checkpoint {attn0_path}")


if __name__ == "__main__":
    main()
