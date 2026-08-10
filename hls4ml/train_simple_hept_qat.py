"""Quantization-aware training (QAT) for the single-layer HEPT Transformer on
tracking-600, using the PQuant-ML library (pip install pquant-ml[torch]) so trained
weight/activation ranges match a target `ap_fixed<total_bits, integer_bits>` HLS
precision -- the same precision convertSimpleHEPT.py converts to (default
ap_fixed<16,6>) -- instead of drifting outside it, which is what caused the poor
PyTorch-vs-HLS numerical correlation seen with plain (non-QAT) trained weights.

See HEPT/src/tracking_trainer_qat.py for the training loop itself (adapted from
https://github.com/Divij-Agarwal-42/HEPT_pquant/blob/pquant-update-v0.0.6-compatibility/src/tracking_trainer_and_quantizer.py).
Like train_simple_hept.py, this trains the full Transformer(n_layers=1) and then
extracts the single HEPT attention block for use by PseudoHEPT.

Separately from PQuant's generic weight/activation quantization, this also trains
HEPTAttention's LSH bucket-sort key to fit inside the fixed [0, 31] range the HLS
bucket_sort_single kernel hardcodes (32 buckets) -- PQuant has no visibility into that
custom, non-nn.Linear computation. See --bucket-lo/--bucket-hi/--range-penalty-* below,
and hept.py's `hls_bucket_range` hook / tracking_trainer_qat.py's RangePenaltyCriterion.

Note: PQuant's add_compression_layers() hardcodes model.to("cuda"), so this
requires a CUDA device.

Usage:
    python train_simple_hept_qat.py --epochs 20 --total-bits 16 --integer-bits 6
"""
import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "HEPT" / "src"))

from HEPT.src.tracking_trainer_qat import run_one_seed_qat  # noqa: E402
from train_simple_hept import extract_attn0_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="QAT-train the single-layer HEPT Transformer on tracking-600.")
    parser.add_argument("-m", "--model", type=str, default="hept")
    parser.add_argument("--epochs", type=int, default=20, help="Override the yaml's num_epochs (default 2000).")
    parser.add_argument(
        "--pretraining-epochs",
        type=int,
        default=0,
        help="Full-precision warm-up epochs before quantization kicks in.",
    )
    parser.add_argument(
        "--total-bits",
        type=int,
        default=16,
        help="Total fixed-point width, matching the HLS ap_fixed<W,I> precision (default: ap_fixed<16,6>).",
    )
    parser.add_argument(
        "--integer-bits",
        type=int,
        default=6,
        help="Integer bits including sign, matching the HLS ap_fixed<W,I> precision.",
    )
    parser.add_argument(
        "--enable-pruning",
        action="store_true",
        help="Also enable PQuant's wanda pruning (off by default -- this script's focus is QAT, not compression).",
    )
    parser.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity if --enable-pruning is set.")
    parser.add_argument(
        "--disable-bucket-range-training",
        action="store_true",
        help="Disable the HEPT LSH bucket-sort range penalty/emulation (on by default). See "
        "HEPT/src/models/attention/hept.py's hls_bucket_range hook and tracking_trainer_qat.py's "
        "RangePenaltyCriterion.",
    )
    parser.add_argument(
        "--bucket-lo", type=float, default=0.0, help="Lower bound of the HLS bucket_sort_single key range."
    )
    parser.add_argument(
        "--bucket-hi", type=float, default=31.0, help="Upper bound of the HLS bucket_sort_single key range (32 buckets)."
    )
    parser.add_argument(
        "--no-bucket-emulate-sort",
        action="store_true",
        help="Keep the overflow penalty but don't also emulate the hardware's clamp+truncate+stable-sort "
        "during training (penalty-only, softer).",
    )
    parser.add_argument(
        "--qw-max",
        type=float,
        default=-6.0,
        help="Cap on w_rpe's pre-exponential term (see prep_qk in hept.py) -- the primary, well-conditioned "
        "mechanism for keeping the LSH hash value in range. Calibrated empirically against a converged "
        "checkpoint; more negative squeezes the coords-scaled term harder.",
    )
    parser.add_argument(
        "--qk-norm-max",
        type=float,
        default=0.1,
        help="Cap on w_q/w_k output per-token norm -- the other well-conditioned mechanism, complementing "
        "--qw-max (which only covers w_rpe's contribution, not w_q/w_k's).",
    )
    parser.add_argument(
        "--final-penalty-weight",
        type=float,
        default=0.0,
        help="Weight for an additional penalty directly on the final (post-exp, post-random-projection) "
        "hash value. Off by default -- empirically badly conditioned and drives oscillation rather than "
        "convergence when combined with --qw-max/--qk-norm-max at meaningful weight (verified). Only use "
        "as a small (<<1) fine-tuning nudge once qw/qk are already under control.",
    )
    parser.add_argument(
        "--range-penalty-weight",
        type=float,
        default=1.0,
        help="Max weight of the bucket-range overflow penalty once fully ramped in.",
    )
    parser.add_argument(
        "--range-penalty-warmup-epochs",
        type=int,
        default=0,
        help="Epochs to train with the penalty weight held at 0 before ramping starts.",
    )
    parser.add_argument(
        "--range-penalty-ramp-epochs",
        type=int,
        default=None,
        help="Epochs over which the penalty weight ramps from 0 to --range-penalty-weight, "
        "starting after --range-penalty-warmup-epochs. Default: num_epochs // 4.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Global gradient-norm clip (torch.nn.utils.clip_grad_norm_), applied whenever bucket-range "
        "training is enabled -- the LSH range penalty backprops through prep_qk's exp(clamp(., max=50)) "
        "term, which can produce very large/unstable gradients without clipping. Pass 0 to disable.",
    )
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
    config["note"] = args.note if args.note is not None else f"{config['note']}_qat{args.total_bits}_{args.integer_bits}"

    config["qat"] = {
        "total_bits": args.total_bits,
        "integer_bits": args.integer_bits,
        "enable_pruning": args.enable_pruning,
        "sparsity": args.sparsity,
        "pretraining_epochs": args.pretraining_epochs,
        "bucket_range_training": not args.disable_bucket_range_training,
        "bucket_lo": args.bucket_lo,
        "bucket_hi": args.bucket_hi,
        "bucket_emulate_sort": not args.no_bucket_emulate_sort,
        "qw_max": args.qw_max,
        "qk_norm_max": args.qk_norm_max,
        "final_penalty_weight": args.final_penalty_weight,
        "range_penalty_weight": args.range_penalty_weight,
        "range_penalty_warmup_epochs": args.range_penalty_warmup_epochs,
        "range_penalty_ramp_epochs": (
            args.range_penalty_ramp_epochs if args.range_penalty_ramp_epochs is not None else max(1, args.epochs // 4)
        ),
        "max_grad_norm": args.max_grad_norm if args.max_grad_norm > 0 else None,
    }

    log_dir = run_one_seed_qat(config)

    best_model_path = Path(log_dir) / "best_model.pt"
    attn0_path = Path(log_dir) / "hept_block0.pt"
    extract_attn0_checkpoint(best_model_path, attn0_path)

    print(f"\nTrained (QAT) Transformer checkpoint : {best_model_path}")
    print(f"Extracted HEPT attn-block ckpt        : {attn0_path}")
    print(f"Use it with: python convertSimpleHEPT.py --checkpoint {attn0_path} --real-data")


if __name__ == "__main__":
    main()
