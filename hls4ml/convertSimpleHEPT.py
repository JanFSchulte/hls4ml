"""Minimal end-to-end demo of the HEPT attention contrib kernel in hls4ml.

It builds a single-layer HEPT attention block in PyTorch, converts it to an HLS
project with the `hls4ml.contrib.hept` extension, runs the C-simulation, and
compares the HLS output against the PyTorch reference element-by-element.

Note on scope: the generated HLS kernel implements only the attention sub-block
    norm1 -> q/k/v projections -> HEPT attention -> out_linear   (== `aggr_out`)
NOT the full transformer block (input residual + norm2 + feed-forward + second
residual). The reference below therefore computes just `aggr_out` so the two
sides are comparing the same quantity.

Usage:
    python convertSimpleHEPT.py [-m hept] [--seed 0]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

import hls4ml
from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model
from hls4ml.contrib.hept.torch import parse_hept_attention_layer
from hls4ml.contrib.hept.registration import register

from HEPT.src.models.baselines.transformer import HEPT
from HEPT.src.models.model_utils.hash_utils import get_regions, compute_combined_shifts
from HEPT.src.datasets.tracking import Tracking, TrackingTransform
from HEPT.src.utils.get_model import get_model

# Demo input dimensions.
SEQ_LEN = 600
COORDS_DIM = 6
PAR_FACTOR = 5
OUTPUT_DIR = "hept_test"
BACKEND = "Vitis"
IO_TYPE = "io_stream"


class PseudoHEPT(nn.Module):
    """Wraps a single HEPT attention block with an HLS-friendly forward signature."""

    def __init__(self, attn_type, coords_dim, **kwargs):
        super().__init__()
        self.attn = HEPT(attn_type, coords_dim, **kwargs)
        self.regions = nn.Parameter(
            get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]),
            requires_grad=False,
        )
        self.block_size = kwargs["block_size"]

    def forward(self, data, coords, combined_shifts):
        # data:            (seq_len, embed_dim)
        # coords:          (seq_len, coords_dim)
        # combined_shifts: (seq_len, num_heads * num_hashes)  — precomputed region indices
        return self.attn(data, coords, combined_shifts)


def reference_attention(model, x, coords, combined_shifts):
    """PyTorch reference for the attention sub-block the HLS kernel implements (`aggr_out`)."""
    hept = model.attn
    with torch.no_grad():
        x_normed = hept.norm1(torch.tensor(x))  # pe_type == "none" -> no positional encoding
        q = hept.w_q(x_normed)
        k = hept.w_k(x_normed)
        v = hept.w_v(x_normed)
        out = hept.attn(
            q, k, v,
            coords=torch.tensor(coords),
            combined_shifts=torch.tensor(combined_shifts),
            raw_size=x.shape[0],
            key_padding_mask=None,
            edge_index=None,
            w_rpe=hept.w_rpe,
        )
    return out.detach().numpy()


def load_real_event(args, model_kwargs):
    """Encode a real tracking-600 event's raw features/coords through the trained
    feat_encoder, truncated to exactly SEQ_LEN nodes.

    Truncating (rather than padding) keeps raw_size == SEQ_LEN, so there's no
    padding-token masking to reconcile: the HLS kernel has no notion of padding at
    all (it treats every one of its SEQ_LEN slots as a real node), while the
    PyTorch reference masks tokens past raw_size. Picking an event with >= SEQ_LEN
    real nodes sidesteps that mismatch entirely.
    """
    full_ckpt_path = args.full_checkpoint or str(Path(args.checkpoint).parent / "best_model.pt")
    full_model = get_model(f"trans_{args.model}", model_kwargs, "tracking-600")
    full_model.load_state_dict(torch.load(full_ckpt_path, map_location="cpu"))
    full_model.eval()

    dataset = Tracking(Path("data/tracking"), dataset_name="tracking-600", transform=TrackingTransform())

    event_idx = args.event_idx
    if event_idx is None:
        for idx in dataset.idx_split["test"].tolist():
            if dataset[idx].x.shape[0] >= SEQ_LEN:
                event_idx = idx
                break
        if event_idx is None:
            raise RuntimeError(f"No test-split event with >= {SEQ_LEN} nodes found.")

    event = dataset[event_idx]
    if event.x.shape[0] < SEQ_LEN:
        raise ValueError(f"Event {event_idx} has only {event.x.shape[0]} nodes (< SEQ_LEN={SEQ_LEN}).")

    print(f"Using real tracking-600 event idx={event_idx} ({event.x.shape[0]} nodes, truncated to {SEQ_LEN}).")

    with torch.no_grad():
        encoded_x = full_model.feat_encoder(event.x[:SEQ_LEN])
    coords_real = event.coords[:SEQ_LEN]

    x = encoded_x.unsqueeze(0).numpy().astype(np.float32)
    coords = coords_real.unsqueeze(0).numpy().astype(np.float32)
    return x, coords


def report(reference, hls_out):
    """Print a concise numerical comparison of the two outputs."""
    ref = reference.astype(np.float64)
    hls = hls_out.astype(np.float64)
    diff = hls - ref
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / np.maximum(np.abs(ref), 1e-12)
    corr = np.corrcoef(hls.ravel(), ref.ravel())[0, 1]

    print("\n=== PyTorch vs HLS numerical comparison ===")
    print(f"output shape        : {reference.shape}")
    print(f"max |abs diff|      : {abs_diff.max():.3e}")
    print(f"mean |abs diff|     : {abs_diff.mean():.3e}")
    print(f"RMS abs diff        : {np.sqrt(np.mean(diff ** 2)):.3e}")
    print(f"max relative diff   : {rel_diff.max():.3e}")
    print(f"mean relative diff  : {rel_diff.mean():.3e}")
    print(f"ref range [min,max] : [{ref.min():.3e}, {ref.max():.3e}]")
    print(f"Pearson correlation : {corr:.6f}")
    print("\ntoken 0 (first 6 dims):")
    print(f"  reference : {np.array2string(reference[0, :6], precision=4, floatmode='fixed')}")
    print(f"  hls       : {np.array2string(hls_out[0, :6], precision=4, floatmode='fixed')}")


def main():
    parser = argparse.ArgumentParser(description="Convert a single HEPT attention block to HLS and compare outputs.")
    parser.add_argument("-m", "--model", type=str, default="hept")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible inputs/weights.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained HEPT attention-block state dict (see train_simple_hept.py) "
        "to load instead of random weights.",
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use a real tracking-600 event's features/coords instead of random synthetic "
        "data (requires --checkpoint; the event is truncated to the first SEQ_LEN nodes).",
    )
    parser.add_argument(
        "--event-idx",
        type=int,
        default=None,
        help="Dataset index of the tracking-600 event to use with --real-data. "
        "Default: first test-split event with >= SEQ_LEN nodes.",
    )
    parser.add_argument(
        "--full-checkpoint",
        type=str,
        default=None,
        help="Path to the full Transformer state dict (best_model.pt) providing feat_encoder "
        "weights for --real-data. Defaults to 'best_model.pt' next to --checkpoint.",
    )
    args = parser.parse_args()

    if args.real_data and not args.checkpoint:
        parser.error("--real-data requires --checkpoint (a trained HEPT attention-block state dict)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config_dir = Path(f"./HEPT/src/configs/tracking/tracking_trans_{args.model}.yaml")
    model_kwargs = yaml.safe_load(config_dir.open("r").read())["model_kwargs"]
    embed_dim = model_kwargs["h_dim"]
    num_heads_x_hashes = model_kwargs["num_heads"] * model_kwargs["n_hashes"]

    # Register the HEPT contrib kernel (layer class, parser, backend templates, headers).
    register()
    hls4ml.converters.register_pytorch_layer_handler("HEPT", parse_hept_attention_layer)

    model = PseudoHEPT(args.model, COORDS_DIM, **model_kwargs)
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.attn.load_state_dict(state_dict, strict=True)
    model.eval()  # disable dropout so the reference forward is deterministic

    # combined_shifts are the per-token region indices the kernel expects.
    if args.real_data:
        x, coords = load_real_event(args, model_kwargs)
    else:
        x = np.random.rand(1, SEQ_LEN, embed_dim).astype(np.float32)
        coords = np.random.rand(1, SEQ_LEN, COORDS_DIM).astype(np.float32)
    combined_shifts = compute_combined_shifts(
        torch.tensor(coords[0]), model.regions, model.block_size
    ).unsqueeze(0).numpy()  # (1, seq_len, num_heads * num_hashes)

    # PyTorch reference (attention sub-block only).
    reference = reference_attention(model, x[0], coords[0], combined_shifts[0])

    # Convert to HLS and run the C-simulation.
    hls_config = config_from_pytorch_model(
        model,
        input_shape=[
            (None, SEQ_LEN, embed_dim),
            (None, SEQ_LEN, COORDS_DIM),
            (None, SEQ_LEN, num_heads_x_hashes),
        ],
        channels_last_conversion="off",
        transpose_outputs=False,
    )
    hls_config["Model"]["par_factor"] = PAR_FACTOR

    hls_model = convert_from_pytorch_model(
        model,
        hls_config=hls_config,
        output_dir=OUTPUT_DIR,
        backend=BACKEND,
        io_type=IO_TYPE,
    )
    hls_model.compile()

    # hls4ml returns the output flattened per sample; reshape to (seq_len, embed_dim).
    hls_prediction = np.asarray(hls_model.predict([x, coords, combined_shifts])).reshape(reference.shape)

    report(reference, hls_prediction)


if __name__ == "__main__":
    main()
