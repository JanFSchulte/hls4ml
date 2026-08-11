import typing
from collections.abc import Callable, Sequence

import numpy as np
from quantizers import get_fixed_quantizer_np

from hls4ml.model.types import FixedPrecisionType

from ._base import KerasV3LayerHandler, QLayerHandler

if typing.TYPE_CHECKING:
    import hgq
    from hgq.quantizer import Quantizer
    from hgq.quantizer.internal import FixedPointQuantizerBase
    from keras import KerasTensor

from decimal import Decimal


def fixed_grid_by_bit_pattern(k: int, i: int, f: int) -> np.ndarray:
    """All values representable by a fixed-point type, ordered such that the array index
    equals the two's-complement bit pattern of the corresponding word.

    This is the addressing convention used by ``nnet::get_index_unary_lut`` and
    ``nnet::softmax_real_val_from_idx`` in the generated C++.

    Args:
        k: 1 if the type is signed, 0 otherwise.
        i: Number of integer bits, excluding the sign bit.
        f: Number of fractional bits.

    Returns:
        np.ndarray: The table domain, of length ``2 ** (k + i + f)``.
    """
    K, I, F = Decimal(int(k)), Decimal(int(i)), Decimal(int(f))  # noqa: E741
    _eps = Decimal(2) ** -F
    _min = -K * Decimal(2) ** I
    _max = Decimal(2) ** I - _eps
    N = (_max - _min) / _eps + 1
    assert float(N).is_integer(), 'Invalid quantizer range'
    N = int(N)
    assert N <= 2**20, f'Too large quantizer range: {N} entries for kif={(k, i, f)}'
    assert N & (N - 1) == 0, f'Invalid quantizer range: N must be power of 2, got {N}'

    grid = np.linspace(float(_min), float(_max), N, dtype=np.float32)
    if k:
        # idx by binary repr, move the positive part to the front
        grid = np.concatenate([grid[N // 2 :], grid[: N // 2]])
    return grid


def kif_of(q: 'FixedPointQuantizerBase') -> tuple[int, int, int]:
    """Get the (k, i, f) of a homogeneous hgq fixed-point quantizer.

    Entries that were trained to represent nothing at all (``k + i + f <= 0``) are mapped
    to a sentinel that makes them contribute no range.
    """
    from keras import ops

    k, i, f = q.kif
    mask = k + i + f > 0
    i, f = np.where(mask, i, -32), np.where(mask, f, -32)  # type: ignore
    return int(ops.max(k)), int(ops.max(i)), int(ops.max(f))  # type: ignore


def extract_lut_table(activation: Callable, oq: 'Quantizer|None', kif: tuple[int, int, int]) -> np.ndarray:
    """Materialize a lookup table by evaluating ``activation`` over every value a
    fixed-point type can hold, then quantizing the result with ``oq``.

    ``kif`` describes the *address* type, i.e. the type the generated C++ uses to index the
    table, and is passed explicitly rather than read off a layer's input quantizer: a
    ``QSoftmax`` with ``stable=False`` has ``exp_table.enable_iq == False``, and for the
    latency softmax implementation the domain is the softmax input precision, which is only
    final after the ``bit_exact`` pass.

    Args:
        activation: Elementwise function to tabulate.
        oq: hgq output quantizer applied to the table values, or None to leave them alone.
        kif: (k, i, f) of the address type.

    Returns:
        np.ndarray: The table, indexed by the two's-complement bit pattern of the address.
    """
    from hgq.quantizer.internal import FixedPointQuantizerBase
    from keras import ops

    grid = fixed_grid_by_bit_pattern(*kif)
    table = activation(grid)

    if oq is not None:
        internal_q = oq.quantizer
        if not isinstance(internal_q, FixedPointQuantizerBase):
            raise NotImplementedError('FloatPointQuantizer is not supported yet')

        # Quantize numerically rather than by calling the hgq Quantizer layer: the layer
        # broadcasts against the shape it was built for, which is not the rank-1 grid here
        # (QSoftmax's sublayers are built for the softmax input/reduced shapes). The
        # quantizer is homogeneous, so this is the same operation.
        round_mode = internal_q.round_mode
        if round_mode.startswith('S_'):
            round_mode = round_mode[2:]  # stochastic rounding
        fixed_q = get_fixed_quantizer_np(round_mode, internal_q.overflow_mode)
        k, i, f = (ops.convert_to_numpy(x).ravel().item() for x in internal_q.kif)
        table = fixed_q(table, k, i, f)  # type: ignore

    return np.asarray(ops.convert_to_numpy(table))


class QUnaryLUTHandler(QLayerHandler, KerasV3LayerHandler):
    handles = ('hgq.layers.activation.QUnaryFunctionLUT',)

    def handle(
        self,
        layer: 'hgq.layers.QUnaryFunctionLUT',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        from hgq.quantizer.internal import FixedPointQuantizerBase
        from keras import ops

        if not layer.enable_iq and not layer.enable_oq:
            raise ValueError('Currently only support input_quantizer enabled UnaryFunctionLUT layer')
        assert not layer._allow_heterogeneous_table, 'Heterogeneous table is not supported in QUnaryFunctionLUT layer'

        iq = layer.iq.quantizer
        if not isinstance(iq, FixedPointQuantizerBase):
            raise NotImplementedError('FloatPointQuantizer is not supported yet')

        table = extract_lut_table(layer.activation, layer.oq if layer.enable_oq else None, kif_of(iq))

        oq = layer.oq.quantizer
        if not isinstance(oq, FixedPointQuantizerBase):
            raise NotImplementedError('FloatPointQuantizer is not supported yet')
        k, i, f = (ops.convert_to_numpy(x).ravel().item() for x in oq.kif)
        k, b, I = bool(k), k + i + f, k + i  # noqa: E741
        table_t = FixedPrecisionType(b, I, k)

        config = {}
        config.update(self.default_config)
        config.update(
            {
                'class_name': 'UnaryLUT',
                'table_data': table,
                'table_t': table_t,
                'activation': 'unary_lut',
            }
        )

        return (config,)
