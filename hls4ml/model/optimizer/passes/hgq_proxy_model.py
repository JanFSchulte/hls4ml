import re
import typing
from copy import copy
from warnings import warn

import numpy as np

from hls4ml.model.attributes import Attribute, TypeAttribute, WeightAttribute
from hls4ml.model.layers import Activation, Layer, Reshape, Transpose, register_layer
from hls4ml.model.optimizer import OptimizerPass, register_pass
from hls4ml.model.types import FixedPrecisionType, UnspecifiedPrecisionType

if typing.TYPE_CHECKING:
    from hls4ml.model import ModelGraph

re_purge_prefix = re.compile(r'(?<!\w)(?:ap_|ac_)', re.IGNORECASE)
re_parse_fixed = re.compile(r'\s*(u?)fixed<([^>]+)>\s*', re.IGNORECASE)


class FixedPointQuantizer(Layer):
    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        self.add_output_variable(shape)
        self.set_attr('n_in', self.get_input_variable().size())
        self.overrides = self.attributes['overrides']
        self.fusible = self.attributes['fusible']
        self.SAT, self.RND = self.attributes['SAT'], self.attributes['RND']
        self.mask_kbi = self.attributes['mask_kbi']


class UnaryLUT(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        TypeAttribute('table_t', default=FixedPrecisionType(18, 8, True)),
        WeightAttribute('table'),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        self.add_output_variable(shape)
        self.set_attr('n_in', inp.size())
        self.table = self.attributes['table_data']
        self.attributes['table_size'] = len(self.table)

        self.add_weights_variable(name='table')


class MaterializeSoftmaxTables(OptimizerPass):
    """Turn HGQ's trained softmax exp/reciprocal lookup tables into weight arrays.

    Without this, the generated C++ rebuilds both tables at runtime from ``std::exp`` and
    ``1/x`` (``nnet::init_exp_table`` / ``nnet::init_invert_table``), which is not
    guaranteed to reproduce HGQ's quantized table contents.

    ``QSoftmaxHandler`` stashes ``_exp_table_fn`` / ``_inv_table_fn`` on the layer; both
    take the ``(k, i, f)`` of the type the C++ uses to address the table. That address type
    is only known here: for the latency implementation it is the softmax input precision,
    which the ``bit_exact`` pass decides. This pass must therefore run after ``bit_exact``
    and before ``<backend>:transform_types``.
    """

    _supported_backends = ('vivado', 'vitis')

    def match(self, node: Layer):
        from hls4ml.model.layers import Softmax

        # Keyed on the stash rather than on the layer type: non-HGQ softmax layers never
        # carry it, and every layer that does must have it removed (it is not
        # JSON-serializable, so ModelGraph.save() would choke on a leftover).
        return isinstance(node, Softmax) and '_exp_table_fn' in node.attributes

    def transform(self, model: 'ModelGraph', node: Layer):
        exp_table_fn = node.attributes['_exp_table_fn']
        inv_table_fn = node.attributes['_inv_table_fn']
        del node.attributes['_exp_table_fn']
        del node.attributes['_inv_table_fn']

        impl = node.get_attr('implementation')
        if impl not in ('latency', 'stable'):
            return False  # argmax / legacy use no lookup table

        backend = str(model.config.config.get('Backend', '')).lower()
        if backend not in self._supported_backends:
            warn(
                f'Backend {backend} does not support pre-computed softmax tables; layer {node.name} will fall back to '
                'tables generated at runtime by the HLS code and may not be bit-exact with Keras.',
                stacklevel=1,
            )
            return False

        # The address type of the exp table. For the stable implementation the table is
        # indexed by the normalized input (x_max - x), for the latency one by the layer
        # input itself. Note this reads the *element* precision, which is only unambiguous
        # because this pass runs before transform_types wraps io_stream variables.
        if impl == 'stable':
            exp_inp_t: FixedPrecisionType = node.attributes['inp_norm_t'].precision
        else:
            exp_inp_t = node.get_input_variable().type.precision
        inv_inp_t: FixedPrecisionType = node.attributes['inv_inp_t'].precision

        tables = {}
        for name, addr_t in (('exp_table', exp_inp_t), ('inv_table', inv_inp_t)):
            # nnet::softmax_idx_from_real_val slices the top ceillog2(table_size) bits of
            # the address word. Sizing the table as 2**width makes that slice the whole
            # word, so the index is a plain bit reinterpretation: no truncation of the low
            # bits, and no negative bit range when the table is larger than the word.
            size = 1 << int(addr_t.width)
            if int(node.get_attr(f'{name}_size') or 0) != size:
                warn(
                    f'{node.name}: {name}_size {node.get_attr(f"{name}_size")} does not match the {addr_t.width}-bit '
                    f'address type {addr_t}; overriding to {size} to keep table indexing bit-exact.',
                    stacklevel=1,
                )
            node.set_attr(f'{name}_size', size)
            k = int(bool(addr_t.signed))
            fn = exp_table_fn if name == 'exp_table' else inv_table_fn
            tables[name] = fn((k, addr_t.integer - k, int(addr_t.width) - addr_t.integer))

        for name, data in tables.items():
            # Capture the type bit_exact derived before creating the variable: storing a
            # WeightVariable under `name` makes AttributeDict overwrite `name_t` with the
            # variable's own type.
            named_t = node.attributes[f'{name}_t']
            node.set_attr(f'{name}_data', data)
            node.add_weights_variable(
                name=name, var_name=name + '{index}', precision=named_t.precision, type_name=named_t.name
            )
            weight_var = node.get_weights(name)
            weight_var.type = named_t
            node.attributes[f'{name}_t'] = named_t
            model.config.layer_name_precision[f'{node.name}_{name}'] = str(named_t.precision)

        return False


def userconf_ifdef(key: str, layer_name: str, model):
    hls_config: dict = model.config.config['HLSConfig']
    layer_confs: dict = hls_config.get('LayerName', None)
    if not layer_confs:
        return False
    layer_conf = layer_confs.get(layer_name, None)
    if not layer_conf:
        return False
    # return key in layer_conf # Ideal case. Not for now.
    if key.endswith('_t') and key != 'table_t':
        # table_t cannot be defined in Precision, for some reason.
        # On the other hand, result_t, weight_t, bias_t, accum_t cannot be decleared explicitly outside Precision, for now.
        # However, still assume that they can be defined explicitly outside Precision.
        precision_conf = layer_conf.get('Precision', None)
        if not precision_conf:
            return key in layer_conf
        return key[:-2] in precision_conf or key in layer_conf

    if key == 'parallelization_factor':
        # Irregular config key name.
        return 'ParallelizationFactor' in layer_conf

    return key in layer_conf


q_kifRS_t = tuple[np.ndarray, np.ndarray, np.ndarray, str, str]


class FuseFixedPointQuantizer(OptimizerPass):
    def match(self, node: Layer):
        if not node.attributes.get('bit_exact_transformed', False):
            return False

        if isinstance(node, FixedPointQuantizer):
            return all(np.unique(x).size == 1 for x in node.mask_kbi)

        if isinstance(node, Activation):
            return node.get_attr('activation') == 'linear' and node.get_attr('trusted', False)

        return False

    def propagate(self, node: Layer, precision: FixedPrecisionType):
        from hls4ml.model.optimizer.passes.bit_exact import get_input_layers, get_output_layers

        if node.attributes.get('_result_t_propagated', False):
            msg = f'result_t for {node.name} propagated multiple times. \
                Bit-exactness may be compromised. Avoid using consecutive quantizers in your model.'
            warn(msg, stacklevel=1)

        node.get_output_variable().type.precision = precision
        node.attributes['result_t'].precision = precision
        node.attributes['_result_t_propagated'] = True

        if not isinstance(node, (Reshape, Transpose)):
            return node

        inp_layer = get_input_layers(node)[0]
        can_propagate = len(get_output_layers(inp_layer)) == 1

        if not can_propagate:
            return node

        new_precision = copy(precision)
        precision.saturation_bits = 0
        precision.rounding_mode = 'TRN'
        precision.saturation_mode = 'WRAP'
        self.propagate(inp_layer, new_precision)

    def transform(self, model: 'ModelGraph', node: FixedPointQuantizer):
        from hls4ml.model.optimizer.passes.bit_exact import get_input_layers, get_output_layers

        if isinstance(node, FixedPointQuantizer):
            # Rounding and saturation for FixedPointQuantizer are applied in generated code, thus not reflected in result_t.
            if node.RND == 'TRN' and node.SAT == 'WRAP':
                precision: FixedPrecisionType = copy(node.get_output_variable().type.precision)
            else:
                k, b, i = node.mask_kbi
                k, b, i = bool(k.ravel()[0]), max(int(b.ravel()[0]), 1), int(i.ravel()[0])
                precision = FixedPrecisionType(b, i, k, node.RND, node.SAT)
        else:
            precision = copy(node.get_output_variable().type.precision)

        inp_layer = get_input_layers(node)[0]
        can_fuse = len(get_output_layers(inp_layer)) == 1
        attributes = copy(node.attributes)
        attributes['activation'] = 'linear'
        attributes['table_size'] = -1
        attributes['table_t'] = FixedPrecisionType(1, 0)
        attributes['result_t'] = precision
        if not can_fuse:
            linear = Activation(model, node.name, attributes, node.inputs, node.outputs)
            linear.get_output_variable().type.precision = precision
            model.replace_node(node, linear)
        else:
            self.propagate(inp_layer, precision)
            model.remove_node(node)
        return True


class EnforceProxyModelEmbeddedConfig(OptimizerPass):
    def match(self, node: Layer):
        if not isinstance(node, FixedPointQuantizer):
            return False
        if not node.overrides:
            return False
        return True

    def transform(self, model, node: FixedPointQuantizer):
        if 'layers' not in node.overrides:
            return False

        graph_changed = False
        layers = node.overrides['layers']
        for name, conf in layers.items():
            conf: dict[str, str]
            name: str
            if name not in model.graph:
                # Some layer may be removed by other passes. (e.g. Final flatten layer)
                continue
            target_node: Layer = model.graph[name]

            # Invoke automatic precision derivation for pooling layers accum_t, if undefined.
            if 'pool' in target_node.__class__.__name__.lower():
                if not userconf_ifdef('accum_t', name, model):
                    target_node.attributes['accum_t'].precision = UnspecifiedPrecisionType()

            for k, v in conf.items():
                if userconf_ifdef(k, name, model):
                    warn(
                        f'Config key {k} is defined in hls_config for layer {name} by user. Proxy model config is ignored.',
                        stacklevel=1,
                    )
                    continue

                if k.endswith('_t'):
                    continue  # Handled in bit-exact flow.
                elif k in target_node.attributes:
                    target_node.set_attr(k, v)
                elif k == 'parallelization_factor':
                    target_node.set_attr(k, int(v))

            if linear_node := model.graph.get(f'{name}_linear'):
                # Proxy model does not assume any extra linear layer.
                # Purge them on sight
                model.remove_node(linear_node)
                graph_changed = True

        return graph_changed


def register_hgq_proxy_model():
    register_layer('FixedPointQuantizer', FixedPointQuantizer)
    register_layer('HGQ>FixedPointQuantizer', FixedPointQuantizer)
    register_layer('UnaryLUT', UnaryLUT)
    register_layer('HGQ>UnaryLUT', UnaryLUT)
    register_pass('enforce_proxy_model_embedded_config', EnforceProxyModelEmbeddedConfig)
    register_pass('fuse_fixed_point_quantizer', FuseFixedPointQuantizer)
    register_pass('materialize_softmax_tables', MaterializeSoftmaxTables)
