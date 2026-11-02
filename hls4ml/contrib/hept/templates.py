from __future__ import annotations

import math

from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate

from .layers import HEPTAttention


_HEPT_CONFIG_TEMPLATE = """struct config{index} : nnet::hept_config {{
    static const unsigned seq_len = {seq_len};
    static const unsigned embed_dim = {embed_dim};
    static const unsigned num_heads = {num_heads};
    static const unsigned dim_per_head = {dim_per_head};
    static const unsigned coords_dim = {coords_dim};
    static const unsigned num_hashes = {num_hashes};
    static const unsigned num_w_per_dist = {num_w_per_dist};
    static const unsigned block_size = {block_size};
    static const unsigned inter_reuse_factor = {inter_reuse_factor};
    static const unsigned qkv_reuse_factor = {qkv_reuse_factor};
    static const unsigned par_factor = {par_factor};
    static const unsigned par_factor_sort = {par_factor_sort};

    static const unsigned norm_inv_sqrt_table_size = {norm_inv_sqrt_table_size};
    static const unsigned norm_inv_sqrt_range = {norm_inv_sqrt_range};
    static const unsigned qkv_exp_table_size = {qkv_exp_table_size};
    static const unsigned qkv_exp_range = {qkv_exp_range};
    static const unsigned qkv_sqrt_table_size = {qkv_sqrt_table_size};
    static const unsigned qkv_sqrt_range = {qkv_sqrt_range};
    static const unsigned e2lsh_exp_table_size = {e2lsh_exp_table_size};
    static const unsigned e2lsh_exp_range = {e2lsh_exp_range};

    typedef {input_t} input_t;
    typedef {coords_t} coords_t;
    typedef {shift_t} shift_t;
    typedef {output_t} output_t;
    typedef {qkv_weight_t} qkv_weight_t;
    typedef {sqrt_w_t} sqrt_w_t;
    typedef {alpha_t} alpha_t;
    typedef {out_linear_weight_t} out_linear_weight_t;
    typedef {out_linear_bias_t} out_linear_bias_t;
    typedef {layer_norm_weight_t} layer_norm_weight_t;
    typedef {layer_norm_bias_t} layer_norm_bias_t;
    typedef {accum_t} accum_t;

    typedef {norm_accum_t} norm_accum_t;
    typedef {norm_sum_t} norm_sum_t;
    typedef {norm_sum_sqr_t} norm_sum_sqr_t;
    typedef {norm_mean_t} norm_mean_t;
    typedef {norm_var_table_t} norm_var_table_t;
    typedef {qkv_exp_table_t} qkv_exp_table_t;
    typedef {qkv_sqrt_table_t} qkv_sqrt_table_t;
    typedef {e2lsh_exp_table_t} e2lsh_exp_table_t;
    typedef {hash_mul_accum_t} hash_mul_accum_t;

    typedef {norm_out_t} norm_out_t;
    typedef {prepare_qk_t} prepare_qk_t;
    typedef {combined_shift_out_t} combined_shift_out_t;
    typedef {sorted_index_t} sorted_index_t;
    typedef {batched_index_select_out_t} batched_index_select_out_t;
    typedef {qkv_res_out_denom_t} qkv_res_out_denom_t;
    typedef {qkv_res_out_so_t} qkv_res_out_so_t;
    typedef {batched_index_select_unsort_out_o_t} batched_index_select_unsort_out_o_t;
    typedef {batched_index_select_unsort_out_logits_t} batched_index_select_unsort_out_logits_t;
    typedef {o_divide_logits_out_t} o_divide_logits_out_t;
}};
"""

_HEPT_FUNCTION_TEMPLATE = """nnet::hept_attention<{input_t}, {coords_t}, {shift_t}, {output_t}, {config}>(
    {input},
    {coords},
    {combined_shifts},
    {output},
    {layer_norm_weight},
    {layer_norm_bias},
    {q_weight},
    {k_weight},
    {v_weight},
    {sqrt_w},
    {alpha},
    {out_linear_weight},
    {out_linear_bias}
);"""


_DEFAULT_TYPE_STRINGS = {
    "input_t": "ap_fixed<16,6>",
    "coords_t": "ap_fixed<16,6>",
    # Explicit AP_RND_CONV: this is data_T_combined_shifts, the exact multiplicand in
    # lsh_mapping_combined_shift's tmp_q/tmp_k = col_buffer + hashed_shift*combined_shift
    # computation that directly produces the bucket-sort key -- Vitis's unspecified
    # default (AP_TRN, truncate toward -inf) biases this value low on every conversion
    # from the wider upstream float/accum_t source, same class of issue as accum_t above.
    "shift_t": "ap_fixed<16,6,AP_RND_CONV,AP_WRAP,0>",
    "output_t": "ap_fixed<16,6>",
    # Explicit AP_RND_CONV: Vitis's default (unspecified) rounding mode for ap_fixed is
    # AP_TRN (truncate toward -inf), which silently biases every narrowing accumulation
    # step in this type low. That bias cascades through the LSH hash chain (row_max/
    # row_min/hashed_shift/tmp_q/tmp_k all use accum_t) into bucket_sort_single's final
    # floor-based bucket assignment: a true hash value just above an integer boundary
    # (e.g. 13.03) can get truncated to just below it (e.g. 12.99) before the floor ever
    # runs, assigning the wrong bucket. Round-to-nearest removes this systematic bias
    # without changing the floor semantics the bucket assignment correctly relies on
    # (see hept.py's _clamp_floor_ste -- floor is the reference's actual behavior, this
    # only fixes what gets floored). accum_t is also the general wide accumulator for
    # layernorm/QKV computation elsewhere in this kernel; round-to-nearest is strictly
    # more accurate than truncation there too, so this is a uniform improvement, not a
    # narrow hash-chain-only patch.
    "accum_t": "ap_fixed<62,38,AP_RND_CONV,AP_WRAP,0>",
    "norm_accum_t": "ap_fixed<32,10>",
    "norm_sum_t": "ap_fixed<18,8,AP_RND_CONV,AP_WRAP,0>",
    "norm_sum_sqr_t": "ap_fixed<18,8,AP_RND_CONV,AP_WRAP,0>",
    "norm_mean_t": "ap_fixed<18,8,AP_RND_CONV,AP_WRAP,0>",
    "norm_var_table_t": "ap_ufixed<64,8,AP_RND_CONV,AP_SAT,0>",
    "qkv_exp_table_t": "ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0>",
    "qkv_sqrt_table_t": "ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0>",
    "e2lsh_exp_table_t": "ap_ufixed<64,20,AP_RND_CONV,AP_SAT,0>",
    "hash_mul_accum_t": "ap_fixed<32,16,AP_RND_CONV,AP_WRAP,0>",
    # Narrowed from the wide accum_t (col_buffer + hashed_shift*combined_shift_index)
    # immediately before bucket_sort_single's explicit [0,31] clamp -- this is the ONLY
    # use of this type (feeds q_hashed_combined/k_hashed_combined, consumed solely by
    # bucket_sort_single), so it must saturate rather than wrap: a wrapped out-of-range
    # value can land anywhere in-range and silently produce a wrong bucket instead of
    # being correctly clamped.
    "combined_shift_out_t": "ap_fixed<16,6,AP_RND_CONV,AP_SAT,0>",
}


def _type_name(obj, default: str) -> str:
    if obj is None:
        return default
    if hasattr(obj, "name"):
        return obj.name
    return str(obj)


def _var_name(obj) -> str:
    return getattr(obj, "name", str(obj))


class HEPTConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HEPTAttention)
        self.template = _HEPT_CONFIG_TEMPLATE

    def format(self, node):
        if node.model.config.get_config_value("IOType") != "io_stream":
            raise ValueError("HEPTAttention currently requires IOType='io_stream'.")

        params = self._default_config_params(node)
        # Each weight array gets its own type, read individually from hls4ml's own
        # per-weight-variable precision resolution (so a `LayerName.<this>.Precision.
        # <weight_var>` override in HLSConfig takes effect independently per array,
        # e.g. for HGQ-learned per-layer bit-widths). q/k/v share one type since they
        # are always trained/quantized together in practice; absent any override,
        # every weight defaults to the same model-wide Precision as before, so this
        # is behaviorally identical to the old shared `weight_t` when unconfigured.
        params["qkv_weight_t"] = node.get_weights("q_weight").type.name
        params["sqrt_w_t"] = node.get_weights("sqrt_w").type.name
        params["alpha_t"] = node.get_weights("alpha").type.name
        params["out_linear_weight_t"] = node.get_weights("out_linear_weight").type.name
        params["out_linear_bias_t"] = node.get_weights("out_linear_bias").type.name
        params["layer_norm_weight_t"] = node.get_weights("layer_norm_weight").type.name
        params["layer_norm_bias_t"] = node.get_weights("layer_norm_bias").type.name

        params["input_t"] = _type_name(node.get_attr("input_t", None), _DEFAULT_TYPE_STRINGS["input_t"])
        params["coords_t"] = _type_name(node.get_attr("coords_t", None), _DEFAULT_TYPE_STRINGS["coords_t"])
        params["shift_t"] = _type_name(node.get_attr("shift_t", None), _DEFAULT_TYPE_STRINGS["shift_t"])
        params["output_t"] = _type_name(node.get_attr("output_t", None), _DEFAULT_TYPE_STRINGS["output_t"])

        params["accum_t"] = _type_name(node.get_attr("accum_t", None), _DEFAULT_TYPE_STRINGS["accum_t"])
        params["norm_accum_t"] = _type_name(node.get_attr("norm_accum_t", None), _DEFAULT_TYPE_STRINGS["norm_accum_t"])
        params["norm_sum_t"] = _type_name(node.get_attr("norm_sum_t", None), _DEFAULT_TYPE_STRINGS["norm_sum_t"])
        params["norm_sum_sqr_t"] = _type_name(
            node.get_attr("norm_sum_sqr_t", None), _DEFAULT_TYPE_STRINGS["norm_sum_sqr_t"]
        )
        params["norm_mean_t"] = _type_name(node.get_attr("norm_mean_t", None), _DEFAULT_TYPE_STRINGS["norm_mean_t"])
        params["norm_var_table_t"] = _type_name(
            node.get_attr("norm_var_table_t", None), _DEFAULT_TYPE_STRINGS["norm_var_table_t"]
        )
        params["qkv_exp_table_t"] = _type_name(
            node.get_attr("qkv_exp_table_t", None), _DEFAULT_TYPE_STRINGS["qkv_exp_table_t"]
        )
        params["qkv_sqrt_table_t"] = _type_name(
            node.get_attr("qkv_sqrt_table_t", None), _DEFAULT_TYPE_STRINGS["qkv_sqrt_table_t"]
        )
        params["e2lsh_exp_table_t"] = _type_name(
            node.get_attr("e2lsh_exp_table_t", None), _DEFAULT_TYPE_STRINGS["e2lsh_exp_table_t"]
        )
        params["hash_mul_accum_t"] = _type_name(
            node.get_attr("hash_mul_accum_t", None), _DEFAULT_TYPE_STRINGS["hash_mul_accum_t"]
        )

        params["norm_out_t"] = _type_name(node.get_attr("norm_out_t", None), params["input_t"])
        params["prepare_qk_t"] = _type_name(node.get_attr("prepare_qk_t", None), params["input_t"])
        params["combined_shift_out_t"] = _type_name(
            node.get_attr("combined_shift_out_t", None), _DEFAULT_TYPE_STRINGS["combined_shift_out_t"]
        )

        seq_len = int(node.get_attr("seq_len"))
        index_width = max(1, math.ceil(math.log2(max(2, seq_len))))
        params["sorted_index_t"] = _type_name(node.get_attr("sorted_index_t", None), f"ap_uint<{index_width}>")

        params["batched_index_select_out_t"] = _type_name(
            node.get_attr("batched_index_select_out_t", None), params["input_t"]
        )
        params["qkv_res_out_denom_t"] = _type_name(node.get_attr("qkv_res_out_denom_t", None), params["input_t"])
        params["qkv_res_out_so_t"] = _type_name(node.get_attr("qkv_res_out_so_t", None), params["input_t"])
        params["batched_index_select_unsort_out_o_t"] = _type_name(
            node.get_attr("batched_index_select_unsort_out_o_t", None), params["input_t"]
        )
        params["batched_index_select_unsort_out_logits_t"] = _type_name(
            node.get_attr("batched_index_select_unsort_out_logits_t", None), params["input_t"]
        )
        params["o_divide_logits_out_t"] = _type_name(
            node.get_attr("o_divide_logits_out_t", None), params["output_t"]
        )

        params["norm_inv_sqrt_table_size"] = int(node.get_attr("norm_inv_sqrt_table_size", 1024))
        params["norm_inv_sqrt_range"] = int(node.get_attr("norm_inv_sqrt_range", 16))
        params["qkv_exp_table_size"] = int(node.get_attr("qkv_exp_table_size", 1024))
        params["qkv_exp_range"] = int(node.get_attr("qkv_exp_range", 8))
        params["qkv_sqrt_table_size"] = int(node.get_attr("qkv_sqrt_table_size", 1024))
        params["qkv_sqrt_range"] = int(node.get_attr("qkv_sqrt_range", 8))
        params["e2lsh_exp_table_size"] = int(node.get_attr("e2lsh_exp_table_size", 1024))
        params["e2lsh_exp_range"] = int(node.get_attr("e2lsh_exp_range", 8))

        return self.template.format(**params)


class HEPTFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HEPTAttention, include_header=["nnet_utils/nnet_hept.h"])
        self.template = _HEPT_FUNCTION_TEMPLATE

    def format(self, node):
        if node.model.config.get_config_value("IOType") != "io_stream":
            raise ValueError("HEPTAttention currently requires IOType='io_stream'.")

        params = self._default_function_params(node)

        coords_var = node.get_input_variable(node.inputs[1])
        shift_var = node.get_input_variable(node.inputs[2])

        params["coords_t"] = coords_var.type.name
        params["shift_t"] = shift_var.type.name
        params["coords"] = coords_var.name
        params["combined_shifts"] = shift_var.name

        for name in (
            "layer_norm_weight",
            "layer_norm_bias",
            "q_weight",
            "k_weight",
            "v_weight",
            "sqrt_w",
            "alpha",
            "out_linear_weight",
            "out_linear_bias",
        ):
            params[name] = _var_name(node.get_weights(name))

        return self.template.format(**params)
