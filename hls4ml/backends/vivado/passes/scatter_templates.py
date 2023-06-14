from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import ScatterAdd1D, ScatterAdd2D, ScatterAdd3D

# scatter_add config templates

scatteradd1d_config_template = """struct config{index} : nnet::scatter_add_config_1d {{
    static const unsigned in_x = {in_x};
    static const unsigned index_x = {index_x};
    static const unsigned src_x = {src_x};
    static const unsigned dim = {dim};
}};\n"""

scatteradd2d_config_template = """struct config{index} : nnet::scatter_add_config_2d {{
    static const unsigned in_x = {in_x};
    static const unsigned in_y = {in_y};
    static const unsigned index_x = {index_x};
    static const unsigned index_y = {index_y};
    static const unsigned src_x = {src_x};
    static const unsigned src_y = {src_y};
    static const unsigned dim = {dim};
}};\n"""


scatteradd3d_config_template = """struct config{index} : nnet::scatter_add_config_3d {{
    static const unsigned in_x = {in_x};
    static const unsigned in_y = {in_y};
    static const unsigned in_z = {in_z};
    static const unsigned index_x = {index_x};
    static const unsigned index_y = {index_y};
    static const unsigned index_z = {index_z};
    static const unsigned src_x = {src_x};
    static const unsigned src_y = {src_y};
    static const unsigned src_z = {src_z};
    static const unsigned dim = {dim};
}};\n"""


scatteraddd1d_function_template = (
    'nnet::scatter_add_1d<{input1_t}, {output_t}, {input2_t}, {input3_t}, {config}>({input1}, {output}, {input2}, {input3});'
)
scatteraddd2d_function_template = (
    'nnet::scatter_add_2d<{input1_t}, {output_t}, {input2_t}, {input3_t}, {config}>({input1}, {output}, {input2}, {input3});'
)
scatteraddd2dt_function_template = 'nnet::scatter_add_2d_t<{input1_t}, {output_t}, {input2_t}, {input3_t}, {config}>({input1}, {output}, {input2}, {input3});'  # noqa: E501
scatteraddd3d_function_template = (
    'nnet::scatter_add_3d<{input1_t}, {output_t}, {input2_t}, {input3_t}, {config}>({input1}, {output}, {input2}, {input3});'
)
scatteraddd3dt_function_template = 'nnet::scatter_add_3d_t<{input1_t}, {output_t}, {input2_t}, {input3_t}, {config}>({input1}, {output}, {input2}, {input3});'  # noqa: E501


class ScatterAddConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((ScatterAdd1D, ScatterAdd2D, ScatterAdd3D))
        self.templates = {
            'ScatterAdd1D': scatteradd1d_config_template,
            'ScatterAdd2D': scatteradd2d_config_template,
            'ScatterAdd3D': scatteradd3d_config_template,
        }

    def format(self, node):
        params = self._default_config_params(node)
        return self.templates[node.class_name].format(**params)


scatter_add_include_list = ['nnet_utils/nnet_scatteradd.h']


class ScatterAddFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((ScatterAdd1D, ScatterAdd2D, ScatterAdd3D), include_header=scatter_add_include_list)
        self.templates = {
            'ScatterAdd1D': scatteraddd1d_function_template,
            'ScatterAdd2D': scatteraddd2d_function_template,
            'ScatterAdd2DTransposed': scatteraddd2dt_function_template,
            'ScatterAdd3D': scatteraddd3d_function_template,
            'ScatterAdd3DTransposed': scatteraddd3dt_function_template,
        }

    def format(self, node):
        params = self._default_function_params(node)
        params['input1_t'] = node.get_input_variable(node.inputs[0]).type.name
        params['input2_t'] = node.get_input_variable(node.inputs[1]).type.name
        params['input3_t'] = node.get_input_variable(node.inputs[2]).type.name
        params['input1'] = node.get_input_variable(node.inputs[0]).name
        params['input2'] = node.get_input_variable(node.inputs[1]).name
        params['input3'] = node.get_input_variable(node.inputs[2]).name
        if hasattr(node, 'channels_last_converted'):
            return self.templates[node.class_name + "Transposed"].format(**params)
        else:
            return self.templates[node.class_name].format(**params)
