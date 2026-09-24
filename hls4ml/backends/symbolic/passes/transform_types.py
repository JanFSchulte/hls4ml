from hls4ml.backends.catapult.passes.transform_types import TransformTypes as CatapultTransformTypes
from hls4ml.backends.vivado.passes.transform_types import TransformTypes as VivadoTransformTypes
from hls4ml.model.optimizer import GlobalOptimizerPass


class TransformTypes(GlobalOptimizerPass):
    """Converts the types to ap_* (Vivado/Vitis HLS) or ac_* (Catapult HLS) types, depending on the chosen compiler"""

    def __init__(self):
        self.vivado_transform = VivadoTransformTypes()
        self.catapult_transform = CatapultTransformTypes()

    def transform(self, model, node):
        if model.config.get_config_value('Compiler') == 'catapult':
            return self.catapult_transform.transform(model, node)
        else:
            return self.vivado_transform.transform(model, node)
