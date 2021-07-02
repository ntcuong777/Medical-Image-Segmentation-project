from .hardnet.hardnet import HarDNet
from .hardnet_cbam.hardnet_cbam import HarDNet_CBAM

"""
I set default HarDNet variant as `HarDNet68ds` since I want to use
depthwise convolution only (for performance reason). The model is
already large enough, I want more speed.
"""
class BaselineFactory:
    """ 
    This is the factory of baseline network models
    The activation used is `hard_swish` by default
    """

    @staticmethod
    def create_baseline_as(baseline_model='hardnet', model_variant='HarDNet68ds', use_attention=False, activation='hard_swish'):
        assert(baseline_model in ['hardnet'])

        if baseline_model == 'hardnet':
            if model_variant == None:
                model_variant = 'HarDNet68ds'
            
            if not use_attention:
                return BaselineFactory.create_hardnet_model(model_variant=model_variant, activation=activation)
            else:
                return BaselineFactory.create_hardnet_cbam_model(model_variant=model_variant, activation=activation)


    def create_hardnet_model(model_variant, activation='hard_swish'):
        return HarDNet(model_variant=model_variant, activation=activation)
    
    def create_hardnet_cbam_model(model_variant, activation='hard_swish'):
        return HarDNet_CBAM(model_variant, activation=activation)