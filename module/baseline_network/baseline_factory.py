from .hardnet.hardnet import HarDNet
from .hardnet_cbam.hardnet_cbam import HarDNet_CBAM

class BaselineFactory:
    """ 
    This is the factory of ImageNet baseline network models

    """

    @staticmethod
    def create_baseline_as(baseline_model='hardnet', model_variant='HarDNet68ds', use_attention=False, activation='relu'):
        assert(baseline_model in ['hardnet'])

        if baseline_model == 'hardnet':
            if model_variant == None:
                model_variant = 'HarDNet68ds'
            
            if not use_attention:
                return BaselineFactory.create_hardnet_model(model_variant=model_variant, activation=activation)
            else:
                return BaselineFactory.create_hardnet_cbam_model(model_variant=model_variant, activation=activation)


    def create_hardnet_model(model_variant, activation='relu'):
        return HarDNet(model_variant=model_variant, activation=activation)
    
    
    def create_hardnet_cbam_model(model_variant, activation='relu'):
        return HarDNet_CBAM(model_variant, activation=activation)