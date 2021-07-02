from .hardmseg.HarDMSEG import HarDMSEG
from .mobile_w_net.mobile_w_net import MobileWnet

class SegmenterFactory:
    """ 
    This is the factory of segmenter models
    The activation used is `mish` by default
    """

    @staticmethod
    def create_segmenter_as(segmenter='HarDMSEG', use_attention=False, activation='hard_swish'):
        assert(segmenter in ['HarDMSEG', 'MobileWnet'])

        if segmenter == 'HarDMSEG':
            return SegmenterFactory.create_hardmseg_model(use_attention=use_attention, activation=activation)
        elif segmenter == 'MobileWnet':
            return SegmenterFactory.create_mobilewnet_model(activation=activation)


    def create_hardmseg_model(use_attention=False, activation='hard_swish'):
        return HarDMSEG(activation=activation, use_attention=use_attention)

    def create_mobilewnet_model(activation='hard_swish'):
        return MobileWnet(activation=activation)