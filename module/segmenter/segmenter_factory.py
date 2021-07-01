from .hardmseg.HarDMSEG import HarDMSEG
from .dcunet.DCUnet import DcUnet
from .double_unet.double_unet import DoubleUnet

class SegmenterFactory:
    """ 
    This is the factory of segmenter models
    The activation used is `mish` by default
    """

    @staticmethod
    def create_segmenter_as(segmenter='HarDMSEG', use_attention=False, activation='mish'):
        assert(segmenter in ['HarDMSEG', 'Double-Unet', 'DCUnet'])

        if segmenter == 'HarDMSEG':
            return SegmenterFactory.create_hardmseg_model(use_attention=use_attention, activation=activation)
        elif segmenter == 'DCUnet':
            return SegmenterFactory.create_dcunet_model(activation=activation)
        elif segmenter == 'Double-Unet':
            return SegmenterFactory.create_double_unet_model()


    def create_hardmseg_model(use_attention=False, activation='mish'):
        return HarDMSEG(activation=activation, use_attention=use_attention)

    def create_dcunet_model(activation='mish', double_unet_style=True):
        if not double_unet_style:
            return DcUnet(input_channels=3)
        else:
            return DcUnet(input_channels=4)

    def create_double_unet_model():
        return DoubleUnet()