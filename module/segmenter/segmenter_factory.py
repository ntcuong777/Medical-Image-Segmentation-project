from .hardmseg.HarDMSEG import HarDMSEG
from .double_net import DoubleNet

# Factory design pattern
class SegmenterFactory:
    """ 
    This is the factory of segmenter models
    The activation used is `hard_swish` by default
    """

    @staticmethod
    def create_segmenter_as(segmenter='HarDMSEG', activation='relu', **kwargs):
        assert(segmenter in ['HarDMSEG', 'MedT'])

        if segmenter == 'HarDMSEG':
            return SegmenterFactory.create_hardmseg_model(activation=activation, **kwargs)
        elif segmenter == 'DoubleNet':
            return SegmenterFactory.create_double_net_model(**kwargs)


    def create_hardmseg_model(model_variant='HarDNet68', activation='relu', channel=32):
        return HarDMSEG(activation=activation, channel=channel)

    # def create_mobilewnet_model(activation='hard_swish'):
    #     return MobileWnet(activation=activation)
    
    def create_double_net_model(img_size=512, imgchan=3, hardnet_channel=32, pretrained_hardmseg=None):
        return DoubleNet(img_size=img_size, imgchan=imgchan, hardnet_channel=32, pretrained_hardmseg=pretrained_hardmseg)