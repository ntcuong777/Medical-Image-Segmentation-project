from .hardmseg.HarDMSEG import HarDMSEG
# from .mobile_w_net.mobile_w_net import MobileWnet
from .medical_transformer.medt import MedT

# Factory design pattern
class SegmenterFactory:
    """ 
    This is the factory of segmenter models
    The activation used is `hard_swish` by default
    """

    @staticmethod
    def create_segmenter_as(segmenter='HarDMSEG', img_size=512, imgchan=3, activation='relu'):
        assert(segmenter in ['HarDMSEG', 'MedT'])

        if segmenter == 'HarDMSEG':
            return SegmenterFactory.create_hardmseg_model(activation=activation)
        elif segmenter == 'MedT':
            return SegmenterFactory.create_medt_model(img_size=img_size, imgchan=imgchan)


    def create_hardmseg_model(activation='relu'):
        return HarDMSEG(activation=activation, channel=64)

    # def create_mobilewnet_model(activation='hard_swish'):
    #     return MobileWnet(activation=activation)
    
    def create_medt_model(img_size=512, imgchan=3):
        return MedT(img_size=img_size, imgchan=imgchan)