from .hardmseg.HarDMSEG import HarDMSEG
from .double_net import DoubleNet
from .uaca_net import UACANet
from easydict import EasyDict as ed
import yaml

# Factory design pattern
class SegmenterFactory:
    """ 
    This is the factory of segmenter models
    The activation used is `hard_swish` by default
    """

    @staticmethod
    def create_segmenter_as(segmenter='HarDMSEG', **kwargs):
        assert(segmenter in ['HarDMSEG', 'UACANet', 'DoubleNet'])

        if segmenter == 'HarDMSEG':
            return SegmenterFactory.create_hardmseg_model()
        elif segmenter == 'DoubleNet':
            return SegmenterFactory.create_double_net_model(**kwargs)
        elif segmenter == 'UACANet':
            return SegmenterFactory.create_uacanet_model(**kwargs)


    def create_hardmseg_model():
        return HarDMSEG()


    def create_double_net_model(pretrained_hardmseg='snapshots/HarDMSEG/best.pth'):
        return DoubleNet(pretrained_hardmseg=pretrained_hardmseg)


    def create_uacanet_model(uaca_model='UACANet-L'):
        # Load config based on model name
        uaca_opt = ed(yaml.load(open('config/uaca_config/' + uaca_model + '.yaml'), yaml.FullLoader))
        return UACANet(uaca_opt.Model)