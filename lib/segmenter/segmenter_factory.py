from .HarDMSEG import HarDMSEG
from .PraNet import PraNet
from .CANet import CANet
from .UACANet import UACANet

# Factory design pattern
class SegmenterFactory:
    """ 
    This is the factory of segmenter models
    The activation used is `hard_swish` by default
    """

    @staticmethod
    def create_segmenter_as(options):
        assert(options.Model.name in ['HarDMSEG', 'UACANet', 'PraNet', 'CANet'])

        segmenter = options.Model.name
        if segmenter == 'HarDMSEG':
            return SegmenterFactory.create_hardmseg_model(options.Model)
        elif segmenter == 'PraNet':
            return SegmenterFactory.create_pranet_model(options.Model)
        elif segmenter == 'UACANet':
            return SegmenterFactory.create_uacanet_model(options.Model)
        elif segmenter == 'CANet':
            return SegmenterFactory.create_canet_model(options.Model)

    def create_pranet_model(options):
        return PraNet(options)

    def create_canet_model(options):
        return CANet(options)

    def create_hardmseg_model(options):
        return HarDMSEG(options)

    def create_uacanet_model(options):
        return UACANet(options)