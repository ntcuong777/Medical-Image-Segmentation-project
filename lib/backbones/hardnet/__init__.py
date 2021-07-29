from .HarDNet import HarDNet
from .HarDNet_CBAM import HarDNet_CBAM

def get_hardnet_baseline(options):
    if not options.use_cbam:
        model = HarDNet(arch=options.arch, depth_wise=options.depth_wise)
        if options.pretrained:
            model_name = 'HarDNet' + str(options.arch) + ('ds' if options.depth_wise else '')
            model.load_pretrained(model_name=model_name)

        model.delete_classification_head() # delete classification head to reduce memory usage
        return model
    else:
        model = HarDNet_CBAM(arch=options.arch, depth_wise=options.depth_wise, pretrained_hardnet=options.pretrained)