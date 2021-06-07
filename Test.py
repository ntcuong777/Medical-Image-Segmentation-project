import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.HarDMSEG import HarDMSEG
from utils.dataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='HarD-MSEG-best.pth')
#for _data_name in ['CVC-ClinicDB']:
for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    
    ##### put ur data_path here #####
    data_path = '/work/james128333/PraNet/TestDataset/{}/'.format(_data_name)
    #####                       #####
    
    save_path = './results/HarDMSEG/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = HarDMSEG()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        misc.imsave(save_path+name, res)
