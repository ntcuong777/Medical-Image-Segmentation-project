from typing import Counter

from numpy.lib.function_base import average
import torch
import yaml
import os
import argparse
import tqdm
import sys
import time
import torch.nn.functional as F
import numpy as np

from PIL import Image
from easydict import EasyDict as ed

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib.segmenter import SegmenterFactory
from utils.dataloader import *
from utils.utils import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/xnet.yaml')
    return parser.parse_args()

def speedtest(opt):
    device = 'cuda:' + str(opt.Model.cuda_device) if torch.cuda.is_available() else 'cpu'
    model = SegmenterFactory.create_segmenter_as(opt) # eval(opt.Model.name)(opt.Model)
    model.load_state_dict(torch.load(opt.Test.pth_path))
    model.to(device) # model.cuda()
    model.eval()

    # Init runtime on GPU
    init_tensor = torch.tensor(np.ones(shape=(32, 3, 352, 352)).astype(np.float32)).to(device)
    count_init = 5
    while count_init > 0:
        model(init_tensor)
        init_tensor += 1
        count_init -= 1

    print('#' * 20, 'Init done, starting speedtest', '#' * 20)

    average_fps = 0.0
    count_samples = 20

    while count_samples > 0:
        count_samples -= 1

        dataset = 'Kvasir'
        data_path = os.path.join(opt.Test.gt_path, dataset)
        save_path = os.path.join(opt.Test.out_path, dataset)

        os.makedirs(save_path, exist_ok=True)
        image_root = os.path.join(data_path, 'images')
        gt_root = os.path.join(data_path, 'masks')
        test_dataset = PolypDataset(image_root, gt_root, opt.Test)
        test_loader = data.DataLoader(dataset=test_dataset,
                                        batch_size=opt.Test.batch_size,
                                        num_workers=opt.Test.num_workers,
                                        pin_memory=opt.Test.pin_memory)

        total_time = 0.0
        count_imgs = 0
        for i, sample in enumerate(test_loader):
            image = sample['image']

            image = image.to(device)
            # Only test inference speed of the model
            start_time = time.time()
            out = model(image)['pred']
            out = out.data.sigmoid().cpu().numpy() # Convert to numpy for fairness
            total_time += (time.time() - start_time)

            count_imgs += out.shape[0]

        average_fps += (1.0 / (total_time / count_imgs))
    
    print('AVERAGE FPS = %.5f' % (average_fps / 20.0))

    print('#' * 20, 'Speedtest done', '#' * 20)

if __name__ == "__main__":
    args = _args()
    opt = ed(yaml.load(open(args.config), yaml.FullLoader))
    speedtest(opt)