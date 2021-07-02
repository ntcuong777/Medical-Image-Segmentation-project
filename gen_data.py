import argparse
from utils.dataloader import get_loader
from PIL import Image
import numpy as np
import torch

augmentation = True
batch_size = 100
trainsize = 512
image_root = 'data/train/images/'
gt_root = 'data/train/masks/'

train_loader = get_loader(image_root, gt_root, batchsize=batch_size, trainsize=trainsize, augmentation=augmentation)
total_step = len(train_loader)

img_count = 0
for c in range(10):
    for i, pack in enumerate(train_loader, start=1):
        # ---- data prepare ----
        images, gts = pack
        images = images.permute(0, 2, 3, 1).numpy()
        gts = gts.permute(0, 2, 3, 1).squeeze().numpy()

        images = (images * 255.0).astype(np.uint8)
        gts = (gts * 255.0).astype(np.uint8)
        for j in range(100):
            Image.fromarray(images[j, ...]).save('augmented_data/train/images/%06d.png' % img_count)
            Image.fromarray(gts[j, ...]).save('augmented_data/train/masks/%06d.png' % img_count)
            img_count += 1