import os
import cv2 as cv
import torch.utils.data as data
import numpy as np
import random
from utils import print_warning
from .augmentation import TrainAugmentation, TestAugmentation
from configs import TrainConfig, TestConfig

class TrainDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, config: TrainConfig):
        self.im_size = config.input_dim
        self.augmentation = config.augmentation

        image_root = '{}/images/'.format(config.train_path)
        gt_root = '{}/masks/'.format(config.train_path)

        if config.augmentation:
            print_warning("NOTE: Applying augmentation on the fly!")

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

        self.data_transformer = TrainAugmentation(config)
            

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        image, gt = self.data_transformer(image, gt)

        return image, gt


    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = cv.imread(img_path)
            gt = cv.imread(gt_path)
            if img.shape == gt.shape:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts


    def rgb_loader(self, path):
        im_bgr = cv.imread(path)
        im_rgb = cv.cvtColor(im_bgr, cv.COLOR_BGR2RGB)
        return im_rgb


    def binary_loader(self, path):
        im_gt = cv.imread(path, cv.IMREAD_GRAYSCALE)
        return im_gt


    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.im_size[0] or w < self.im_size[1]:
            h = max(h, self.im_size[0])
            w = max(w, self.im_size[1])
            return cv.resize(img, (w, h), interpolation=cv.INTER_LINEAR), cv.resize(gt, (w, h), interpolation=cv.INTER_NEAREST)
        else:
            return img, gt


    def __len__(self):
        return self.size


def get_train_loader(config: TrainConfig, pin_memory=True):

    dataset = TrainDataset(config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=config.shuffle,
                                  num_workers=config.num_workers,
                                  pin_memory=pin_memory)
    return data_loader



class TestDataset:
    def __init__(self, config: TestConfig):
        self.im_size = config.input_dim

        image_root = '{}/images/'.format(config.test_path)
        gt_root = '{}/masks/'.format(config.test_path)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.data_transformer = TestAugmentation(config)
        self.size = len(self.images)


    # def load_data(self, index):
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image, gt = self.data_transformer(image, gt)

        # name = self.images[self.index].split('/')[-1]
        # if name.endswith('.jpg'):
        #     name = name.split('.jpg')[0] + '.png'
        # self.index += 1
        # return image, gt, name
        return image, gt


    def rgb_loader(self, path):
        im_bgr = cv.imread(path)
        im_rgb = cv.cvtColor(im_bgr, cv.COLOR_BGR2RGB)
        return im_rgb


    def binary_loader(self, path):
        im_gt = cv.imread(path, cv.IMREAD_GRAYSCALE)
        return im_gt


    def __len__(self):
        return self.size


def get_test_loader(config: TestConfig, pin_memory=True):
    dataset = TestDataset(config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=pin_memory)
    return data_loader