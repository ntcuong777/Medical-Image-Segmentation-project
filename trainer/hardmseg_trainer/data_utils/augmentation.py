import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from config import TrainConfig, TestConfig

class TrainAugmentation:
    def __init__(self, config: TrainConfig):
        if config.augmentation:
            self.augmentation = A.Compose([A.Resize(height=config.input_dim, width=config.input_dim, p=1),
                                           A.SomeOf([A.HorizontalFlip(),
                                                     A.VerticalFlip(),
                                                     A.RandomBrightnessContrast(),
                                                     A.ChannelDropout(),
                                                     A.CoarseDropout(min_holes=1, min_height=3, min_width=3),
                                                     A.GridDistortion(),
                                                     A.RandomRotate90()], 2),
                                           ToTensorV2(transpose_mask=True)], p=1)
        else:
            self.augmentation = A.Compose([A.Resize(height=config.input_dim, width=config.input_dim, p=1),
                                           ToTensorV2(transpose_mask=True)], p=1)

    def __call__(self, x: np.ndarray, mask: np.ndarray): 
        transform = self.augmentation(image=x, mask=mask)
        return transform['image'], transform['mask']


class TestAugmentation:
    def __init__(self, config: TestConfig):
        self.augmentation = A.Compose([A.Resize(height=config.input_dim, width=config.input_dim),
                                       ToTensorV2(transpose_mask=True)], p=1)
    
    def __call__(self, x: np.ndarray, mask: np.ndarray):
        transform = self.augmentation(image=x, mask=mask)
        return transform['image'], transform['mask']