from .losses import *

def get_loss_fn(loss_fn_name='bce_iou_loss'):
    return eval(loss_fn_name)