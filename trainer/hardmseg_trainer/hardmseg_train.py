import torch
from torch.autograd import Variable
import os
from datetime import datetime
from data_utils.dataloader import get_train_loader, get_test_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from lib.losses import BCEIoULoss
from lib.HarDMSEG import HarDMSEG
from config import TrainConfig, TestConfig

def test(model):
    model.eval()

    config = TestConfig.load_config_class('config/test_config/test_config.yaml')

    test_loader = get_test_loader(config)
    b = 0.0
    count_imgs = 0
    for i, (image, gt) in enumerate(test_loader, start=1):
        image = image.cuda()
        gt = gt.data.numpy()

        res  = model(image, use_sigmoid=True)
        res = res.data.cpu().numpy()

        for j in range(gt.shape[0]):
            out = res[j]
            out = (out - out.min()) / (out.max() - out.min() + 1e-8)
            target = gt[j] * 255.0
            target /= (target.max() + 1e-8)
            smooth = 1

            intersection = (out*target)
            loss = (2 * intersection.sum() + smooth) / (out.sum() + target.sum() + smooth)

            count_imgs += 1
            a = float('{:.4f}'.format(loss))
            b = b + a

    return b / count_imgs


def train_loop(config: TrainConfig, train_loader, model, optimizer, epoch, best_dice, total_step):
    model.train()

    # ---- the loss to use ----
    loss_fn = BCEIoULoss()

    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record5 = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(config.input_dim*rate/32)*32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear')
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear')

            # ---- forward ----
            out = model(images)

            # ---- loss function ----
            loss = loss_fn(out, gts)

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, config.clip)
            optimizer.step()

            # ---- recording loss ----
            if rate == 1:
                loss_record5.update(loss.data, config.batch_size)

        # ---- training visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, config.epochs, i, total_step,
                          loss_record5.show()))


    save_path = config.save_model_path
    os.makedirs(save_path, exist_ok=True)

    # Test & save best every epoch
    best = best_dice
    meandice = test(model)
    print('{} Epoch [{:03d}/{:03d}], mDice = {:05f}'.format(datetime.now(), epoch, config.epochs, meandice))
    if meandice > best:
        best = meandice
        torch.save(model.state_dict(), os.path.join(save_path, config.model_name + '-best.pt'))
        print('[Saving Snapshot:]', os.path.join(save_path, config.model_name + '-best.pt'), 'New best:', meandice)
    else:
        print('Current best is:', best)

    # Save by epoch number
    if (epoch+1) % config.save_freq == 0:
        torch.save(model.state_dict(), os.path.join(save_path, config.model_name + '-{}.pt'.format(epoch)))

    return best # return best meandice for save the best later


def train_hardmseg(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HarDMSEG()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if config.resume:
        model.load_state_dict(torch.load(config.resume_model_path))
    model.to(device)

    train_loader = get_train_loader(config)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    params = model.parameters()
    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, config.learning_rate)
    else:
        optimizer = torch.optim.SGD(params, config.learning_rate, weight_decay = 1e-4, momentum = 0.9)

    print(optimizer)

    best_dice = 0.0
    for epoch in range(1, config.epochs):
        adjust_lr(optimizer, config.learning_rate, epoch, config.decay_rate, config.decay_epoch)
        best_dice = train_loop(config, train_loader, model, optimizer, epoch, best_dice, total_step)


if __name__ == "__main__":
    config = TrainConfig.load_config_class('config/train_config/train_config.yaml')
    train_hardmseg(config)