from numpy.core.fromnumeric import mean
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from module.segmenter.HarDMSEG import HarDMSEG
from utils.dataloader import get_loader,test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
from torchinfo import summary
import torch.nn as nn
from module.losses.dice_focal_loss import DiceFocalLoss

def structure_loss(pred, mask):
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()



def test(model, path):
    
    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    #####                                             #####
    
    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, 352)
    b=0.0
    for i in range(100):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        
        res  = model(image)
        res = F.interpolate(res, size=gt.shape, mode='bilinear')
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        input = res
        target = np.array(gt)
        smooth = 1
        input_flat = np.reshape(input,(-1))
        target_flat = np.reshape(target,(-1))
 
        intersection = (input_flat*target_flat)
        
        loss =  (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        a =  '{:.4f}'.format(loss)
        a = float(a)
        b = b + a
        
    return b/100



def train(train_loader, model, optimizer, scheduler, epoch, test_path, best_dice):
    model.train()

    # ---- the loss to use ----
    #loss_fn = structure_loss
    loss_fn = DiceFocalLoss()

    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    # loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    loss_record5 = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear')
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear')
            # ---- forward ----
            #lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            lateral_map_5 = model(images)
            # ---- loss function ----
            loss5 = loss_fn(lateral_map_5, gts)
            
            
            #loss = loss2 + 0.4*loss3 + 0.4*loss4 + 0.2*loss5    # TODO: try different weights for loss
            loss = loss5
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            scheduler.step() # step scheduler
            # ---- recording loss ----
            if rate == 1:
                loss_record5.update(loss5.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                          loss_record5.show()))
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    
    best = best_dice
    if (epoch+1) % 1 == 0:
        meandice = test(model,test_path)
        print('{} Epoch [{:03d}/{:03d}], mDice = {:05f}'.format(datetime.now(), epoch, opt.epoch, meandice))
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path + 'Moded-HarD-MSEG-best.pth' )
            print('[Saving Snapshot:]', save_path + 'Moded-HarD-MSEG-best.pth', 'New best:', meandice)
        else:
            print('Current best is:', best)
        torch.save(model.state_dict(), save_path + 'Moded-HarD-MSEG-last.pth' )
    return best # return best meandice for save the best later 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epoch', type=int,
                        default=300, help='epoch number')
    
    parser.add_argument('--lr', type=float,
                        default=1e-2, help='learning rate')
    
    parser.add_argument('--optimizer', type=str,
                        default='SGD', help='choosing optimizer Adam or SGD')
    
    parser.add_argument('--augmentation', type=bool,
                        default=True, help='choose to do random flip rotation')
    
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    
    parser.add_argument('--trainsize', type=int,
                        default=512, help='training dataset size')
    
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    
    parser.add_argument('--train_path', type=str,
                        default='/work/james128333/PraNet/TrainDataset', help='path to train dataset')
    
    parser.add_argument('--test_path', type=str,
                        default='/work/james128333/PraNet/TestDataset/Kvasir' , help='path to testing Kvasir dataset')
    
    parser.add_argument('--train_save', type=str,
                        default='Moded-HarDMSEG-best')

    parser.add_argument('--resume', type=bool, default=False)

    parser.add_argument('--pth_path', type=str,
                        default='snapshots/Moded-HarDMSEG-best/Moded-HarDMSEG-best.pth')
    
    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = HarDMSEG(model_variant='HarDNet68ds', use_attention=True, activation='mish')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    
    if opt.resume:
        model.load_state_dict(torch.load(opt.pth_path))
    model.to(device)

    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    params = model.parameters()
 
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation = opt.augmentation)
    total_step = len(train_loader)

    # Summarize model
    summary(model, input_size=(32, 3, 512, 512))

    print("#"*20, "Start Training", "#"*20)

    params = model.parameters()
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay = 1e-4, momentum = 0.9)
        # optimizer = torch.optim.SGD(params, opt.lr, momentum = 0.9)

    print(optimizer)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.3, steps_per_epoch=len(train_loader)*3, epochs=opt.epoch)

    best_dice = 0.0
    for epoch in range(1, opt.epoch):
        # adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        best_dice = train(train_loader, model, optimizer, scheduler, epoch, opt.test_path, best_dice)