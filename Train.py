from numpy.core.fromnumeric import mean
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.dataloader import get_loader,test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
from torchinfo import summary
import torch.nn as nn
from utils.losses import StructureLoss, FocalTverskyLoss, DiceFocalLoss, DiceBCELoss
from module.segmenter.segmenter_factory import SegmenterFactory

def test(model, path, requires_test=True):
    
    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    #####                                             #####

    dirs_in_path = None
    if requires_test:
        dirs_in_path = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

    if (requires_test) and ('images' in dirs_in_path) and ('masks' in dirs_in_path) and (len(dirs_in_path) > 2):
        # The folder contains data from different datasets
        sum_acc = 0
        for fol in dirs_in_path:
            if fol == 'images' or fol == 'masks':
                continue
            
            # recursive call to eval each dataset separately, we have already know that the recursive call
            # will be given a specific dataset folder, no need to test it again 
            sum_acc += test(model, os.path.join(data_path, fol), requires_test=False)

        return sum_acc / (len(dirs_in_path) - 2) # Average accuracy over all datasets

    else: # The folder only contains data from one dataset only
        model.eval()
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, 352)
        b=0.0
        for i in range(100):
            image, gt, _ = test_loader.load_data()
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



def train(train_loader, model, optimizer, epoch, test_path, best_dice):
    model.train()

    # ---- the loss to use ----
    loss_fn = FocalTverskyLoss()

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
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear')
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear')

            # ---- forward ----
            out = model(images)

            # ---- loss function ----
            loss = loss_fn(out, gts)
            
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            # ---- recording loss ----
            if rate == 1:
                loss_record5.update(loss.data, opt.batchsize)

        # ---- training visualization ----
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
            torch.save(model.state_dict(), save_path + 'Medical_Model-best.pth' )
            print('[Saving Snapshot:]', save_path + 'Medical_Model-best.pth', 'New best:', meandice)
        else:
            print('Current best is:', best)
        torch.save(model.state_dict(), save_path + 'Medical_Model-last.pth')
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
    torch.cuda.set_device(0)  # set your gpu device
    model = SegmenterFactory.create_segmenter_as(segmenter='MobileWnet')
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
 
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation = opt.augmentation)
    total_step = len(train_loader)

    # Summarize model
    # summary(model, input_size=(8, 3, 512, 512))

    print("#"*20, "Start Training", "#"*20)

    params = model.parameters()
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay = 1e-4, momentum = 0.9)

    print(optimizer)

    best_dice = 0.0
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        best_dice = train(train_loader, model, optimizer, epoch, opt.test_path, best_dice)