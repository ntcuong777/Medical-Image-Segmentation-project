# Code for MedT

import os
import datetime
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from metrics import LogNLLLoss
from module.segmenter.medical_transformer import medt
from config import TrainConfig, TestConfig
from utils.losses import StructureLoss, DiceBCELoss, DiceFocalLoss, FocalTverskyLoss
from data_utils.dataloader import get_train_loader, get_test_loader

def test(model):
    model.eval()

    config = TestConfig.load_config_class('config/test_config/test_config.yaml')

    test_loader = get_test_loader(config)
    b = 0.0
    for i, (image, gt) in enumerate(test_loader, start=1):
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


def train_medt(config: TrainConfig, model_name='MedT'):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "axialunet":
        model = medt.axialunet(img_size = config.input_dim, imgchan = config.num_channels)
    elif model_name == "MedT":
        model = medt.axialnet.MedT(img_size = config.input_dim, imgchan = config.num_channels)
    elif model_name == "gatedaxialunet":
        model = medt.axialnet.gated(img_size = config.input_dim, imgchan = config.num_channels)
    elif model_name == "logo":
        model = medt.axialnet.logo(img_size = config.input_dim, imgchan = config.num_channels)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    criterion = DiceBCELoss() # LogNLLLoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=config.learning_rate, weight_decay=1e-5)


    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    seed = 3000
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataloader = get_train_loader(config)
    save_path = config.save_model_path
    os.makedirs(save_path, exist_ok=True)

    best_dice = 0.0
    for epoch in range(config.epochs):
        epoch_running_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader, start=1):
            X_batch = Variable(X_batch.to(device ='cuda'))
            y_batch = Variable(y_batch.to(device='cuda'))
            
            # ===================forward=====================
            output = model(X_batch)
            loss = criterion(output, y_batch)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_running_loss += loss.item()

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch, config.epochs, epoch_running_loss/(batch_idx+1)))
        
        if epoch == 10:
            for param in model.parameters():
                param.requires_grad = True

        # Test & save best every epoch
        meandice = test(model)
        print('{} Epoch [{:03d}/{:03d}], mDice = {:05f}'.format(datetime.now(), epoch, config.epochs, meandice))
        if meandice > best_dice:
            best_dice = meandice
            torch.save(model.state_dict(), os.path.join(save_path, config.model_name + '-best.pt'))
            print('[Saving Snapshot:]', os.path.join(save_path, config.model_name + '-best.pt'), 'New best:', meandice)
        else:
            print('Current best is:', best_dice)

        if ((epoch+1) % config.save_freq) ==0:
            torch.save(model.state_dict(), os.path.join(save_path, config.model_name + '-{}.pt'.format(epoch)))