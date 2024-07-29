import os
import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import dataset
import argparse
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.LFSS_model import LFSS
from apex import amp
CE = torch.nn.BCEWithLogitsLoss()

def get_model(cfg):
    model = LFSS(cfg).cuda()
    return model.cuda()

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

# F Loss
def f_loss(pred, mask, beta=0.3, log_like=False):
    eps = 1e-10
    n = n = pred.size(0)
    tp = (pred * mask).view(n, -1).sum(dim=1)
    h = beta * mask.view(n, -1).sum(dim=1) + pred.view(n, -1).sum(dim=1)
    fm = (1+beta) * tp / (h+eps)
    if log_like:
        floss = -torch.log(fm)
    else:
        floss = (1-fm)

    return floss.mean()

import shutil

def train(Dataset, parser):
    args = parser.parse_args()

    cfg = Dataset.Config(datapath=args.dataset, savepath=args.savepath, mode='train', batch=args.batchsize, lr=args.lr, momen=args.momen, decay=args.decay, epoch=args.epoch)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True)

    net = get_model(cfg)
    net.train(True)
    net.cuda()

    base, head = [], []
    
    for name, param in net.named_parameters():
        if 'encoder.conv1' in name or 'encoder.bn1' in name:
            pass
        elif 'encoder' in name:
            base.append(param)
        elif 'network' in name:
            base.append(param)     
        else:
            head.append(param)

    optimizer = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)

    net, optimizer = amp.initialize(net, optimizer, opt_level='O1') 
    sw = SummaryWriter(cfg.savepath)
    
    global_step = 0
    
    head1 = '../tools/evaltool/Prediction'
    if os.path.exists(head1):
            shutil.rmtree(head1)
    os.mkdir(head1)
    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        head = '../tools/evaltool/Prediction/Images'
        if os.path.exists(head):
            shutil.rmtree(head)
        os.mkdir(head)

        for step, (image, mask, depth, name) in enumerate(loader):
            image, mask, depth = image.cuda(), mask.cuda(), depth.cuda()

            out2, out3, out4, out5, pose= net(image, depth)

            loss1  = F.binary_cross_entropy_with_logits(out2, mask) + iou_loss(out2, mask)
            loss2  = F.binary_cross_entropy_with_logits(out3, mask) + iou_loss(out3, mask)
            loss3  = F.binary_cross_entropy_with_logits(out4, mask) + iou_loss(out4, mask)
            loss4  = F.binary_cross_entropy_with_logits(out5, mask) + iou_loss(out5, mask)
            lossp  = F.binary_cross_entropy_with_logits(pose, mask) + iou_loss(pose, mask)

            loss = loss1 + loss2 +loss3 + loss4 + lossp
            
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward(retain_graph=True)
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            ## log
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            sw.add_scalars('loss', {'loss1':loss1.item(), 'loss2':loss2.item(), 'loss3':loss3.item(), 'loss4':loss4.item(), 'lossp':lossp.item()}, global_step=global_step)


            print('%s | epoch:%d/%d | step:%d/%d | lr=%.6f | loss1-5=%.6f, %.6f, %.6f, %.6f, %.6f'
                %(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'), epoch+1, cfg.epoch, step + 1, len(loader), optimizer.param_groups[0]['lr'], loss1.item(), loss2.item(), loss3.item(), loss4.item(), lossp.item()))

        if epoch >= 0:
            torch.save(net.state_dict(), cfg.savepath+'/'+ "LFSS" +str(epoch+1))
            for path in ['../datasets/BBS/Test']:
                t = Valid(dataset, path, epoch, 'LFSS', args.savepath)
                t.save()
            


class Valid(object):
    def __init__(self, Dataset, Path, epoch, model_name, checkpoint_path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=Path, snapshot=checkpoint_path+'/'+model_name+str(epoch+1), mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False)

        ## network
        self.net = get_model(self.cfg)
        self.net.train(False)
        self.net.cuda()
        self.epoch = epoch


    def save(self):
        with torch.no_grad():
            for image, depth, (H, W), name in self.loader:
                image, depth, shape = image.cuda().float(), depth.cuda().float(), (H, W)
                out1, out2, out3, out4, pose = self.net(image, depth, shape, name)
                pred = torch.sigmoid(out4[0,0]).cpu().numpy()*255

                head = '../tools/evaltool/Prediction/LFSS-valid-'+str(self.epoch+1)+'/' + self.cfg.datapath.split('/')[-2]

                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))
 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='../datasets/BBS/Train')
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momen", type=float, default=0.9)  
    parser.add_argument("--decay", type=float, default=1e-4)  
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--savepath", default='../LFSS_model')
    parser.add_argument("--valid", default=True)  
    train(dataset, parser)

