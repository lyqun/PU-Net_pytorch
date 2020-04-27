import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import PUNet
from dataset import PUNET_Dataset_Whole
from ply_utils import save_ply

import os
import argparse
import numpy as np

torch.cuda.set_device(4)

if __name__ == '__main__':
    resume_path = './logs/test_log/punet_epoch_99.pth'
    model = PUNet(npoint=1024, up_ratio=4, 
                use_normal=False, use_bn=True)

    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval().cuda()

    eval_dst = PUNET_Dataset_Whole()
    eval_loader = DataLoader(eval_dst, batch_size=1, 
                        shuffle=False, pin_memory=True, num_workers=0)
    
    names = eval_dst.names
    for itr, batch in enumerate(eval_loader):
        name = names[itr]
        points = batch.float().cuda()
        preds = model(points, npoint=points.shape[1])
        
        preds = preds.data.cpu().numpy()
        save_ply('./outputs/{}.ply'.format(name), preds[0])
        print('{} with shape {}, output shape {}'.format(name, points.shape, preds.shape))
