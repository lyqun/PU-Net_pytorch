import argparse
import os, sys

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument('--resume', type=str, required=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ply_utils import save_ply
import numpy as np

from model import PUNet
from dataset import PUNET_Dataset_Whole


if __name__ == '__main__':
    model = PUNet(npoint=1024, up_ratio=4, 
                use_normal=False, use_bn=False)

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state'])
    model.eval().cuda()

    eval_dst = PUNET_Dataset_Whole(data_dir='./datas/test_data/our_collected_data/MC_5k')
    eval_loader = DataLoader(eval_dst, batch_size=1, 
                        shuffle=False, pin_memory=True, num_workers=0)
    
    names = eval_dst.names
    for itr, batch in enumerate(eval_loader):
        name = names[itr]
        points = batch.float().cuda()
        preds = model(points, npoint=points.shape[1])
        
        preds = preds.data.cpu().numpy()
        points = points.data.cpu().numpy()
        save_ply('./outputs/{}_input.ply'.format(name), points[0, :, :3])
        save_ply('./outputs/{}.ply'.format(name), preds[0])
        print('{} with shape {}, output shape {}'.format(name, points.shape, preds.shape))
