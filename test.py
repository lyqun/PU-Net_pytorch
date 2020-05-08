import argparse
import os, sys

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument("--model", type=str, default='punet')
parser.add_argument('--up_ratio',  type=int,  default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument("--use_bn", action='store_true', default=False)
parser.add_argument("--use_res", action='store_true', default=False)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument('--resume', type=str, required=True)

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.ply_utils import save_ply
from utils.utils import save_xyz_file
import numpy as np

from dataset import PUNET_Dataset_Whole
import importlib


if __name__ == '__main__':
    MODEL = importlib.import_module('models.' + args.model)
    model = MODEL.get_model(npoint=1024, up_ratio=args.up_ratio, 
                use_normal=False, use_bn=args.use_bn, use_res=args.use_res)

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
        save_ply(os.path.join(args.save_dir, '{}_input.ply'.format(name)), points[0, :, :3])
        save_ply(os.path.join(args.save_dir, '{}.ply'.format(name)), preds[0])
        save_xyz_file(preds[0], os.path.join(args.save_dir, '{}.xyz'.format(name)))
        print('{} with shape {}, output shape {}'.format(name, points.shape, preds.shape))
