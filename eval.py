import argparse
import os, sys

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument("--model", type=str, default='punet')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument('--up_ratio',  type=int,  default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument("--use_bn", action='store_true', default=False)
parser.add_argument("--use_res", action='store_true', default=False)
parser.add_argument('--resume', type=str, required=True)

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import PUNET_Dataset_WholeFPS_1k, PUNET_Dataset
from chamfer_distance import chamfer_distance
from auction_match import auction_match
from pointnet2 import pointnet2_utils as pn2_utils
import importlib

def get_emd_loss(pred, gt, pcd_radius):
    idx, _ = auction_match(pred, gt)
    matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
    matched_out = matched_out.transpose(1, 2).contiguous()
    dist2 = (pred - matched_out) ** 2
    dist2 = dist2.view(dist2.shape[0], -1) # <-- ???
    dist2 = torch.mean(dist2, dim=1, keepdims=True) # B,
    dist2 /= pcd_radius
    return torch.mean(dist2)

def get_cd_loss(pred, gt, pcd_radius):
    cost_for, cost_bac = chamfer_distance(gt, pred)
    cost = 0.5 * cost_for + 0.5 * cost_bac
    cost /= pcd_radius
    cost = torch.mean(cost)
    return cost


if __name__ == '__main__':
    MODEL = importlib.import_module('models.' + args.model)
    model = MODEL.get_model(npoint=1024, up_ratio=args.up_ratio, 
                use_normal=False, use_bn=args.use_bn, use_res=args.use_res)

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state'])
    model.eval().cuda()

    eval_dst = PUNET_Dataset(h5_file_path='./datas/Patches_noHole_and_collected.h5', split='test', is_training=False)
    eval_loader = DataLoader(eval_dst, batch_size=args.batch_size, 
                        shuffle=False, pin_memory=True, num_workers=args.workers)

    emd_list = []
    cd_list = []
    with torch.no_grad():
        for itr, batch in enumerate(eval_loader):
            points, gt, radius = batch
            points = points[..., :3].float().cuda().contiguous()
            gt = gt[..., :3].float().cuda().contiguous()
            radius = radius.float().cuda()
            preds = model(points, npoint=None) #points.shape[1])

            emd = get_emd_loss(preds, gt, radius)
            cd = get_cd_loss(preds, gt, radius)
            print(' -- iter {}, emd {}, cd {}.'.format(itr, emd, cd))
            emd_list.append(emd.item())
            cd_list.append(cd.item())
    
    print('mean emd: {}'.format(np.mean(emd_list)))
    print('mean cd: {}'.format(np.mean(cd_list)))
