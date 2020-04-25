import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pointnet2.pointnet2_utils import grouping_operation as group_point
from utils import knn_point
# from emd.emd import earth_mover_distance
from chamfer_distance import ChamferDistance

from model import PUNet
from dataset import PUNET_Dataset

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=4, help='GPU to use')
parser.add_argument('--log_dir', default='logs/test_log', help='Log dir [default: logs/test_log]')
parser.add_argument('--npoint', type=int, default=1024,help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--up_ratio',  type=int,  default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epochs to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--workers', type=int, default=4)

args = parser.parse_args()
torch.cuda.set_device(args.gpu)


class UpsampleLoss(nn.Module):
    def __init__(self, alpha=0.1, nn_size=5, radius=0.07, h=0.03, eps=1e-12):
        super().__init__()
        self.alpha = alpha
        self.nn_size = nn_size
        self.radius = radius
        self.h = h
        self.eps = eps
        self.cd_loss = ChamferDistance()

    def get_emd_loss(self, pred, gt, pcd_radius):
        dist = earth_mover_distance(pred, gt, transpose=False)
        dist_norm = dist / pcd_radius
        emd_loss = torch.mean(dist_norm)
        return emd_loss

    def get_cd_loss(self, pred, gt, pcd_radius):
        cost_for, cost_bac = self.cd_loss(gt, pred)
        cost = 0.8 * cost_for + 0.2 * cost_bac
        cost /= pcd_radius
        cost = torch.mean(cost)
        return cost

    def get_repulsion_loss(self, pred):
        _, idx = knn_point(self.nn_size, pred, pred)
        idx = idx.transpose(1, 2).to(torch.int32)
        idx = idx[:, :, 1:] # remove first one
        idx = idx.contiguous()
        grouped_points = group_point(pred, idx)

        grouped_points = grouped_points - pred.unsqueeze(-1)
        dist2 = torch.sum(grouped_points ** 2, dim=1)
        dist2 = torch.max(dist2, torch.tensor(self.eps).cuda())
        dist = torch.sqrt(dist2)
        weight = torch.exp(- dist2 / self.h ** 2)

        uniform_loss = torch.mean((self.radius - dist) * weight)
        # uniform_loss = torch.mean(self.radius - dist * weight) # punet
        return uniform_loss

    def forward(self, pred, gt, pcd_radius):
        return self.get_cd_loss(pred, gt, pcd_radius) \
             + self.alpha * self.get_repulsion_loss(pred)


if __name__ == '__main__':
    train_dst = PUNET_Dataset(npoint=args.npoint, 
            use_random=True, use_norm=True)
    train_loader = DataLoader(train_dst, batch_size=args.batch_size, 
                        shuffle=True, pin_memory=True, num_workers=args.workers)

    model = PUNet(npoint=args.npoint, up_ratio=args.up_ratio, 
                use_normal=False, use_bn=True)
    model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    loss_func = UpsampleLoss()

    model.train()
    for epoch in range(args.max_epoch):
        loss_list = []
        for batch in train_loader:
            optimizer.zero_grad()
            input_data, gt_data, radius_data = batch

            input_data = input_data.float().cuda()
            gt_data = gt_data.float().cuda()
            gt_data = gt_data[..., :3].contiguous()
            radius_data = radius_data.float().cuda()

            preds = model(input_data)
            loss = loss_func(preds, gt_data, radius_data)

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
        print(' -- epoch {}, loss {}.'.format(epoch, np.mean(loss_list)))