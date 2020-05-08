import torch
import torch.nn as nn
import torch.nn.functional as F
import pointnet2.pytorch_utils as pt_utils

def get_model(npoint=1024, up_ratio=2, use_normal=False, use_bn=False, use_res=False):
    return PointNet(npoint, up_ratio, use_normal, use_bn, use_res)

class PointNet(nn.Module):
    def __init__(self, npoint=1024, up_ratio=2, use_normal=False, use_bn=False, use_res=False):
        super().__init__()

        self.npoint = npoint
        self.use_normal = use_normal
        self.up_ratio = up_ratio

        mlps = [64, 128, 256, 1024]
        fc_mlps = [1024, 512, 64, 3]

        ## for feature extraciton
        in_ch = 3 if not use_normal else 6
        self.SA_layer = pt_utils.SharedMLP(
                [in_ch] + mlps,
                bn=use_bn)
        
        ## feature Expansion
        in_ch = mlps[-1] + 3 # fp output + input xyz
        self.FC_Modules = nn.ModuleList()
        for k in range(up_ratio):
            self.FC_Modules.append(
                pt_utils.SharedMLP(
                    [in_ch, 256, 128],
                    bn=use_bn))

        ## coordinate reconstruction
        in_ch = 128
        self.pcd_layer = nn.Sequential(
            pt_utils.SharedMLP([in_ch, 64], bn=use_bn),
            pt_utils.SharedMLP([64, 3], activation=None, bn=False)) 


    def forward(self, points, npoint=None):
        ## points: bs, N, 3/6
        xyz = points[..., :3].contiguous()
        feats = points if self.use_normal else points[..., :3]
        npoint = xyz.shape[1]

        feats = feats.transpose(1, 2).unsqueeze(-1).contiguous() # b, C, N, 1
        feats = self.SA_layer(feats)

        feats = F.max_pool2d(feats, kernel_size=[npoint, 1]) # b, C, 1, 1
        feats = feats.expand(-1, -1, npoint, -1) # b, C, N, 1
        feats = torch.cat(
            [xyz.transpose(1, 2).unsqueeze(-1), feats], dim=1).contiguous()

        ## expansion
        r_feats = []
        for k in range(len(self.FC_Modules)):
            feat_k = self.FC_Modules[k](feats) # bs, mid_ch, N, 1
            r_feats.append(feat_k)
        r_feats = torch.cat(r_feats, dim=2) # bs, mid_ch, r * N, 1

        ## reconstruction
        output = self.pcd_layer(r_feats) # bs, 3, r * N, 1
        return output.squeeze(-1).transpose(1, 2).contiguous() # bs, 3, r * N


if __name__ == '__main__':
    model = PointNet(up_ratio=2, use_normal=False).cuda()
    points = torch.randn([1, 1024, 3]).float().cuda()
    output = model(points)
    print(output.shape)