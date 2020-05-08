import torch
import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetSAModule, PointnetFPModule
import pointnet2.pytorch_utils as pt_utils

def get_model(npoint=1024, up_ratio=2, use_normal=False, use_bn=False, use_res=False):
    return PointNet2_SSG(npoint, up_ratio, use_normal, use_bn, use_res)

class PointNet2_SSG(nn.Module):
    def __init__(self, npoint=1024, up_ratio=2, use_normal=False, use_bn=False, use_res=False):
        super().__init__()

        self.npoint = npoint
        self.use_normal = use_normal
        self.up_ratio = up_ratio

        self.npoints = [
            npoint // 2, 
            npoint // 4, 
            npoint // 8
        ]

        mlps = [
            [64, 64, 128],
            [128, 128, 256],
            [256, 256, 512]
        ]

        fp_mlps = [
            [128, 128, 128],
            [256, 128],
            [256, 256]
        ]

        radius = [0.1, 0.2, 0.3]

        nsamples = [32, 32, 32, 32]

        in_ch = 0 if not use_normal else 3
        self.conv0 = PointnetSAModule(
                npoint=self.npoint,
                radius=radius[0] / 2,
                nsample=nsamples[0],
                mlp=[in_ch, 32, 32, 64],
                use_xyz=True,
                use_res=use_res,
                bn=use_bn)

        ## for 4 downsample layers
        in_ch = 64
        skip_ch_list = [in_ch]
        self.SA_modules = nn.ModuleList()
        for k in range(len(self.npoints)):
            sa_mlpk = [in_ch] + mlps[k]
            print(' -- sa_mlpk {}, radius {}, nsample {}, npoint {}.'.format(
                sa_mlpk, radius[k], nsamples[k], self.npoints[k]))
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=self.npoints[k],
                    radius=radius[k],
                    nsample=nsamples[k],
                    mlp=sa_mlpk,
                    use_xyz=True,
                    use_res=use_res,
                    bn=use_bn))
            in_ch = mlps[k][-1]
            skip_ch_list.append(in_ch)

        ## upsamples for layer 2 ~ 4
        self.FP_Modules = nn.ModuleList()
        for k in range(len(self.npoints)):
            pre_ch = fp_mlps[k + 1][-1] if k < len(self.npoints) - 1 else skip_ch_list[-1]
            fp_mlpk = [pre_ch + skip_ch_list[k]] + fp_mlps[k]
            print(' -- fp_mlpk:', fp_mlpk)
            self.FP_Modules.append(
                PointnetFPModule(
                    mlp=fp_mlpk,
                    bn=use_bn))
        
        ## feature Expansion
        in_ch = fp_mlps[0][-1] + 3 # fp output + input xyz
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
        if npoint is None:
            npoints = [None] * len(self.npoints)
        else:
            npoints = []
            for k in range(len(self.npoints)):
                npoints.append(npoint // 2 ** (k + 1))

        ## points: bs, N, 3/6
        xyz = points[..., :3].contiguous()
        feats = points[..., 3:].transpose(1, 2).contiguous() \
            if self.use_normal else None
        _, feats = self.conv0(xyz, feats, npoint=npoint if npoint is not None else None)

        ## downsample
        l_xyz, l_feats = [xyz], [feats]
        for k in range(len(self.SA_modules)):
            lk_xyz, lk_feats = self.SA_modules[k](l_xyz[k], l_feats[k], npoint=npoints[k])
            l_xyz.append(lk_xyz)
            l_feats.append(lk_feats)

        ## upsample
        l_fp = l_feats[-1]
        for i in range(len(self.FP_Modules)):
            i = len(self.npoints) - i
            l_fp = self.FP_Modules[i - 1](l_xyz[i - 1], l_xyz[i], l_feats[i - 1], l_fp)

        ## aggregation
        # [xyz, l_fp]
        feats = torch.cat([
            xyz.transpose(1, 2).contiguous(),
            l_fp], dim=1).unsqueeze(-1) # bs, mid_ch, N, 1

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
    model = PointNet2_SSG(up_ratio=2, use_normal=False).cuda()
    points = torch.randn([1, 1024, 3]).float().cuda()
    output = model(points)
    print(output.shape)