import torch
import torch.nn as nn
from pointnet2.pointnet2_modules import PointNetSSG_Base, PointnetSAModule, PointnetFPModule
import pointnet2.pytorch_utils as pt_utils
from models.base import Resnet_BaseBlock

def get_model(npoint=1024, up_ratio=2, use_normal=False, use_bn=False, use_res=False):
    return PUNetRes(npoint, up_ratio, use_normal, use_bn, use_res)

class PUNetRes(nn.Module):
    def __init__(self, npoint=1024, up_ratio=2, use_normal=False, use_bn=False, use_res=False):
        super().__init__()

        self.npoint = npoint
        self.use_normal = use_normal
        self.up_ratio = up_ratio

        self.npoints = [
            npoint, 
            npoint // 2, 
            npoint // 4, 
            npoint // 8
        ]

        mlps = [64, 128, 256, 512]

        radius = [0.05, 0.1, 0.2, 0.3]

        nsamples = [32, 32, 32, 32]

        in_ch = 0 if not use_normal else 3
        self.conv0 = PointnetSAModule(
            npoint=self.npoints[0],
            radius=radius[0],
            nsample=nsamples[0],
            mlp=[in_ch, 32],
            use_xyz=True)
        in_ch = 32

        ## for 4 downsample layers
        self.SA_modules = nn.ModuleList()
        for k in range(len(self.npoints)):
            self.SA_modules.append(
                Resnet_BaseBlock(
                    PNCONV=PointNetSSG_Base,
                    npoint=self.npoints[k], 
                    nsample=nsamples[k], 
                    radius=radius[k], 
                    in_channel=in_ch, 
                    out_channel=mlps[k], 
                    bn=use_bn, 
                    use_xyz=True))
            in_ch = mlps[k]

        ## upsamples for layer 2 ~ 4
        self.FP_Modules = nn.ModuleList()
        for k in range(len(self.npoints) - 1):
            self.FP_Modules.append(
                PointnetFPModule(
                    mlp=[mlps[k + 1], 64], 
                    bn=use_bn))
        
        ## feature Expansion
        in_ch = len(self.npoints) * 64 + 3 # 4 layers + input xyz
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
            npoints = [npoint]
            for k in range(len(self.npoints) - 1):
                npoints.append(npoint // 2 ** k)

        ## points: bs, N, 3/6
        xyz = points[..., :3].contiguous()
        feats = points[..., 3:].transpose(1, 2).contiguous() \
            if self.use_normal else None
        xyz, feats = self.conv0(xyz, feats, npoint=npoints[0])

        ## downsample
        l_xyz, l_feats = [xyz], [feats]
        for k in range(len(self.SA_modules)):
            lk_xyz, lk_feats = self.SA_modules[k](l_xyz[k], l_feats[k], npoint=npoints[k])
            l_xyz.append(lk_xyz)
            l_feats.append(lk_feats)

        ## upsample
        up_feats = []
        for k in range(len(self.FP_Modules)):
            upk_feats = self.FP_Modules[k](xyz, l_xyz[k + 2], None, l_feats[k + 2])
            up_feats.append(upk_feats)

        ## aggregation
        # [xyz, l0, l1, l2, l3]
        feats = torch.cat([
            xyz.transpose(1, 2).contiguous(),
            l_feats[1],
            *up_feats], dim=1).unsqueeze(-1) # bs, mid_ch, N, 1

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
    model = PUNet(up_ratio=2, use_normal=True).cuda()
    points = torch.randn([1, 1024, 6]).float().cuda()
    output = model(points)
    print(output.shape)