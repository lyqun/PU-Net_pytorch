import torch
from torch.utils.cpp_extension import load
import os

script_dir = os.path.dirname(__file__)
sources = [
    os.path.join(script_dir, "auction_match_gpu.cpp"),
    os.path.join(script_dir, "auction_match_gpu.cu"),
]

am = load(name="am", sources=sources)

class AuctionMatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1: torch.Tensor, xyz2: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz1: (B, N, 3)
        :param xyz2: (B, N, 3)
        :return:
             match_left: (B, N) tensor containing the set
             match_right: (B, N) tensor containing the set
        """
        assert xyz1.is_contiguous() and xyz2.is_contiguous()
        assert xyz1.shape[1] <= 4096

        B, N, _ = xyz1.size()
        match_left = torch.cuda.IntTensor(B, N)
        match_right = torch.cuda.IntTensor(B, N)
        temp = torch.cuda.FloatTensor(B, N, N).fill_(0)

        am.auction_match_cuda(B, N, xyz1, xyz2, match_left, match_right, temp)
        return match_left, match_right

    @staticmethod
    def backward(ml, mr, a=None):
        return None, None

auction_match = AuctionMatch.apply

if __name__ == '__main__':
    import numpy as np
    # p1 = torch.randn(1, 128, 3).float().cuda()
    # p2 = torch.randn(1, 128, 3).float().cuda()
    p1 = torch.from_numpy(np.array([[[1,0,0], [2,0,0], [3,0,0], [4,0,0]]], dtype=np.float32)).cuda()
    p2 = torch.from_numpy(np.array([[[-10,0,0], [1,0, 0], [2,0, 0], [3,0,0]]], dtype=np.float32)).cuda()
    ml, mr = auction_match(p2, p1)
    print(ml, mr)