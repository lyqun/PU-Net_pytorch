import torch
from pointnet2 import pointnet2_utils
from dataset import PUNET_Dataset_Whole
import numpy as np
import os

def FPS_np(xyz, npoint):
    N, _ = xyz.shape
    sample_indices = np.zeros(npoint, dtype=np.int)
    farthest_index = np.random.randint(0, N, dtype=np.int)
    distance = np.ones(N) * 1e10
    for i in range(npoint):
        sample_indices[i] = farthest_index
        centroid = xyz[farthest_index, :]
        dist2 = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist2 < distance
        distance[mask] = dist2[mask]
        farthest_index = np.argmax(distance)
    return sample_indices


def FPS_cuda(points, npoint):
    points_cuda = torch.from_numpy(points).float().cuda()
    points_cuda = points_cuda.unsqueeze(0)
    with torch.no_grad():
        index_cuda = pointnet2_utils.furthest_point_sample(
            points_cuda, npoint)
    return index_cuda.squeeze(0).cpu().numpy()


if __name__ == '__main__':
    data_folder = './datas/test_data/our_collected_data/MC_5k'
    save_folder = './datas/test_data/obj_1k'

    dst = PUNET_Dataset_Whole(data_dir=data_folder)
    obj_names = dst.names
    for i, points in enumerate(dst):
        print(' -- processing {}/{}'.format(i + 1, len(dst)))
        index_4k = FPS_cuda(points[:, :3], 4096)
        points_fps_4k = points[index_4k, :]
        points_fps_1k = points_fps_4k[:1024, :]
        np.save(os.path.join(save_folder, 
                'data_1k/{}_1k.npy'.format(obj_names[i])), points_fps_1k)
        np.save(os.path.join(save_folder, 
                'data_4k/{}_4k.npy'.format(obj_names[i])), points_fps_4k)

        

