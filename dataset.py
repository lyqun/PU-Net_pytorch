import torch.utils.data as torch_data
import h5py
import numpy as np
from utils import utils
from glob import glob
import os

class PUNET_Dataset_Whole(torch_data.Dataset):
    def __init__(self, data_dir='./datas/test_data/our_collected_data/MC_5k'):
        super().__init__()

        file_list = os.listdir(data_dir)
        self.names = [x.split('.')[0] for x in file_list]
        self.sample_path = [os.path.join(data_dir, x) for x in file_list]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        points = np.loadtxt(self.sample_path[index])
        return points


class PUNET_Dataset_WholeFPS_1k(torch_data.Dataset):
    def __init__(self, data_dir='./datas/test_data/obj_1k', use_norm=True):
        super().__init__()
        self.use_norm = use_norm

        folder_1k = os.path.join(data_dir, 'data_1k')
        folder_4k = os.path.join(data_dir, 'data_4k')
        file_list = os.listdir(folder_1k)
        self.names = [x.split('_')[0] for x in file_list]
        self.path_1k = [os.path.join(folder_1k, x) for x in os.listdir(folder_1k)]
        self.path_4k = [os.path.join(folder_4k, x) for x in os.listdir(folder_4k)]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        points = np.load(self.path_1k[index])
        gt = np.load(self.path_4k[index])

        if self.use_norm:
            centroid = np.mean(gt[:, :3], axis=0, keepdims=True) # 1, 3
            furthest_distance = np.amax(np.sqrt(np.sum((gt[:, :3] - centroid) ** 2, axis=-1)), axis=0, keepdims=True)

            gt[:, :3] -= centroid
            gt[:, :3] /= np.expand_dims(furthest_distance, axis=-1)
            points[:, :3] -= centroid
            points[:, :3] /= np.expand_dims(furthest_distance, axis=-1)
            return points, gt, np.array([1.0])
        else:
            raise NotImplementedError


class PUNET_Dataset(torch_data.Dataset):
    def __init__(self, h5_file_path='./datas/Patches_noHole_and_collected.h5', 
                    skip_rate=1, npoint=1024, use_random=True, use_norm=True, split='train', is_training=True):
        super().__init__()
        
        self.npoint = npoint
        self.use_random = use_random
        self.use_norm = use_norm
        self.is_training = is_training

        h5_file = h5py.File(h5_file_path)
        self.gt = h5_file['poisson_4096'][:] # [:] h5_obj => nparray
        self.input = h5_file['poisson_4096'][:] if use_random \
                            else h5_file['montecarlo_1024'][:]
        
        if split in ['train', 'test']:
            with open('./datas/{}_list.txt'.format(split), 'r') as f:
                split_choice = [int(x) for x in f]
            self.gt = self.gt[split_choice, ...]
            self.input = self.input[split_choice, ...]
        elif split != 'all':
            raise NotImplementedError

        assert len(self.input) == len(self.gt), 'invalid data'
        self.data_npoint = self.input.shape[1]

        centroid = np.mean(self.gt[..., :3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((self.gt[..., :3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        self.radius = furthest_distance[:, 0] # not very sure?

        if use_norm:
            self.radius = np.ones(shape=(len(self.input)))
            self.gt[..., :3] -= centroid
            self.gt[..., :3] /= np.expand_dims(furthest_distance, axis=-1)
            self.input[..., :3] -= centroid
            self.input[..., :3] /= np.expand_dims(furthest_distance, axis=-1)

        self.input = self.input[::skip_rate]
        self.gt = self.gt[::skip_rate]
        self.radius = self.radius[::skip_rate]

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        input_data = self.input[index]
        gt_data = self.gt[index]
        radius_data = np.array([self.radius[index]])

        sample_idx = utils.nonuniform_sampling(self.data_npoint, sample_num=self.npoint)
        input_data = input_data[sample_idx, :]

        if self.use_norm:
            if not self.is_training:
                return input_data, gt_data, radius_data
            
            # for data aug
            input_data, gt_data = utils.rotate_point_cloud_and_gt(input_data, gt_data)
            input_data, gt_data, scale = utils.random_scale_point_cloud_and_gt(input_data, gt_data,
                                                                               scale_low=0.9, scale_high=1.1)
            input_data, gt_data = utils.shift_point_cloud_and_gt(input_data, gt_data, shift_range=0.1)
            radius_data = radius_data * scale

            # for input aug
            if np.random.rand() > 0.5:
                input_data = utils.jitter_perturbation_point_cloud(input_data, sigma=0.025, clip=0.05)
            if np.random.rand() > 0.5:
                input_data = utils.rotate_perturbation_point_cloud(input_data, angle_sigma=0.03, angle_clip=0.09)
        else:
            raise NotImplementedError

        return input_data, gt_data, radius_data

            
if __name__ == '__main__':
    test_choice = np.random.choice(4000, 800, replace=False)
    # f_test = open('test_list.txt', 'w')
    # f_train = open('train_list.txt', 'w')
    # train_list = []
    # test_list = []
    # for i in range(4000):
    #     if i in test_choice:
    #         test_list.append(i)
    #     else:
    #         train_list.append(i)
    # f_test.close()
    # f_train.close()

    # dst = PUNET_Dataset_WholeFPS_1k()
    # for batch in dst:
    #     pcd, gt, r = batch
    #     print(pcd.shape)
    #     print(gt.shape)
    #     print(r.shape)
    #     import pdb
    #     pdb.set_trace()

    ## test <PUNET_Dataset>
    # dst = PUNET_Dataset()
    # print(len(dst))
    # for batch in dst:
    #     pcd, gt, r = batch
    #     print(pcd.shape)
    #     import pdb
    #     pdb.set_trace()

    ## test <PUNET_Dataset_Whole>
    # dst = PUNET_Dataset_Whole()
    # points, name = dst[0]
    # print(points, name)