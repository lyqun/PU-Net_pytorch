import torch.utils.data as torch_data
import h5py
import numpy as np
import utils


class PUNET_Dataset(torch_data.Dataset):
    def __init__(self, h5_file_path='./datas/Patches_noHole_and_collected.h5', 
                    skip_rate=1, npoint=1024, use_random=True, use_norm=True):
        super().__init__()
        
        self.npoint = npoint
        self.use_random = use_random
        self.use_norm = use_norm

        h5_file = h5py.File(h5_file_path)
        self.gt = h5_file['poisson_4096'][:] # [:] h5_obj => nparray
        self.input = h5_file['poisson_4096'][:] if use_random \
                            else h5_file['montecarlo_1024'][:]
        assert len(self.input) == len(self.gt), 'invalid data'
        self.data_npoint = self.input.shape[1]

        centroid = np.mean(self.gt[..., :3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((self.gt[..., :3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        self.radius = furthest_distance[:, 0] # not very sure?

        if use_norm:
            self.radius = np.ones(shape=(len(self.input)))
            self.gt[..., :3] -= centroid
            self.gt[..., :3] /= np.expand_dims(furthest_distance, axis=-1)
            self.input[..., :3] -= - centroid
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
    dst = PUNET_Dataset()
    for batch in dst:
        pcd, gt, r = batch
        print(pcd.shape)