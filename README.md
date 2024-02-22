## PU-Net: Point Cloud Upsampling Network

PyTorch implementation of PU-Net. Official TF implementation: [punet_tf](https://github.com/yulequan/PU-Net). This repo is tested with PyTorch 1.2, cuda 10.0 and Python 3.6.

### 1. Installation

Follow [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch) to compile pointnet utils. Or run the following commands.

```shell
cd pointnet2
python setup.py install
```

You should install `knn_cuda` by running the following command or refering to [KNN_CUDA](https://github.com/unlimblue/KNN_CUDA)

```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```


### 2. Data Preparation

#### a. Prepare Patches

First, follow the official repo, download patches in HDF5 format from [GoogleDrive](https://drive.google.com/file/d/1wMtNGvliK_pUTogfzMyrz57iDb_jSQR8/view?usp=sharing) and put it into `./datas/`. Patches are splitted for training (3200) and testing (800). See `./datas/train_list.txt` and `./datas/test_list.txt`.

#### b. Prepare Datas for Visualization

Objects with 5k points for testing can be downloaded from the official repo, [link](https://github.com/yulequan/PU-Net/tree/master/data/test_data/our_collected_data/MC_5k). Put them into `./datas/test_data/our_collected_data/MC_5k`.

#### c. Prepare Datas for NUC Calculation

The training and testing mesh files can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1R21MD1O6q8E7ANui8FR0MaABkKc30PG4/view?usp=sharing). Put test mesh files into `./datas/test_data/test_mesh`.

The `./datas` folder should be organized as follows:

```shell
PU-Net_pytorch
├── datas
│   ├── Patches_noHole_and_collected.h5
│   ├── test_list.txt
│   ├── train_list.txt
│   ├── test_data
│   │  │   ├── test_mesh
│   │  │   │   ├── *.off
│   │  │   ├── our_collected_data/MC_5k
│   │  │   │   ├── *.xyz
```

### 3. Train

Run the following commands for training.

```shell
mkdir logs
bash train_punet.sh
```

### 4. Evaluation (EMD and CD)

Run the following commands for evaluation.

```shell
python eval.py --gpu 0 --resume logs/punet_baseline/punet_epoch_99.pth
```

### 5. Visualization and Test (NUC)

Run the following commands to generate upsampled datas from full mesh objects with 5k points. Upsampled point clouds are saved in `./outputs/punet_baseline/*.ply`. And the dumpped `*.xyz` files are used for NUC calculation.

```shell
mkdir outputs
bash test_punet.sh
```

#### NUC Calculation

1. install CGAL

2. run the following commands to compile cpp code

   ```shell
   cd nuc_utils
   mkdir build
   cd build
   cmake
   make
   cd ../..
   ```

3. run the following commands to calculate disk density, the results are saved in `./outputs/punet_baseline/`.

   ```
   bash nuc_utils/evaluate_all.sh
   ```

4. run the following commands to calculate NUC

   ```shell
   python nuc_utils/calculate_nuc.py
   ```

Note that, the disk size (D) is 40 in default setting.

### Performance
Please refer to this [issue#1](https://github.com/lyqun/PU-Net_pytorch/issues/1). I will update later.

### Update
1. The auction matching is modified from [PU-Net/code/tp_ops/emd](https://github.com/yulequan/PU-Net/tree/master/code/tf_ops/emd). The number of points should be fewer than 4096 and better chosen as $2^K$ (e.g., 1024, 4096).
2. For the calculation of CD and EMD (evaluation), you should take the square root of the distance to get correct evaluation results.
