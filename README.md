## PU-Net: Point Cloud Upsampling Network

PyTorch implementation of PU-Net. Official TF implementation: [punet_tf](https://github.com/yulequan/PU-Net). This repo is tested with PyTorch 1.2, cuda 10.0 and Python 3.6.

### 1. Installation

Follow [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch) to compile pointnet utils. Or run the following commands.

```shell
cd pointnet2
python setup.py install
```

### 2. Data Preparation

#### a. Prepare Patches

First, follow the official repo, download patches in HDF5 format from [GoogleDrive](https://drive.google.com/file/d/1wMtNGvliK_pUTogfzMyrz57iDb_jSQR8/view?usp=sharing) and put it into `./datas/`. Patches are splitted for training (3200) and testing (800). See `./datas/train_list.txt` and `./datas/test_list.txt`.

#### b. Prepare Datas for Visualization

Full mesh object can be downloaded from the official repo, [link](https://github.com/yulequan/PU-Net/tree/master/data/test_data/our_collected_data/MC_5k). Put the full mesh datas into `./datas/test_data/our_collected_data/MC_5k`.

### 3. Train

Run the following commands for training.

```shell
mkdir logs
bash train_punet.sh
```

### 4. Evaluation

Run the following commands for evaluation.

```shell
python eval.py --gpu 0 --resume logs/punet_baseline/punet_epoch_99.pth
```

### 5. Visualization

Run the following commands to generate upsampled datas from full mesh objects with 5k points. Upsampled point clouds are saved in `./outputs/*.ply`.

```shell
mkdir outputs
python test.py --gpu 0 --resume logs/punet_baseline/punet_epoch_99.pth
```

