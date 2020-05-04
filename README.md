## PU-Net: Point Cloud Upsampling Network

PyTorch implementation of PU-Net. Official TF implementation: [punet_tf](https://github.com/yulequan/PU-Net). This repo is tested with PyTorch 1.2, cuda 10.0 and Python 3.6.

### 1. Installation

Follow [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch) to compile pointnet utils. Or run the following commands.

```shell
cd pointnet2
python setup.py install
```

### 2. Data Preparation

#### a. Training Data

Follow the official repo, download training patches in HDF5 format from [GoogleDrive](https://drive.google.com/file/d/1wMtNGvliK_pUTogfzMyrz57iDb_jSQR8/view?usp=sharing) and put it into `./datas/`.

#### b. Testing Data

Testing mesh files can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1R21MD1O6q8E7ANui8FR0MaABkKc30PG4/view?usp=sharing) and unzip it into `./datas/test_data/`.

### 3. Train

Run the following commands for training.

```shell
mkdir logs
python train --gpu 0 --log_dir punet_baseline
```

### 4. Testing

Run the following commands for testing.

```shell
mkdir outputs
python test.py --gpu 0 --resume logs/punet_baseline/punet_epoch_99.pth
```

