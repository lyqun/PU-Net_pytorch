#!/bin/bash
extra_tag=punet

result_dir=./outputs/$extra_tag/
gt_dir=./datas/test_data/test_mesh
pcd_list=$(ls $gt_dir)
echo $pcd_list
evaluate_src_path=./nuc_utils/build/evaluation

for pcd in $pcd_list
do
    pcd_name=${pcd%.*}
    echo $pcd_name
    pcd_src_file_name=$pcd_name'.xyz'
    src_pcd_path=$result_dir'/'$pcd_src_file_name
    gt_pcd_path=$gt_dir'/'$pcd
    $evaluate_src_path $gt_pcd_path $src_pcd_path
done