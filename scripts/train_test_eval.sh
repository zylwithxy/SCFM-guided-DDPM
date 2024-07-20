#!/bin/bash
expr_name="warp_32fsep_16fsep_8fsep_wo|tarenc|qkvattn_lambregular_0.0025_bran_warplooseweig_affinedict33|42|52_skipblocks256|100_128|100_64|100_cct256|128|64_flowdisp"

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 49995 train.py \
--dataset_path "/media/user/Data/XUEYu/Datasets/deepfashion_pidm" \
--batch_size 2 \
--exp_name=$expr_name \
# --resume_train \
# --start_epoches 300 \
# --end_epoches 400 \

valimg=8570
save_dir='./infer_results'

# Inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 49995 inference.py \
--dataset_path "/media/user/Data/XUEYu/Datasets/deepfashion_pidm" \
--expr_name=$expr_name \
--valimg_num=$valimg \
--val_savedir=$save_dir \
--use_warp \

# Eval
gt_path="/media/user/Data/XUEYu/Datasets/deepfashion_pidm/pidm_256_256_bicubicresize_png/test"
fid_real_path="/media/user/Data/XUEYu/Datasets/deepfashion_pidm/pidm_256_256_bicubicresize_png/train"
# distorated_path="/media/user/Toshiba4T/XUEYu/codes/pose_transfer_baseline/PIDM/pretrained_results/PIDM" 
CUDA_VISIBLE_DEVICES=0 python -m utils.metrics \
--gt_path=$gt_path \
--distorated_path="$save_dir/$expr_name" \
--fid_real_path=$fid_real_path \
--name="./infer_results_metric"