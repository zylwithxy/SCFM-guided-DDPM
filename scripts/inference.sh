#!/bin/bash
expr_name="exp1"
valimg=8570
save_dir='./infer_results'

# Inference
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 49995 inference.py \
--dataset_path "/media/user/Data/XUEYu/Datasets/deepfashion_pidm" \
--expr_name=$expr_name \
--valimg_num=$valimg \
--val_savedir=$save_dir \
--use_warp \