#!/bin/bash
expr_name="exp1"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 49995 train.py \
--dataset_path "specified_path" \
--batch_size 2 \
--exp_name=$expr_name \