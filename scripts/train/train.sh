#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

accelerate launch ./scripts/train/train_colbert.py scripts/configs/qwen2/train_beir_sparse_qwen2_model.yaml