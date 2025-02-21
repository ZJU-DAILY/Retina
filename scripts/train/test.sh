#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=7

# python  scripts/train/train_colbert.py --config ./scripts/configs/qwen2/test/test_beir_sparse_qwen2_model.yaml
accelerate launch ./scripts/train/train_colbert.py scripts/configs/qwen2/test_beir_sparse_qwen2_model.yaml