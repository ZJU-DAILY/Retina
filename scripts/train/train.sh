#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=6

# accelerate launch ./scripts/train/train_colbert.py scripts/configs/qwen2/train_triple_sparse_qwen2_model.yaml

accelerate launch ./scripts/train/train_colbert.py scripts/configs/qwen2/train_point_reranker_qwen2_model.yaml