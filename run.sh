#!/usr/bin/env bash
python resnet_main.py --train_data_path=./cifar10/data_batch* \
                      --log_root=./log \
                      --train_dir=./train \
                      --dataset='cifar10' \
                      --num_gpus=1