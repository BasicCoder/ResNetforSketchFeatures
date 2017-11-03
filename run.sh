#!/usr/bin/env bash
python resnet_main.py --train_data_path=./photo.tfrecords \
                      --log_root=./log \
                      --train_dir=./train \
                      --dataset='sketchy' \
                      --num_gpus=1