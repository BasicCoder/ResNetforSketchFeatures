#!/usr/bin/env bash
python input_test.py --train_data_path=./photo_test.tfrecords \
                      --log_root=./log \
                      --train_dir=./train \
                      --dataset='sketchy' \
                      --num_gpus=0