#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf 
import numpy as np 
import os
import cv2 
from PIL import Image

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def ConvertToRecord(data_path, record_name):
    (root, classes, files) = next(os.walk(data_path))
    
    writer = tf.python_io.TFRecordWriter(record_name + '.tfrecords')

    for index, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            image = Image.open(image_path)
            image = image.resize((256, 256))       
            image_raw = image.tobytes()

            print(image_name)

            example = tf.train.Example(features = tf.train.Features(feature = {
                'image_label': _int64_feature(index),
                'image_raw': _bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())
        
    writer.close()


if __name__ == '__main__':
    data_path = r'~/Work/Database/sketchy/256x256/photo/tx_000100000000'
    record_name = 'photo'

    ConvertToRecord(data_path, record_name)