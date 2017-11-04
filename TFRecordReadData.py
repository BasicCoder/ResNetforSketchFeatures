#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf 
import numpy as np 
import os 
from PIL import Image
import sketchy_input 


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'image_label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    })

    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    #img = tf.cast(img, tf.float32)# * (1. / 255) - 0.5
    label = tf.cast(features['image_label'], tf.int32)
    return img, label

if __name__ == '__main__':
    init = tf.global_variables_initializer()

    # image, label = read_and_decode("photo_test.tfrecords")
    images, labels = sketchy_input.build_input("sketchy", "./photo_test.tfrecords", 32, "train")
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(1):
            example, l = sess.run([images, labels])
            #print(key)
            # img = Image.fromarray(example, 'RGB')
            # img.save("./log/" + str(i) + "_label_" + str(l) + '.jpg')
            np.set_printoptions(threshold='nan')
            print("label", l)
            # print("example", example)

        coord.request_stop()
        coord.join(threads)