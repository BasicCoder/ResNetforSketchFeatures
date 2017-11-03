#!/usr/bin/env python
#-*- coding:utf-8 -*-

import tensorflow as tf

def build_input(dataset, data_path, batch_size, mode):
    """Build Sketchy image and labels.
    Args:
        dataset: sketchy or Others.
        data_path: Filename for data.
        batch_size: Input batch size.
        mode: Either 'train' or 'eval'.
    Returns:
        images: Batchs of images. [batch_size, image_size, image_size, 3]
        labels: Batchs of labels. [batch_size, num_classes]
    Raises:
        ValueError: when the specified dataset is not supported.
    """

    image_size = 256
    if dataset == 'sketchy':
        num_classes = 125

    else:
        raise ValueError('Not supported dataset %s', dataset)

    depth = 3

    data_files = tf.gfile.Glob(data_path)
    file_queue = tf.train.string_input_producer(data_files, shuffle=True)
    # Read examples from files in the filename queue.
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)

    #Convert these example to dense labels and processed images.
    features = tf.parse_single_example(value, features={
        'image_label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [image_size, image_size, depth])

    label = tf.cast(features['image_label'], tf.int32)
    image = tf.cast(image, tf.float32)

    if mode == 'train':
        image = tf.image.resize_image_with_crop_or_pad(image, image_size+4, image_size+4)
        image = tf.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)

        image = tf.image.per_image_standardization(image)

        example_queue = tf.RandomShuffleQueue(
            capacity = 16 * batch_size,
            min_after_dequeue = 8 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], []]
        )
        num_threads = 16

    else:
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.per_image_standardization(image)

        example_queue = tf.FIFOQueue(
            3 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], []]
        )
        num_threads = 1

    assert len(image.get_shape()) == 3
    assert len(example_queue.shapes[0]) == 3
    example_queue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue, [example_queue_op] * num_threads))

    # Read 'batch' labels + images from the example queue.
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(tf.concat(values = [indices, labels], axis = 1), [batch_size, num_classes], 1.0, 0.0)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes

    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    return images, labels