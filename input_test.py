#!/usr/bin/env python
# -*- coding:utf-8 -*-
import six 
import sys

import sketchy_input
import numpy as np 
import resnet_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10, cifar100 or sketchy.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '', 'Filepattern for training data')
tf.app.flags.DEFINE_string('eval_data_path', '', 'Filepattern for eval data.')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '', 'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '', 'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50, 'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False, 'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '', 'Directory to keep the checkpoints. Should be a parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used for training. (0 or 1)')



def train(hps):
    """Training loop."""
    # images, labels = cifar_input.build_input(FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode)
    images, labels = sketchy_input.build_input(FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode)
    global_step = tf.train.get_or_create_global_step()

    truth = tf.argmax(labels, axis=1)
    summary_hook = tf.train.SummarySaverHook(
        save_steps = 100,
        output_dir = FLAGS.train_dir,
        summary_op = tf.summary.merge([tf.summary.scalar('labels', labels)])
    )

    logging_hook = tf.train.LoggingTensorHook(
        tensors = {
            'step': global_step,
            'labels': labels,
            'truth': truth
        },
        every_n_iter=100
    )

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir = FLAGS.log_root,
        hooks = [logging_hook, tf.train.StopAtStepHook(last_step=500)],
        save_summaries_steps = 0,
        config = tf.ConfigProto(allow_soft_placement = True)) as mon_sess:
            for i in range(10): 
                example, l, t = mon_sess.run([images, labels, truth])
                print(t)
                '''
                np.set_printoptions(threshold='nan')
                print("label", l[0])
                print("example_length:", example[0][0].shape)
                print("example", example[0][223])
                '''

def main(_):

    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu')

    if FLAGS.mode == 'train':
        batch_size = 32
    elif FLAGS.mode == 'eval':
        batch_size = 100

    if FLAGS.dataset == 'cifar10':
        num_classes = 10
    elif FLAGS.dataset == 'cifar100':
        num_classes = 100
    elif FLAGS.dataset == 'sketchy':
        num_classes = 125


    hps = resnet_model.HParams(batch_size = batch_size,
                               num_classes = num_classes,
                               num_layers=34,
                               min_lrn_rate=0.0001,
                               lrn_rate=0.1,
                               num_residual_units=5,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='mom')
    # with tf.device(dev):
    if FLAGS.mode == 'train':
        train(hps)
    elif FLAGS.mode == 'eval':
        pass
        # evaluate(hps)

if __name__ == '__main__':
    tf.app.run()


