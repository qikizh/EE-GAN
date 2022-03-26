# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import imageio
from PIL import Image

from inception.slim import slim
import numpy as np
import tensorflow as tf
import argparse

import math
import os.path
import scipy.misc
# import time
# import scipy.io as sio
# from datetime import datetime
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', '../data/Models/inception_finetuned_models/flowers_valid299/model.ckpt',
                           """Path where to read model checkpoints.""")
tf.app.flags.DEFINE_string('image_folder', '../../data/SSA_OUT/12_2/base/Gen_Image/repeat_times_0',
                           """Path where to load the images """)
tf.app.flags.DEFINE_string('h5_file', '', """Path where to load the images """)
tf.app.flags.DEFINE_integer('num_classes', 20,      # 20 for flowers
                            """Number of classes """)
tf.app.flags.DEFINE_integer('splits', 10,
                            """Number of splits """)
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_string('gpu', '0', "The ID of GPU to use")

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997
MOVING_AVERAGE_DECAY = 0.9999
max_nums = 30000
# fullpath = FLAGS.image_folder
# print(fullpath)


def preprocess(img):
    # print('img', img.shape, img.max(), img.min())
    # img = Image.fromarray(img, 'RGB')
    if len(img.shape) == 2:
        img = np.resize(img, (img.shape[0], img.shape[1], 3))
    img = scipy.misc.imresize(img, (299, 299, 3), interp='bilinear')
    img = img.astype(np.float32)
    # [0, 255] --> [0, 2] --> [-1, 1]
    img = img / 127.5 - 1.
    # print('img', img.shape, img.max(), img.min())
    return np.expand_dims(img, 0)

def get_inception_score(sess, images, pred_op):
    splits = FLAGS.splits
    # assert(type(images) == list)
    print(type(images[0]))
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    bs = FLAGS.batch_size
    preds = []
    num_examples = len(images)
    n_batches = int(math.floor(float(num_examples) / float(bs)))
    indices = list(np.arange(num_examples))
    np.random.shuffle(indices)
    for i in range(n_batches):
        inp = []
        # print('i*bs', i*bs)
        for j in range(bs):
            if (i*bs + j) == num_examples:
                break
            img = images[indices[i*bs + j]]
            # print('*****', img.shape)
            img = preprocess(img)
            inp.append(img)
        # print("%d of %d batches" % (i, n_batches))
        # inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        #  print('inp', inp.shape)
        pred = sess.run(pred_op, {'inputs:0': inp})
        preds.append(pred)
        # if i % 100 == 0:
        #     print('Batch ', i)
        #     print('inp', inp.shape, inp.max(), inp.min())
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        istart = i * preds.shape[0] // splits
        iend = (i + 1) * preds.shape[0] // splits
        part = preds[istart:iend, :]
        kl = (part * (np.log(part) -
              np.log(np.expand_dims(np.mean(part, 0), 0))))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    print('mean:', "%.2f" % np.mean(scores), 'std:', "%.2f" % np.std(scores))
    return np.mean(scores), np.std(scores)

def load_data(image_path):
    images = []
    for path, subdirs, files in os.walk(image_path):
        for name in files:
            if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                filename = os.path.join(path, name)
                # print('filename', filename)
                # print('path', path, '\nname', name)
                # print('filename', filename)
                if os.path.isfile(filename):
                    img = scipy.misc.imread(filename)
                    # img = imageio.imread(filename)
                    # img = np.asarray(img)
                    images.append(img)
    if len(images) > max_nums:
        images = images[:max_nums]
    print('images', len(images), images[0].shape)
    return images

def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
    """Build Inception v3 model architecture.
    See here for reference: http://arxiv.org/abs/1512.00567
    Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.
    Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
    """
    # Parameters for BatchNorm.
    batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
    }
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
        with slim.arg_scope([slim.ops.conv2d],
                            stddev=0.1,
                            activation=tf.nn.relu,
                            batch_norm_params=batch_norm_params):
            logits, endpoints = slim.inception.inception_v3(
              images,
              dropout_keep_prob=0.8,
              num_classes=num_classes,
              is_training=for_training,
              restore_logits=restore_logits,
              scope=scope)

    # Grab the logits associated with the side head. Employed during training.
    auxiliary_logits = endpoints['aux_logits']

    return logits, auxiliary_logits

# select_epoch = [300, 320, 340, 360, 380,
#                 400, 420, 440, 460, 480,
#                 500, 510, 520, 530, 540, 550, 560, 570, 580, 590,
#                 600, 610, 620, 630, 640, 650, 660, 670, 680, 690,
#                 700]
# select_epoch = [620]

#st, ed, interval = 500, 700, 10
# select_epoch = [epoch for epoch in range(st, ed+interval, interval)]
select_epoch = [580, 650, 700]
repeat_times = 5
image_folder = FLAGS.image_folder

#select_epoch = [510, 540, 600]
# select_epoch = [590]

def main(unused_argv=None):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # with tf.device("/gpu:%d" % FLAGS.gpu):
            with tf.device("/gpu:0"):
                # Number of classes in the Dataset label set plus 1.
                # Label 0 is reserved for an (unused) background class.
                num_classes = FLAGS.num_classes + 1

                # Build a Graph that computes the logits predictions from the
                # inference model.
                inputs = tf.placeholder( tf.float32, [FLAGS.batch_size, 299, 299, 3], name='inputs')
                # print(inputs)

                logits, _ = inference(inputs, num_classes)
                # calculate softmax after remove 0 which reserve for BG
                known_logits = \
                    tf.slice(logits, [0, 1],
                             [FLAGS.batch_size, num_classes - 1])
                pred_op = tf.nn.softmax(known_logits)

                # Restore the moving average version of the
                # learned variables for eval.
                variable_averages = \
                    tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
                variables_to_restore = variable_averages.variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)
                saver.restore(sess, FLAGS.checkpoint_dir)

                print('Restore the model from %s).' % FLAGS.checkpoint_dir)
                # the following part could be reconstructed
                mean_results = []
                std_results = []
                all_ts = 0

                for idx, epoch in enumerate(select_epoch):
                    epoch_mean_results = []
                    epoch_std_results = []

                    for re_ix in range(repeat_times):
                        start_ts = time.time()
                        cur_image_dir_path = os.path.join(image_folder, "Epoch_%d_%d" % (epoch, re_ix))
                        print(cur_image_dir_path)
                        images = load_data(cur_image_dir_path)
                        mean_score, std_score = get_inception_score(sess, images, pred_op)
                        epoch_mean_results.append(mean_score)
                        epoch_std_results.append(std_score)
                        end_ts = time.time()
                        print('''Epoch_%d_%d model is finished and costs time: %.2f\n''' % (
                        epoch, re_ix, end_ts - start_ts))
                        all_ts += (end_ts - start_ts)

                    mean_results.append(epoch_mean_results)
                    std_results.append(epoch_std_results)

                print(mean_results)
                print(std_results)

if __name__ == '__main__':
    tf.app.run()