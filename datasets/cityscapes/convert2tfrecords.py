# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import slim
import os
import random
import sys
import cv2
import numpy as np
from datasets.dataset_utils import bytes_feature, np_array_feature, int64_feature
SAMPLES_PER_FILES = 200
RANDOM_SEED = 7777


def read_imgs_labels_paths(image_dir, stage='train'):
    '''
    返回imgs和labels path的list
    :param image_dir:
    :return:
    '''
    imgs_txt_path = os.path.join(image_dir, stage + '_images.txt')
    with open(imgs_txt_path) as txt_file:
        imgs_lines = txt_file.readlines()

    labels_txt_path = os.path.join(image_dir, stage + '_labels.txt')
    with open(labels_txt_path) as txt_file:
        labels_lines = txt_file.readlines()
    return [os.path.join(image_dir, line.replace('\n', '')) for line in imgs_lines], \
           [os.path.join(image_dir, line.replace('\n', '')) for line in labels_lines]


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def _process_image(image_path, mask_path):
    img_data = tf.gfile.FastGFile(image_path, 'rb').read()
    mask_data = tf.gfile.FastGFile(mask_path, 'rb').read()
    # img_data = cv2.imread(image_path)
    # mask_data = cv2.imread(mask_path, 0)
    return img_data, mask_data


def _add_to_tfrecord(image_path, mask_path, tfrecord_writer):
    img_data, mask_data = _process_image(image_path, mask_path)
    shape = np.shape(img_data)[:2]
    image_format = b'PNG'
    # print([*shape, 3], np.unique(mask_data))
    example = tf.train.Example(features=tf.train.Features(feature={
        'format': bytes_feature(image_format),
        'image': bytes_feature(img_data),
        'mask': bytes_feature(mask_data),
        'image_name': bytes_feature(str.encode(os.path.basename(image_path))),
        'mask_name': bytes_feature(str.encode(os.path.basename(mask_path)))
        # 'image': np_array_feature(img_data),
        # 'mask': np_array_feature(mask_data),
        # 'image/shape': int64_feature([*shape, 3]),
        # 'mask/shape': int64_feature([*shape, 1])
    }))
    tfrecord_writer.write(example.SerializeToString())


def convert2tfrecords(image_paths, mask_paths, save_dir, dataset_name, shuffle=True):
    if shuffle:
        random.Random(RANDOM_SEED).shuffle(image_paths)
        random.Random(RANDOM_SEED).shuffle(mask_paths)
    i = 0
    tf_idx = 0
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # image_paths = image_paths[:10]
    # mask_paths = mask_paths[:10]
    while i < len(image_paths):
        tf_filename = _get_output_filename(save_dir, dataset_name, tf_idx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(image_paths) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(image_paths)))
                sys.stdout.flush()
                _add_to_tfrecord(image_paths[i], mask_paths[i], tfrecord_writer)
                i += 1
                j += 1
            tf_idx += 1


if __name__ == '__main__':
    image_paths, mask_paths = read_imgs_labels_paths(
        '/media/dl-box/HDD3/ld/Documents/datasets/CITYSCAPES',
        'train'
    )
    save_dir = '/media/dl-box/HDD3/ld/Documents/datasets/CITYSCAPES/train_tfrecords'
    convert2tfrecords(image_paths, mask_paths, save_dir, 'cityscapes')