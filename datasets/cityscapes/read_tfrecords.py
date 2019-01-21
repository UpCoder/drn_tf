import tensorflow as tf
from tensorflow.contrib import slim
import os
import config
from glob import glob
import numpy as np
import cv2
num_samples = {
    'train': 2975
}


def read_tfrecords_dataset(dataset_dir, dataset_name, num_classes):
    file_pattern = '%s*.tfrecord' % dataset_name
    file_pattern = os.path.join(dataset_dir, file_pattern)
    reader = tf.TFRecordReader
    keys_to_feature = {
        'format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image': tf.FixedLenFeature((), tf.string, default_value=''),
        'mask': tf.FixedLenFeature((), tf.string, default_value=''),
        'image_name': tf.FixedLenFeature((), tf.string, default_value=''),
        'mask_name': tf.FixedLenFeature((), tf.string, default_value=''),
        # 'image': tf.VarLenFeature(tf.float32),
        # 'image/shape': tf.FixedLenFeature([3], tf.int64),
        # 'mask': tf.VarLenFeature(tf.float32),
        # 'mask/shape': tf.FixedLenFeature([3], tf.int64)

    }
    items_to_handlers = {
        # 'image': slim.tfexample_decoder.Tensor('image', 'image/shape'),
        # 'mask': slim.tfexample_decoder.Tensor('mask', 'image/shape')
        'image': slim.tfexample_decoder.Image('image', 'format', shape=None, channels=3),
        'mask': slim.tfexample_decoder.Image('mask', 'format', shape=None, channels=1),
        'image_name': slim.tfexample_decoder.Tensor('image_name'),
        'mask_name': slim.tfexample_decoder.Tensor('mask_name')
    }
    items_to_descriptions = {
        'image': 'the input image',
        'mask': 'the ground truth of mask label',
        'image_name': 'the base name of image',
        'mask_name': 'the base name of mask'
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_feature, items_to_handlers)
    print('num sample is ', num_samples[os.path.basename(dataset_dir).split('_')[0]])
    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples[os.path.basename(dataset_dir).split('_')[0]],
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions
    )


def create_dataset_batch_queue(dataset):
    # with tf.device('/cpu:0'):
    with tf.name_scope(config.dataset_config['dataset_name'] + '_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=config.dataset_config['num_readers'],
            common_queue_capacity= 100 * config.dataset_config['batch_size'],
            common_queue_min= 50 * config.dataset_config['batch_size'],
            # shuffle=True
        )
    [image, mask, image_name, mask_name] = provider.get(['image', 'mask', 'image_name', 'mask_name'])
    image = tf.identity(image, 'image')
    mask = tf.identity(mask, 'mask')
    print('image is ', image)
    print('mask is ', mask)
    print('the image name is ', image_name)
    print('the mask name is ', mask_name)
    image.set_shape([*config.dataset_config['input_shape_original'], 3])
    mask.set_shape([*config.dataset_config['input_shape_original'], 1])
    print('image is ', image)
    print('mask is ', mask)
    with tf.name_scope(config.dataset_config['dataset_name'] + '_batch'):
        b_image, b_mask, b_image_name, b_mask_name = tf.train.batch([image, mask, image_name, mask_name],
                                                                    batch_size=config.dataset_config['batch_size'],
                                                                    num_threads=config.dataset_config['num_threads'],
                                                                    capacity=10)
        b_image = tf.cast(b_image, tf.float32)
        b_mask = tf.cast(b_mask, tf.int32)

    with tf.name_scope(config.dataset_config['dataset_name'] + '_prefetch_queue'):
        batch_queue = slim.prefetch_queue.prefetch_queue([b_image, b_mask, b_image_name, b_mask_name], capacity=5)
    return batch_queue


def test_read():
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_config['ids']
    dataset_dir = '/media/dl-box/HDD3/ld/Documents/datasets/CITYSCAPES/train_tfrecords'
    dataset = read_tfrecords_dataset(dataset_dir, 'cityscapes', 19)
    from config import print_config
    print_config(dataset)
    batch_queue = create_dataset_batch_queue(dataset)
    image_tensor, mask_tensor, image_name_tensor, mask_name_tensor = batch_queue.dequeue()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print("image tensor is ", image_tensor)
        for i in range(10):
            image_value, mask_value, image_name_value, mask_name_value = sess.run(
                [image_tensor, mask_tensor, image_name_tensor, mask_name_tensor])
            print(np.shape(image_value), np.shape(mask_value))
            cv2.imwrite('./%d_%s' % (i, image_name_value[0].decode('utf-8')), image_value[0])
            cv2.imwrite('./%d_%s' % (i, mask_name_value[0].decode('utf-8')), mask_value[0])

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # mask_path = '/media/dl-box/HDD3/ld/Documents/datasets/CITYSCAPES/gtFine/train/aachen/aachen_000171_000019_gtFine_trainIds.png'
    # mask = cv2.imread(mask_path, 0)
    # print(np.shape(mask))
    # print(np.unique(mask))
    test_read()