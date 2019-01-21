import tensorflow as tf
from tensorflow.contrib import slim
import config
from config import print_config
from datasets.cityscapes.read_tfrecords import read_tfrecords_dataset, create_dataset_batch_queue
from drn import get_model
import util
from tensorflow.python.ops import control_flow_ops
import os
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_config['ids']
tf.app.flags.DEFINE_string('dataset_dir', '/media/dl-box/HDD3/ld/Documents/datasets/CITYSCAPES/train_tfrecords', '')
tf.app.flags.DEFINE_float('lr', 0.001, '')
tf.app.flags.DEFINE_float('momentum', 0.9, '')
tf.app.flags.DEFINE_string('log_dir', './logs', '')
tf.app.flags.DEFINE_integer('log_every_n_steps', 1, '')
tf.app.flags.DEFINE_integer('max_number_of_steps', 10000000, '')
tf.app.flags.DEFINE_float('gpu_memory_fraction', -1, '')
tf.app.flags.DEFINE_boolean('using_moving_average', True, '')
tf.app.flags.DEFINE_float('moving_average_decay', '0.9999', '')
FLAGS = tf.app.flags.FLAGS


def sum_gradients(clone_grads):
    averaged_grads = []
    print(clone_grads)
    for idx, grad_and_vars in enumerate(zip(*clone_grads)):
        grads = []
        var = grad_and_vars[0][1]
        try:
            for g, v in grad_and_vars:
                print(v)
                assert v == var
                grads.append(g)
            grad = tf.add_n(grads, name=v.op.name+'_summed_gradients')
        except:
            import pdb
            pdb.set_trace()
        averaged_grads.append((grad, v))
    return averaged_grads


def build_network(batch_queue):
    with tf.device('/cpu:0'):
        global_step = slim.create_global_step()
        learning_rate = tf.constant(FLAGS.lr, name='learning_rate')
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=FLAGS.momentum,
                                               name='Momentum')
        tf.summary.scalar('learning_rate', learning_rate)
    total_loss = 0
    num_clones = len(config.gpu_config['ids'].split(','))
    gradients = []
    for clone_idx, gpu in enumerate(config.gpu_config['ids'].split(',')):
        do_summary = clone_idx == 0
        reuse = clone_idx > 0

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            with tf.name_scope('gpu_%d' % clone_idx) as clone_scope:
                with tf.device('/gpu:%d' % (int(gpu))) as clone_device:
                    b_image, b_mask, b_image_name, b_mask_name = batch_queue.dequeue()
                    print('b_image is ', b_image)
                    print('b_mask is ', b_mask)
                    drn_d_22 = get_model('drn_d_22')
                    model_obj = drn_d_22(b_image, num_classes=config.dataset_config['num_classes'])
                    model_obj.build_loss(b_mask, do_summary=do_summary)
                    losses = tf.get_collection(tf.GraphKeys.LOSSES, clone_scope)
                    cur_clone_loss = tf.add_n(losses) / num_clones
                    total_loss += cur_clone_loss
                    if clone_idx == 0:
                        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        cur_clone_loss += regularization_loss
                    clone_gradients = optimizer.compute_gradients(cur_clone_loss)
                    gradients.append(clone_gradients)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    averaged_gradients = sum_gradients(gradients)
    apply_grads_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step)
    train_ops = [apply_grads_op]
    bn_update_op = util.tf.get_update_op()
    if bn_update_op is not None:
        train_ops.append(bn_update_op)
    if FLAGS.using_moving_average:
        tf.logging.info('using moving average in training, with decay=%f' % FLAGS.moving_average_decay)
        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([apply_grads_op]):
            train_ops.append(ema_op)
    train_ops = control_flow_ops.with_dependencies(train_ops, total_loss, name='train_op')
    return train_ops


def train(train_op):
    summary_op = tf.summary.merge_all()
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
    init_fn = None
    saver = tf.train.Saver(max_to_keep=100, write_version=2)
    slim.learning.train(
        train_op,
        logdir=FLAGS.log_dir,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=30,
        saver=saver,
        save_interval_secs=2400,
        session_config=sess_config
    )


def main(_):
    dataset = read_tfrecords_dataset(FLAGS.dataset_dir, 'cityscapes', 19)
    print_config(dataset)
    batch_queue = create_dataset_batch_queue(dataset)
    train_op = build_network(batch_queue)
    train(train_op)


if __name__ == '__main__':
    tf.app.run()