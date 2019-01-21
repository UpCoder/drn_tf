import tensorflow as tf
from tensorflow.contrib import slim
import config


class BasicBlock:
    expansion = 1

    @staticmethod
    def forward(input_tensor, output_dim, stride=1, down_sample=None, dilation=(1, 1), residual=True, arg_scope=None,
                scope_name=None):
        input_shape = input_tensor.get_shape().as_list()
        input_dim = input_shape[-1]
        with tf.variable_scope(scope_name):
            with slim.arg_scope(arg_scope):
                conv1 = slim.conv2d(input_tensor, output_dim, kernel_size=[3, 3], stride=stride, padding='SAME',
                                    rate=dilation[0], biases_initializer=None, scope='conv1')
                bn1 = slim.batch_norm(conv1, scope='bn1')
                relu1 = tf.nn.relu(bn1)

                conv2 = slim.conv2d(relu1, output_dim, [3, 3], padding='SAME', rate=dilation[1], scope='conv2',
                                    biases_initializer=None)
                bn2 = slim.batch_norm(conv2, scope='bn2')
                if down_sample is not None:
                    residual_path = down_sample(input_tensor, output_dim=output_dim, expansion=BasicBlock.expansion,
                                                stride=stride, arg_scope=arg_scope, scope_name='down_sample')
                if residual:
                    if down_sample is not None:
                        bn2 += residual_path
                    else:
                        bn2 += input_tensor
                out = tf.nn.relu(bn2)
                return out


class Bottleneck:
    expansion = 4

    @staticmethod
    def forward(input_tensor, output_dim, stride=1, down_sample=None, dilation=(1, 1), residual=True, arg_scope=None,
                scope_name=None):
        input_shape = input_tensor.get_shape().as_list()
        input_dim = input_shape[-1]
        with tf.variable_scope(scope_name):
            with slim.arg_scope(arg_scope):
                conv1 = slim.conv2d(input_tensor, output_dim, kernel_size=1, biases_initializer=None, scope='conv1')
                bn1 = slim.batch_norm(conv1, scope='bn1')
                bn1 = tf.nn.relu(bn1)

                conv2 = slim.conv2d(bn1, output_dim, kernel_size=3, stride=stride, padding='SAME',
                                    biases_initializer=None, rate=dilation[1], scope='conv2')
                bn2 = slim.batch_norm(conv2, scope='bn2')
                bn2 = tf.nn.relu(bn2)

                conv3 = slim.conv2d(bn2, output_dim * Bottleneck.expansion, kernel_size=1, biases_initializer=None,
                                    scope='conv3')
                bn3 = slim.batch_norm(conv3, scope='bn3')
                if down_sample is not None:
                    residual = down_sample(input_tensor)
                out = bn3 + residual
                out = tf.nn.relu(out)
                return out


class DRN:
    def __init__(self, input_tensor, block, layers, num_classes=19, channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False, out_middle=False, pool_size=28, arch='D', fc_flag=False):
        self.input_tensor = input_tensor
        self.block = block
        self.layers = layers
        self.num_classes = num_classes
        self.channels = channels
        self.out_middle = out_middle
        self.out_map = out_map
        self.pool_size = pool_size
        self.arch = arch
        self.arg_scope = None
        self.fc_flag = fc_flag
        self._build_network()
        self._build_up()

    @staticmethod
    def _down_sample_func(input_tensor, output_dim, expansion, stride, arg_scope, scope_name):
        with tf.variable_scope(scope_name):
            with slim.arg_scope(arg_scope):
                conv1 = slim.conv2d(input_tensor, output_dim * expansion, kernel_size=1, stride=stride,
                                    biases_initializer=None, scope='conv1')
                bn1 = slim.batch_norm(conv1, scope='bn1')
            return bn1

    def _make_layer(self, input_tensor, block, output_dim, blocks, stride=1, dilation=1, new_level=True,
                    residual=True, arg_scope=None, scope_name=None):
        assert dilation == 1 or dilation % 2 == 0
        down_sample = None
        input_shape = input_tensor.get_shape().as_list()
        input_dim = input_shape[-1]
        with tf.variable_scope(scope_name):
            if stride != 1 or input_dim != output_dim * block.expansion:
                down_sample = self._down_sample_func
            output_tensor = block.forward(input_tensor, output_dim, stride=stride, down_sample=down_sample,
                                          dilation=(1, 1) if dilation == 1 else (
                                              dilation // 2 if new_level else dilation, dilation), residual=residual,
                                          arg_scope=arg_scope, scope_name='Block0')
            for i in range(1, blocks):
                output_tensor = block.forward(output_tensor, output_dim, residual=residual,
                                              dilation=(dilation, dilation),
                                              arg_scope=arg_scope, scope_name='Block' + str(i))
            return output_tensor

    def _make_conv_layer(self, input_tensor, output_dim, convs, stride=1, dilation=1, scope_name=None, arg_scope=None):
        with tf.variable_scope(scope_name):
            with slim.arg_scope(arg_scope):
                output = input_tensor
                for i in range(convs):
                    output = slim.conv2d(output, output_dim, kernel_size=3, stride=stride if i == 0 else 1,
                                         padding='SAME', biases_initializer=None, rate=dilation)
                    output = slim.batch_norm(output)
                    output = tf.nn.relu(output)
                return output

    def _build_network(self):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=None,
                            padding='SAME',
                            weights_regularizer=slim.l2_regularizer(config.weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=None):
            with slim.arg_scope([slim.batch_norm], is_training=True, scale=True) as arg_sc:
                self.arg_scope = arg_sc
                if self.arch == 'C':
                    self.conv1 = slim.conv2d(self.input_tensor, self.channels[0], kernel_size=7, stride=1,
                                             scope='conv1')
                    self.bn1 = slim.batch_norm(self.conv1, scope='bn1')
                    self.relu1 = tf.nn.relu(self.bn1)

                    self.layer1 = self._make_layer(self.relu1, BasicBlock, self.channels[0], self.layers[0], stride=1,
                                                   scope_name='layer1')
                    self.layer2 = self._make_layer(self.layer1, BasicBlock, self.channels[1], self.layers[1], stride=2,
                                                   scope_name='layer2')
                elif self.arch == 'D':
                    with tf.variable_scope('layer0'):
                        print(self.input_tensor, self.channels[0])
                        self.conv1 = slim.conv2d(self.input_tensor, self.channels[0], kernel_size=7, stride=1,
                                                 padding='SAME', biases_initializer=None, scope='conv1')
                        self.bn1 = slim.batch_norm(self.conv1, scope='bn1')
                        self.relu1 = tf.nn.relu(self.bn1)

                    self.layer1 = self._make_conv_layer(self.relu1, output_dim=self.channels[0],
                                                        convs=self.layers[0], stride=1, scope_name='layer1',
                                                        arg_scope=arg_sc)
                    self.layer2 = self._make_conv_layer(self.layer1, output_dim=self.channels[1],
                                                        convs=self.layers[1], stride=2, scope_name='layer2',
                                                        arg_scope=arg_sc)
                self.layer3 = self._make_layer(self.layer2, self.block, self.channels[2], self.layers[2], stride=2,
                                               arg_scope=arg_sc, scope_name='layer3', residual=True)
                self.layer4 = self._make_layer(self.layer3, self.block, self.channels[3], self.layers[3], stride=2,
                                               arg_scope=arg_sc, scope_name='layer4')
                self.layer5 = self._make_layer(self.layer4, self.block, self.channels[4], self.layers[4],
                                               arg_scope=arg_sc, scope_name='layer5')

                self.layer6 = None if self.layers[5] == 0 else self._make_layer(self.layer5, self.block,
                                                                                self.channels[5],
                                                                                self.layers[5], dilation=4,
                                                                                new_level=False, arg_scope=arg_sc,
                                                                                scope_name='layer6')
                if self.arch == 'C':
                    self.layer7 = None if self.layers[6] == 0 else self._make_layer(self.layer6, BasicBlock,
                                                                                    self.channels[6], self.layers[6],
                                                                                    dilation=2, new_level=False,
                                                                                    residual=False, arg_scope=arg_sc,
                                                                                    scope_name='layer7')
                    self.layer8 = None if self.layers[7] == 0 else self._make_layer(self.layer7, BasicBlock,
                                                                                    self.channels[7], self.layers[7],
                                                                                    dilation=1, new_level=False,
                                                                                    residual=False, arg_scope=arg_sc,
                                                                                    scope_name='layer8')
                elif self.arch == 'D':
                    self.layer7 = None if self.layers[6] == 0 else self._make_conv_layer(self.layer6, self.channels[6],
                                                                                         self.layers[6], dilation=2,
                                                                                         arg_scope=arg_sc,
                                                                                         scope_name='layer7')
                    self.layer8 = None if self.layers[7] == 0 else self._make_conv_layer(self.layer7, self.channels[7],
                                                                                         self.layers[7], dilation=1,
                                                                                         arg_scope=arg_sc,
                                                                                         scope_name='layer8')
                self.final_feature_map = self.layer8 if self.layer8 is not None else self.layer7 \
                    if self.layer7 is not None else self.layer6 if self.layer6 is not None else self.layer5
                if self.fc_flag and self.num_classes > 0:
                    self.avg_pool = tf.reduce_mean(self.final_feature_map, reduction_indices=[1, 2], keep_dims=True)
                    self.fc = slim.conv2d(self.avg_pool, num_outputs=self.num_classes, kernel_size=1, stride=1,
                                          biases_initializer=tf.zeros_initializer(), scope='fc')

    def _build_up(self):
        with tf.variable_scope('up_brach'):
            with slim.arg_scope(self.arg_scope):
                seg_logits = slim.conv2d(self.final_feature_map, self.num_classes, kernel_size=1,
                                         biases_initializer=tf.zeros_initializer())
                shape = seg_logits.get_shape().as_list()
                stride = 8
                self.seg_logits = tf.image.resize_images(seg_logits, [shape[1] * stride, shape[2] * stride])

    def build_loss(self, gt_mask, do_summary=False):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=self.seg_logits, labels=gt_mask,
                                                               loss_collection=None)
        cross_entropy = tf.reduce_mean(cross_entropy)
        tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy)
        if do_summary:
            tf.summary.scalar('loss', cross_entropy)
        print('build_mask')


def build_drn_d_22(input_tensor, **kwargs):
    model = DRN(input_tensor, BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', **kwargs)
    return model


def get_model(model_name):
    if model_name == 'drn_d_22':
        return build_drn_d_22


if __name__ == '__main__':
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 512, 512, 3], name='input')
    drn_d_22 = get_model('drn_d_22')
    model_obj = drn_d_22(input_tensor, num_classes=1000)
    print(model_obj.fc)
    for tensor in tf.model_variables():
        print(tensor)
    print('seg_logits is ', model_obj.seg_logits)
    model_obj.build_loss(tf.placeholder(tf.int32, shape=[None, 512, 512, 1]))



