import tensorflow as tf
import numpy as np
class feature_extractorl(object):
    def __init__(self, vgg16_npy_path):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.wd = 5e-4
    def build(self, inputs, debug=False):
        bgr = tf.concat([inputs,inputs,inputs],3)
        self.conv1_1 = self._conv_layer(bgr, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)
        out = self.fc_layer(self.pool5, 4096, 'fc')
        return out

    def _max_pool(self, bottom, name, debug):
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            return relu

    def fc_layer(self, bottom, cells, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = tf.get_variable('w',[dim, cells],tf.float32,tf.initializers.truncated_normal())
            biases = tf.get_variable('bias',[cells],tf.float32, tf.initializers.zeros())

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape

        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        return var

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        return var

class classifier(object):
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims

    def build(self, inputs, stage = 'train'):
        if stage == 'train':
            lc1 = tf.nn.dropout(self.layer(inputs, self.dims,'fc1', True),keep_prob=0.5)
        else:
            lc1 = self.layer(inputs, self.dims,'fc1',True)
        if stage == 'train':
            lc2 = tf.nn.dropout(self.layer(lc1, self.dims,'fc2'), keep_prob=0.5)
        else:
            lc2 = self.layer(lc1, self.dims, 'fc2')
        return tf.nn.softmax(lc2)

    def layer(self, bottom, cells, name, activation = False):
        shape = bottom.get_shape()
        with tf.variable_scope(self.name + '/' + name):
            weights = tf.get_variable('w', [shape[-1], cells], tf.float32, tf.initializers.truncated_normal())
            biases = tf.get_variable('bias', [cells], tf.float32, tf.initializers.zeros())

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(bottom, weights), biases)
            if activation is False:
                return fc
            else:
                return tf.nn.relu(fc)