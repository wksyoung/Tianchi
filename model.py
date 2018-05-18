import tensorflow as tf

class feature_extractorl(object):
    def __init__(self):
        pass
    def build(self, inputs):
        conv1 = self.conv_layer(inputs, 64, 'conv1')
        pool1 = self.max_pool(conv1, 'pool1')
        conv3 = self.conv_layer(pool1, 128,'conv2')
        conv4 = self.conv_layer(conv3, 128,'conv3')
        pool2 = self.max_pool(conv4, 'pool2')
        conv5 = self.conv_layer(pool2, 256, 'conv4')
        conv6 = self.conv_layer(conv5, 256, 'conv5')
        pool3 = self.max_pool(conv6, 'pool3')
        conv7 = self.conv_layer(pool3, 512,'conv6')
        out = self.fc_layer(conv7, 4096, 'fc')
        return out

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, channel, name):
        shape = bottom.get_shape()
        with tf.variable_scope(name):
            filt = tf.get_variable('conv',[5, 5, shape[-1], channel], tf.float32, tf.initializers.truncated_normal())

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = tf.get_variable('bias',[channel],tf.float32, tf.initializers.zeros())
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
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

class classifier(object):
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims

    def build(self, inputs, stage = 'train'):
        if stage == 'train':
            lc1 = tf.nn.dropout(self.layer(inputs, 2*self.dims,'fc1', True),keep_prob=0.5)
        else:
            lc1 = self.layer(inputs, 2*self.dims,'fc1',True)
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