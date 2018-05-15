import tensorflow as tf
from random import shuffle
import scipy.misc as smi
import numpy as np
import itertools
import threading
import pickle
import os

def get_dict():
    with open('Data/labels.txt', 'r', encoding='utf-8') as f:
        dic = {}
        dataset = f.read()
        sample_pairs = dataset.split('\r\n')
        max_len = 0
        for pair in sample_pairs:
            image_idx, label = pair.split('    ')
            if len(label) > max_len:
                max_len = len(label)
            for ch in label:
                if not dic.has_key(ch):
                    dic.update({ch: len(dic)})
        return max_len, dic

def _load_data(filedir = 'Data'):
    with open('Data/labels.txt', 'r') as f:
        dataset = f.read()
        sample_pairs = dataset.split('\n')
    for epoch in itertools.count():
        shuffle(sample_pairs)
        for pair in sample_pairs:
            image_idx, label = pair.split('    ')
            impath = os.path.join(filedir, image_idx + '.jpg')
            image = smi.imread(impath, mode='L')
            yield image, label

def _preprocessing(hypes):
    def encode_chars(dic, string, maxlen):
        class_length = len(dic) + 1 #add one more catergory which indicates "no char found"
        strlen = len(string)
        lst = None
        for ch in range(maxlen):
            if ch < strlen:
                num = dic[string[ch]] if dic.has_key(string[ch]) else class_length - 1
            else:
                num = class_length - 1
            curr = np.array([True if i == num else False for i in range(class_length)])
            if lst is None:
                lst = curr
            else:
                np.vstack((lst, curr))

    def process_image(img):
        #substract mean
        image = img - np.mean(img)
        #resize
        return smi.imresize(image, (40, 200))
    with open('','r') as f:
        dic = pickle.load(f)
    maxlen = hypes['arch']['maxlen']
    for pair in _load_data():
        image, label = pair
        enc_label = encode_chars(label, dic, maxlen)
        proc_img = process_image(image)
        yield proc_img, enc_label

def create_queue(hypes):
    height = hypes['']
    width = hypes['']
    shape = [[height, width, 1], [4000]]
    q = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=shape)
    return q

def start_enqueue_thread(hypes,q, sess):
    image_pl = tf.placeholder(tf.float32)
    label_pl = tf.placeholder(tf.int32)

    enqueue_op = q.enqueue(image_pl, label_pl)
    def enqueue_loop(sess, gen):
        for d in gen:
            sess.run(enqueue_op, feed_dict={image_pl:d[0], label_pl:d[1]})
    gen = _preprocessing(hypes)
    thread = threading.Thread(target = enqueue_loop, args = (sess, gen))
    thread.daemon = True
    thread.start()

def input(hypes, q, phase):
    if phase == 'valid':
        image, label = q.dequeue()
        image = tf.expand_dims(image, 0)
        label = tf.expand_dims(label, 0)
        return image, label
    elif phase == 'train':
        image, label = q.dqueue_many(hypes['solve']['batch_size'],)
        # Display the training images in the visualizer.
        tensor_name = image.op.name
        tf.summary.image(tensor_name + '/image', image)
    else:
        raise ValueError('phase mistaken assigned')

    return image, label

if __name__ == '__main__':
    a,d = get_dict()
    with open('Data/dictionary.dat', 'wb') as f:
        pickle.dump(d, f)
    print(a)