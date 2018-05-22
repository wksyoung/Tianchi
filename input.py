import tensorflow as tf
from random import shuffle
import scipy.misc as smi
import numpy as np
import itertools
import threading
import pickle
import os

def get_dict():
    with open('Data/labels.txt', 'r') as f:
        dic = {}
        dataset = f.read()
        sample_pairs = dataset.split('\n')
        max_len = 0
        for pair in sample_pairs[:-1]:
            image_idx, label = pair.split('    ')
            print(image_idx)
            if len(label) > max_len:
                max_len = len(label)
            for ch in label:
                if not ch in dic:
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
    def encode_chars(string, dic, maxlen):
        class_length = len(dic)#add one more catergory which indicates "no char found"
        strlen = len(string)
        lst = None
        for ch in range(maxlen):
            if ch < strlen:
                num = dic[string[ch]] if string[ch] in dic else 32
                #print(string[ch], list(dic.keys())[list(dic.values()).index(num)])
            else:
                num = 32
            curr = np.array([True if i == num else False for i in range(class_length)])
            if lst is None:
                lst = curr
            else:
                lst = np.vstack((lst, curr))
        return lst

    def process_image(img):
        #substract mean
        image = img - np.mean(img)
        if image.shape[0] > image.shape[1]:
            np.rot90(image)
        #resize
        image = smi.imresize(image, (40, 200))
        return np.expand_dims(image, 2)
    with open('dictionary.dat','rb') as f:
        dic = pickle.load(f)
    maxlen = hypes['arch']['maxstr']
    data_path = os.path.join(os.getcwd(), hypes['dirs']['data_dir'])
    for pair in _load_data(data_path):
        image, label = pair
        enc_label = encode_chars(label, dic, maxlen)
        proc_img = process_image(image)
        yield proc_img, enc_label

def create_queue(hypes):
    height = hypes['height']
    width = hypes['width']
    s, cn = hypes['arch']['maxstr'], hypes['arch']['dictlen']
    shape = [[height, width, 1], [s,cn]]
    q = tf.FIFOQueue(capacity=20, dtypes=[tf.float32, tf.bool], shapes=shape)
    return q

def start_enqueue_thread(hypes,q, sess):
    image_pl = tf.placeholder(tf.float32)
    label_pl = tf.placeholder(tf.bool)

    enqueue_op = q.enqueue((image_pl, label_pl))
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
        image, label = q.dequeue_many(hypes['solver']['batch_size'],)
        # Display the training images in the visualizer.
        tensor_name = image.op.name
        tf.summary.image(tensor_name + '/image', image)
    else:
        raise ValueError('phase mistaken assigned')

    return image, label

import operator
if __name__ == '__main__':
    if not os.path.exists('dictionary.dat'):
        a,d = get_dict()
        with open('dictionary.dat', 'wb') as f:
            pickle.dump(d, f)
        print('Max string length:', a)
    import json
    import matplotlib.pyplot as plt
    with open('hyperparameters.json', 'r') as f:
        hypes = json.load(f)
    for pair in _preprocessing(hypes):
        plt.imshow(np.uint8(pair[0]))
        plt.show()


