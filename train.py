from inference import inference
from input import input, start_enqueue_thread, create_queue
import tensorflow as tf
import utils
import os
import time


def loss(hypes, logits, labels):
    """

    :param hypes: hyperparameters
    :param logits: NxCxl tensor where N = batch size C=max string length l is logits of a classifier
    :param labels: NxCxl tensor where N = batch size C=max string length
    :return: the scaler tensor expected to be promoted
    """
    #logits[i,j,label[i,j]]
    ms = hypes['arch']['maxstr']
    dl = hypes['arch']['dictlen']
    bl = []
    for batch in range(hypes['solver']['batch_size']):
        l = [tf.boolean_mask(logits[batch,ch,:],labels[batch,ch,:]) for ch in range(ms)]
        bl.append(l)
    masked = tf.convert_to_tensor(bl)
    log = tf.log(masked)
    return tf.reduce_mean(tf.reduce_sum(log, axis=1))

def build_train_graph(hypes, images, labels):
    pred = inference(hypes,images, phase = 'train')
    obj = loss(hypes, pred, labels)
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    lr = tf.placeholder(dtype=tf.float32)
    optimazer = tf.train.AdamOptimizer(lr)
    var_grads = tf.gradients(obj, vars)
    opt = optimazer.apply_gradients(zip(var_grads, vars))

    train_graph = {}
    train_graph['train_op'] = opt
    train_graph['obj'] = obj
    train_graph['learning_rate'] = lr
    return train_graph

def _print_training_status(hypes, step, loss_value, start_time, lr):

    info_str = utils.cfg.step_str

    # Prepare printing
    duration = (time.time() - start_time) / int(utils.cfg.step_show)
    examples_per_sec = hypes['solver']['batch_size'] / duration
    sec_per_batch = float(duration)

    logging.info(info_str.format(step=step,
                                 total_steps=hypes['solver']['max_steps'],
                                 objective=loss_value,
                                 lr_value=lr,
                                 sec_per_batch=sec_per_batch,
                                 examples_per_sec=examples_per_sec)
                 )

# def _print_eval_dict(eval_names, eval_results, prefix=''):
#     print_str = string.join([nam + ": %.2f" for nam in eval_names],
#                             ', ')
#     print_str = "   " + prefix + "  " + print_str
#     logging.info(print_str % tuple(eval_results))

def run_training(hypes, graph, sess):
    display_iter = hypes['logging']['display_iter']
    save_iter = hypes['logging']['save_iter']
    lr = hypes['solver']['learning_rate']
    feed_dict = {graph['learning_rate']: lr}

    #start clock
    start_time = time.time()
    for step in range(hypes['train']['steps']):
        if step % display_iter:
            sess.run([graph['train_op']], feed_dict=feed_dict)

            # Write the summaries and print an overview fairly often.
        elif step % display_iter == 0:
            # Print status to stdout.
            _, obj_value = sess.run([graph['train_op'],
                                      graph['obj']],
                                     feed_dict=feed_dict)

            _print_training_status(hypes, step, obj_value, start_time, lr)

            # Reset timer
            start_time = time.time()
        # Save a checkpoint periodically.
        if (step) % save_iter == 0 and step > 0 or \
                (step + 1) == hypes['solver']['max_steps']:
            # write checkpoint to disk
            checkpoint_path = os.path.join(hypes['dirs']['output_dir'],
                                           'model.ckpt')
            sess['saver'].save(sess, checkpoint_path, global_step=step)
            # Reset timer
            start_time = time.time()

import json
import logging

if __name__ == '__main__':
    with open('hyperparameters.json', mode='r') as f:
        hypes = json.load(f)
    folder = hypes['dirs']['output_dir']
    weightspath = os.path.join(os.getcwd(), folder)
    if not os.path.exists(weightspath):
        os.makedirs(weightspath)
    hypes['dirs']['output_dir'] = weightspath
    q = create_queue(hypes)
    img, lab = input(hypes, q, 'train')
    train_graph = build_train_graph(hypes, img, lab)
    with tf.Session() as sess:
        session = utils.start_session()
        start_enqueue_thread(hypes, q, sess)
        run_training(hypes, train_graph, session)
        # stopping input Threads
        session['coord'].request_stop()
        session['coord'].join(session['threads'])