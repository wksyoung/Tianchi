import tensorflow as tf

def start_session():
    saver = tf.train.Saver()
    sess = tf.get_default_session()
    init = tf.initialize_all_variables()
    sess.run(init)
    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    session = {}
    session['sess'] = sess
    session['saver'] = saver
    session['coord'] = coord
    session['thread'] = threads

    return session