import tensorflow as tf
import os
import logging
def start_session():
    saver = tf.train.Saver()
    sess = tf.get_default_session()
    init = tf.global_variables_initializer()
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

def cfg():
    """General configuration values."""
    return None


def _set_cfg_value(cfg_name, env_name, default, cfg):
    """Set a value for the configuration.

    Parameters
    ----------
    cfg_name : str
    env_name : str
    default : str
    cfg : function
    """
    if env_name in os.environ:
        setattr(cfg, cfg_name, os.environ[env_name])
    else:
        logging.info("No environment variable '%s' found. Set to '%s'.",
                     env_name,
                     default)
        setattr(cfg, cfg_name, default)


_set_cfg_value('plugin_dir',
               'TV_PLUGIN_DIR',
               os.path.expanduser("~/tv-plugins"),
               cfg)
_set_cfg_value('step_show', 'TV_STEP_SHOW', 50, cfg)
_set_cfg_value('step_eval', 'TV_STEP_EVAL', 250, cfg)
_set_cfg_value('step_write', 'TV_STEP_WRITE', 1000, cfg)
_set_cfg_value('max_to_keep', 'TV_MAX_KEEP', 10, cfg)
_set_cfg_value('step_str',
               'TV_STEP_STR',
               ('Step {step}/{total_steps}: loss = {loss_value:.2f}; '
                'lr = {lr_value:.2e}; '
                '{sec_per_batch:.3f} sec (per Batch); '
                '{examples_per_sec:.1f} imgs/sec'),
               cfg)
