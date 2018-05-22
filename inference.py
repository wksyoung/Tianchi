from model import feature_extractorl, classifier
import tensorflow as tf
def inference(hypes, image, phase):
    fex = feature_extractorl('vgg16.npy')
    feature = fex.build(image)
    string_length = hypes['arch']['maxstr']
    logits = hypes['arch']['dictlen']

    char_list = [tf.expand_dims(classifier('chcnt_'+str(i), logits).build(feature, phase),1) for i in range(string_length)]
    return tf.concat(char_list, 1)