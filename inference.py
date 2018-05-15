from model import feature_extractorl, classifier

def inference(hypes, image, phase):
    fex = feature_extractorl()
    feature = fex.build(image)
    char_list = [classifier('chcnt_'+str(i)).build(feature, phase) for i in range(20)]
    return char_list