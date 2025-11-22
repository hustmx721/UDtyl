import numpy as np
import torch
import os

"""
    create dir if no one exist
"""
def makedir(dir='./default_dir'):
    if not os.path.exists(dir): os.mkdir(dir)
        
"""
    create config.ini file
"""
def create_config(name):
    from configparser import ConfigParser

    #Get the configparser object
    config_object = ConfigParser()

    #Assume we need 2 sections in the config file, let's call them USERINFO and SERVERCONFIG
    config_object["EXAMPLE"] = {
        "a": "something",
        "b": "place holder",
    }

    #Write the above sections to config.ini file
    with open(name, 'w') as conf:
        config_object.write(conf)

def read_config(name):
    from configparser import ConfigParser
    config_object = ConfigParser()
    config_object.read(name)
    return config_object

################## PYTORCH UTILS  ################

def show_images(images, normalize=None, ipython=True,
                margin_height=2, margin_color='red',
                figsize=(18,16), save_npy=None):
    """ Shows pytorch tensors/variables as images """
    import matplotlib.pyplot as plt
    
    # first format the first arg to be hz-stacked numpy arrays
    if not isinstance(images, list):
        images = [images]
    images = [np.dstack(image.cpu().numpy()) for image in images]
    image_shape = images[0].shape
    assert all(image.shape == image_shape for image in images)
    assert all(image.ndim == 3 for image in images) # CxHxW

    # now build the list of final rows
    rows = []
    if margin_height >0:
        assert margin_color in ['red', 'black']
        margin_shape = list(image_shape)
        margin_shape[1] = margin_height
        margin = np.zeros(margin_shape)
        if margin_color == 'red':
            margin[0] = 1
    else:
        margin = None

    for image_row in images:
        rows.append(margin)
        rows.append(image_row)

    rows = [_ for _ in rows[1:] if _ is not None]
    plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='k')

    cat_rows = np.concatenate(rows, 1).transpose(1, 2, 0)
    imshow_kwargs = {}
    if cat_rows.shape[-1] == 1: # 1 channel: greyscale
        cat_rows = cat_rows.squeeze()
        imshow_kwargs['cmap'] = 'gray'

    plt.imshow(cat_rows, **imshow_kwargs)

    if save_npy is not None:
        scipy_img = scipy.misc.toimage(cat_rows)
        scipy_img.save(save_npy)

    plt.show()
    
    

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return str(self.avg)