import numpy as np

#mask部分を取り出す
def crop_local(images=None,points=None,l_size=64):
    '''
    # Arguments:
        images(ndarray): (batch,H,W,1)
        points(int): (x,y) top-left coordinate
        l_size(int): local_size
    '''
    assert points.shape[0] == images.shape[0]
    locals = []
    for img,p in zip(images,points):
        x,y = p
        local = img[y:y+l_size,x:x+l_size,:]
        locals.append(local)

    return np.array(locals)
