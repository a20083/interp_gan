import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from util.load_data import png_to_depth
from util.mask import obj_slit_mask
from util.output_result import draw_heatmap,save_images
#import pdb; pdb.set_trace()

#pandas to numpy
#  (numpy) = (pd).values
#numpy to pandas
#  (pd) = pd.DataFrame(numpy)

def interp_slit(img_data,mask):
    assert img_data.shape == mask.shape
    assert img_data.ndim == 4

    def f(depth,limit=10,method='linear'):
        '''
        # Arguments:
            depth(ndarray): (例) (128,128,1)
            limit(int): 同じ値が何回続いたら補間を止めるか
            method(str):補間手法 linear ,nearest,cubicなど
        '''
        depth = np.squeeze(depth,axis =-1)
        df = pd.DataFrame(depth)
        df = df.where(df > 0)
        df = df.interpolate(axis=1,limit=limit,method=method,limit_direction='both')
        df = df.fillna(0)
        ndarray = df.values
        ndarray = ndarray[:,:,None]
        return ndarray

    interp_data = list(map(lambda x :f(x,limit=15,method='linear'),img_data * (1-mask)))
    interp_data = np.array(interp_data)

    fanc = lambda real,fake,mask: real*(1-mask)+fake*mask
    return fanc(img_data,interp_data,mask)

if __name__ == '__main__':
    dir_path = ''
    image_data = png_to_depth(dir_path)
    slit_mask = obj_slit_mask(image_data)

    assert not np.array_equal(image_data[0],slit_mask[0])
    interp_data = interp_slit(image_data,slit_mask)

    s_path = ''
    # draw_heatmap(s_path,'tmp',np.squeeze(slit_mask*3000,axis=-1))
    # draw_heatmap(s_path,'tmp',np.squeeze(interp_data,axis=-1))
    # draw_heatmap(s_path,'tmp',image_data*slit_mask)
