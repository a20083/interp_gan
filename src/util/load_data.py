from PIL import Image
import os
import numpy as np
import pprint
import re
import time
from tqdm import tqdm

def _open_png_images(dir_path):
    '''
    # Return:
        ndarray: (N,H,W,channel)
    '''
    folder = os.listdir(dir_path)

    #pngファイル以外のファイルは削除
    pattern = r".*.png"
    p = re.compile(pattern)
    pngs = filter(lambda x: p.match(x),folder)

    abs_pngs = map(lambda x:os.path.join(dir_path,x),pngs)

    images =[]
    for img_path in abs_pngs:
        img = Image.open(img_path)
        nd_img = np.asarray(img)
        images.append(nd_img)

    return np.array(images)

def png_to_depth(dir_path):
    '''全ての距離画像を訓練データ(4次元)として返す
    # Arguments
        dir_path(str):画像フォルダまでの絶対パス

    # Returns
        depth_data(ndarray): (n,H,W,1)
    '''
    numpy_imgs = _open_png_images(dir_path)

    def _rgb_to_depth(img):
        image = (256**2 * img[:,:,0]) + 256*img[:,:,1] + img[:,:,2]
        return image

    depth_data = []
    for img in tqdm(numpy_imgs):
        depth_data.append(_rgb_to_depth(img))

    return np.array(depth_data)[:,:,:,None]

def png_to_mask(dir_path):
    numpy_imgs = _open_png_images(dir_path)

    def _img_to_mask(img):
        img = np.asarray(img)
        image = np.where(img>0,1,0)
        return image

    mask_data =[]
    for img in tqdm(numpy_imgs):
        mask_data.append(_img_to_mask(img))

    return np.array(mask_data)[:,:,:,None]

if __name__ == '__main__':
    from output_result import draw_heatmap
    dir_path = ''
    save_path = ''
    a = png_to_depth(dir_path)
    draw_heatmap(save_path,'heatmap',a,0,3500)
