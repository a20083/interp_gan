import os
from PIL import Image
import random
import re
import numpy as np

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

def random_crop(dir_path,H,W,multiple_num):
    '''1枚の画像から数枚の画像を作成する(ランダムでclipする)

    # Arguments:
        dir_path(str)   :画像があるフォルダまでの絶対パス
        H(int)          :取得したい画像の高さ
        W(int)          :取得したい画像の幅
        multiple_num(int) :何枚画像を生成するか

    # Returns:
        None
    '''
    nd_images = _open_png_images(dir_path)
    w_range = nd_images.shape[1] - H
    h_range = nd_images.shape[2] - W

    images = []
    for num,nd_img in enumerate(nd_images):
        for i in range(multiple_num):
            img = np.zeros((H,W,3))
            x = random.randint(0,w_range)
            y = random.randint(0,h_range)
            img = nd_images[num,y:y+H,x:x+W,:]
            images.append(img)

    return np.array(images)

def remove_not_covered_img(dir_path):
    '''画像全体の2割以上depthがない場合削除する
    # Arguments:

    '''
    H = 256
    W = 256
    nd_images = random_crop(dir_path,H,W,30)

    def f(img):
        channel_sum = np.sum(img,axis=-1)
        mask_image = np.where(channel_sum>0,1,0)
        return np.sum(mask_image)/(H*W)>0.8

    covered_images = list(filter(lambda x:f(x),nd_images))
    return np.array(covered_images)

def save_images(save_path,images):
    '''
    # Arguments:
        save_path(str)  : full path
        images(ndarray) : (n,H,W,3)
    '''
    for num,img in enumerate(images):
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(save_path,f'{num}.png'))


if __name__ == '__main__':
    dir_path = ''
    save_path = ''
    H = 256
    W = 256
    num = 10
    # save_images(save_path,random_crop(dir_path,save_path,H,W,num))
    save_images(save_path,remove_not_covered_img(dir_path))
