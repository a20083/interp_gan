import os
import argparse
import sys
import numpy as np
from PIL import Image
from functools import reduce

#自作関数のインポート
from util.output_result import draw_heatmap

class EvalModel:
    def __init__(self,save_path,real_data,fake_data,masks):
        '''
        # Arguments:
            save_path(str): 保存先
            real_data(ndarray): shape=(batch,H,W,1)
            fake_data(ndarray): shape=(batch,H,W,1)
            masks(ndarray): shape=(batch,H,W,1)
        '''
        self.save_path = os.path.join(save_path,'test_result')
        self.real_data = real_data
        self.fake_data = fake_data
        self.masks = masks

        os.makedirs(self.save_path,exist_ok='True')

    def complement_data(self):
        draw_heatmap(self.save_path,'test_result',self.fake_data,0,3500)

    def _abs_error(self):
        assert self.real_data.ndim == self.fake_data.ndim
        return abs(self.real_data-self.fake_data)

    def mean_depth_error(self):
        # batch_size * height * widthで割る
        interp_area = np.sum(self.masks > 0)
        return np.sum(self._abs_error()) / interp_area

    def draw_abs_error_map(self):
        draw_heatmap(self.save_path,'depth_error',self._abs_error(),0,100)

    '''
    3次元形状復元
    (参考) http://www.wakayama-u.ac.jp/~chen/education/cv/pinpara.pdf
    resizeした場合 (焦点距離/ピクセル幅) * (resize前)/(resize後)
    '''
    def _write(self,folder_path,num,img):
        with open(os.path.join(folder_path,f'{num}.xyz'),'w') as f:
            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    # data = bytearray([])
                    z = img[col][row]
                    if z ==0: continue
                    z0 = float(z) * 0.0001
                    x = (row-128)/((0.005893/0.00001)*(256/1200))*z0
                    y = (col-128)/((0.005893/0.00001)*(256/1200))*z0
                    f.write(f'{x} {y} {z0}\n')


    def make_xyz_file(self):
        folder_path = os.path.join(self.save_path,'xyz')
        os.makedirs(folder_path,exist_ok='True')
        for num,img in enumerate(self.fake_data):
            self._write(folder_path,num,img)
