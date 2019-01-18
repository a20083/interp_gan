'''
depth画像にスリットを数ピクセルおきに入れる
'''
import os
from PIL import Image
import numpy as np

def rand_mask(batch_size=32,g_size=128,l_size=64,channel=1):
    '''
    # Arguments:
        batch_size(int): バッチサイズ
        g_size(int): global size
        l_size(int): local size
        channel(int): チャネル
    '''
    points = []
    l_area_mask = []

    for i in range(batch_size):
        x1 , y1 = np.random.randint(0,g_size-l_size+1,2)
        points.append([x1,y1])

        #穴を開けるマスクを作成
        m = np.zeros((g_size, g_size, channel), dtype=np.uint8)
        m[y1:y1+l_size, x1:x1+l_size,:] = 1
        l_area_mask.append(m)

    return np.array(points),np.array(l_area_mask)


def slit_mask(image_size=128,batch_size=32):
    '''スリットバイナリマスクを返す
    #Arguments:
        image_size(int):
    #Returns:
        slit_bin_masks(ndarray):(batch,H,W,1)
    '''
    # バイナリマスクを作成
    slit_bin_masks = np.ones((batch_size,image_size,image_size,1),dtype=np.int8)
    zero_col_idx = [col for col in range(image_size) if col%10<2]
    slit_bin_masks[:,:,zero_col_idx,:] = 0

    return slit_bin_masks


def object_mask(image_data=None):
    '''depthがある部分のマスク画像
    # Arguments:
        image_data(ndarray): (n,H,W,1) (例)(32,128,128,1)

    # Returns:
        obj_mask(ndarray): depthがある部分が1,それ以外は0
                            shape=(n,H,W,1)
    '''
    # オブジェクトのバイナリマスク
    obj_mask = list(map(lambda x: np.where(x>0,1,0),image_data))
    obj_mask = np.uint8(np.array(obj_mask))

    return obj_mask

def obj_slit_mask(image_data=None):
    '''depthがある部分にスリットを入れたマスク画像
    # Arguments:
        image_data(ndarray): (n,H,W,1) (例)(32,128,128,1)

    # Returns:
        obj_mask(ndarray): 補間したい部分(穴)が1

    '''
    image_size = image_data.shape[1]
    batch_size = image_data.shape[0]
    # スリットのバイナリマスク
    s_mask = slit_mask(image_size,batch_size) # (batch,H,W,1)

    #オブジェクトのバイナリマスク
    obj_mask = object_mask(image_data)

    # AND演算でオブジェクトにスリットが入ったマスクを取得
    obj_slit_mask = obj_mask & s_mask[None,:,:,:]
    return obj_slit_mask

def _detect_corner(img,axis=0):
    '''
    # Arguments:
        img(ndarray): (height,width)
        axis(int): 0=height, 1=width

    # Returns:
        if axis==0: return top,bottom
        if axis==1: return left,right

    '''
    detect_1 = False # left or top
    detect_2 = False # right or bottom

    iter = img.shape[axis]
    if axis == 0:
        isDepth_1 = lambda idx: np.any(img[idx,])
        isDepth_2 = lambda idx: np.any(img[iter-idx-1,])
    else:
        isDepth_1 = lambda idx: np.any(img[:,idx])
        isDepth_2 = lambda idx: np.any(img[:,iter-idx-1])

    for i in range(iter):
        if isDepth_1(i) and (not detect_1):
            side_1 = i
            detect_1 = True

        if isDepth_2(i) and (not detect_2):
            side_2 = iter-i-1
            detect_2 = True

        if detect_1 and detect_2:
            break

    return side_1 , side_2

def make_detected_mask(image_data):
    '''
    # Arguments:
        image_data(ndarray): (N,H,W)
    # Return:
        masks(ndaray): (N,H,W,channel)
    # Note:
        補間したい部分が１
    '''
    if image_data.ndim == 4:
        raise ValueError('np.squeezeで次元を削減してください')

    def f(img):
        top,bottom = _detect_corner(img,axis=0)
        left,right = _detect_corner(img,axis=1)
        mask = np.zeros_like(img,dtype=np.uint8)
        mask[top:bottom+1,left:right+1] = 1
        return mask

    detected_masks = list(map(lambda x: f(x),image_data))
    return np.array(detected_masks)[:,:,:,None]

def save_bin_mask(mask):
    '''バイナリマスク確認用
    # Arguments:
        mank(ndarray) :(N,H,W,1)
    '''
    for i,m in enumerate(mask):
        img = Image.fromarray(np.squeeze(np.uint8(m*255),axis=-1))
        img.convert('L')
        img.save(f'{i}samp.png')
