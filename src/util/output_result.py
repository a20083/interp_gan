import matplotlib.pyplot as plt
import numpy as np
import math,os
from PIL import Image

def plot_loss_1(save_path,save_name,loss_list,ylabel_name):
    '''損失をグラフに描画して保存するモジュール
    # Arguments:
        save_path(str): 保存先までの絶対パス
        save_name(str): 保存名
        loss_list(list): 損失が入ったリスト
    '''

    epoch = np.arange(0,len(loss_list))
    fig ,ax = plt.subplots()
    ax.plot(epoch,loss_list,label=save_name)
    ax.grid()
    ax.set(xlabel='iteration',ylabel=ylabel_name,title='loss')
    ax.legend()
    plt.savefig(os.path.join(save_path,save_name+'.png',))
    plt.close()

def plot_loss_2(save_path,save_name,loss_list1,loss_list2,ylabel_name):
    '''2種類の損失値をグラフに描画して保存するモジュール

    # Arguments
        save_path(str): プロットした表の保存先
        save_name(str): 保存名
        loss_list1(list): 本物と識別した場合の損失
        loss_list2(list): 偽物と識別した時の損失
    '''

    epoch = np.arange(0,len(loss_list1)) #次元を合わせる
    fig ,ax = plt.subplots()
    ax.plot(epoch,loss_list1,label='d real loss')
    ax.plot(epoch,loss_list2,label='d fake loss')
    ax.grid()
    ax.set(xlabel='iteration',ylabel=ylabel_name,title='discriminator loss')
    ax.legend()
    plt.savefig(os.path.join(save_path,save_name+'.png'))
    plt.close()

def save_images(save_path,images,save_name):
    '''1枚の画像を保存するモジュール

    # Arguments
        save_path(str): 保存先
        images(ndarray): (n,H,W,channel)
        epoch(int): 学習回数
    '''
    for num,img in enumerate(images):
        pil_img = Image.fromarray(np.uint8(img))
        pil_img.save(os.path.join(save_path,save_name+f'_{num}.png'))

def draw_heatmap(save_path,save_name,depth_array,vmin=0,vmax=3000):
    '''
    # Arguments:
        save_path(str) :保存先までの絶対パス
        save_name(str) :保存名
        depth_array(ndarray) :距離画像 shape=(n,H,W,1)
        vmin(int) :カラーバーの最小値
        vmax(int) :カラーバーの最大値

    # Note
        ax.imshow()はチャネル1を取れない
        np.squeeze() などで対処する

        x (,(image),1)
        o (,(image))
    '''

    # 次元の削減
    if depth_array.ndim==4 and depth_array.shape[-1]==1:
        depth_array = np.squeeze(depth_array,axis=-1)

    for num,image in enumerate(depth_array):
        fig,ax = plt.subplots()
        im=ax.imshow(image,vmin=vmin,vmax=vmax,cmap='hot')

        #メモリの削除
        ax.tick_params(labelbottom=False,bottom=False)
        ax.tick_params(labelleft=False,left=False)

        fig.colorbar(im,ax=ax)
        plt.savefig(os.path.join(save_path, save_name +f'_{num}.png'))
        plt.close(fig)

def depth_to_rgb(depth_array):
    '''
    # Arguments:
        depth_ndarray(ndarray): (n,H,W,1)

    # Returns:
        image(ndarray): (n,H,W,3)

    # Note:
        depth_ndarrayには正規化されていないテンソルを受け取る
    '''
    depth_array = np.squeeze(depth_array,axis=-1)
    image_size = depth_array.shape[1:]

    def f(img):
        image = np.zeros(image_size+(3,))
        img = img.astype(np.int32)
        image[:,:,2] = img & 0xff           # Blue
        image[:,:,1] = (img & 0xff00)>>8    # Green
        image[:,:,0] = (img & 0xff0000)>>16 # Red
        return image

    color_imgs = list(map(lambda x: f(x),depth_array))
    return np.array(color_imgs)
