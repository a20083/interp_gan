import os
import re
import argparse
from PIL import Image

def pngs_path(dir_path):
    '''
    # Return: absolute path list
    '''
    folder = os.listdir(dir_path)

    #pngファイル以外のファイルは削除
    pattern = r".*.png"
    p = re.compile(pattern)
    pngs = list(filter(lambda x: p.match(x),folder))
    abs_pngs = list(map(lambda x:os.path.join(dir_path,x),pngs))
    return abs_pngs

def open_image(path_list):
    '''
    # Return: PIL_object
    '''
    return map(lambda x: Image.open(x),path_list)

def trim_images(pil_imgss,x,y,H,W):
    '''
    # Arguments:
        x(int): トリミングする画像の始点x
        y(int): トリミングする画像の始点y
        H(int): トリミングの高さ
        W(int): トリミングの幅

    # Return: PIL object iterator
    '''
    return map(lambda k: k.crop((x,y,x+W,y+H)),pil_imgs)

def resize_images(pil_imgs,H,W):
    '''
    # Arguments:
        H(int): トリミングの高さ
        W(int): トリミングの幅

    # Return: PIL object iterator
    '''
    return map(lambda k: k.resize((W,H)),pil_imgs)

def rotate_images(pil_imgs,degree_lis):
    '''
    # Arguments:
        degree_lis(list): (例)[0,30,60,90]

    # Return: list
    '''
    rotated_imgs =[]
    for img in pil_imgs:
        for deg in degree_lis:
            rotated = img.rotate(deg,resample=Image.NEAREST)
            rotated_imgs.append(rotated)

    return rotated_imgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image_preprocessing')
    parser.add_argument('-t',help='trimming',action='store_true')
    parser.add_argument('-re',help='resize',action='store_true')
    parser.add_argument('-ro',help='rotate',action='store_true')
    args = parser.parse_args()

    '''
    使い方
    1, 画像が入ったフォルダまでの絶対パスを指定
    2, 保存先までの絶対パスを指定
    3, -t -ro -re　目的にあったオプションをつけてプログラムを実行
    '''

    d_path = r'C:\Users\a20083\tkouno\src\temp_img'
    s_path = r'C:\Users\a20083\tkouno\src\temp_img\crop'
    pil_imgs = open_image(pngs_path(d_path))

    if args.t :
        pil_imgs = trim_images(pil_imgs,300,300,600,600)
    if args.re:
        pil_imgs = resize_images(pil_imgs,100,100)
    if args.ro:
        pil_imgs = rotate_images(pil_imgs,[90])

    for num,img in enumerate(pil_imgs):
        img.save(os.path.join(s_path,f'{num}.png'))
