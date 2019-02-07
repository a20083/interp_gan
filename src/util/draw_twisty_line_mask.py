from PIL import Image,ImageDraw
import random
import os
# https://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html


def twisty_line(x0=0,y0=0,height=256):
    '''
    # Arguments:
        x0(int): start_width
        y0(int): start_height
    '''
    line = []
    for i in range(height):
        x1 = x0 + random.randint(-5,5)
        y1 = y0 + random.randint(0,20)
        line.append((x0,y0,x1,y1))
        x0 = x1
        y0 = y1
        if y0 > height: break
    return line


def draw_twisty_line_mask(save_path,save_name,size,b_color,l_color):
    '''
    # Arguments:
        save_path(str): 保存先までの絶対パス
        save_name(str): 保存名
        size(tuple): (width,height)
        b_color(int): background color
        l_color(int): line color
    '''

    im = Image.new('L',size,b_color)
    draw = ImageDraw.Draw(im)

    line_lis = twisty_line(x0=10,height=size[1])
    step = 0 # 線と線の幅
    for i in range(size[0]):
        for x0,y0,x1,y1 in line_lis:
            draw.line((x0+step, y0, x1+step, y1),fill=l_color,width=2)
        step += 20
    im.save(os.path.join(save_path,save_name+'.png'))

if __name__ == '__main__':
    HEIGHT = 256
    WIDTH = 256
    SIZE = (HEIGHT,WIDTH)
    WHITE = 255
    BLACK = 0

    s_path = ''
    masks_num = 10 # masks_num枚のマスクを作る
    for i in range(masks_num):
        draw_twisty_line_mask(s_path,str(i),SIZE,WHITE,BLACK)
