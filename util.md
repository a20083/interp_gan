## draw_twisty_line_mask.py

def twisty_line()

---
## load_data.py

---
#### \_open_png_images(dir_path)
```
#pngファイル以外のファイルは削除
pattern = r".*.png"
```

pngファイル以外の画像をloadしたいときは、

```
pattern = r".*.bmp"
```

と書き換えること

---
#### png_to_depth(dir_path)
depth画像をdepthの配列に変換
$$
depth=256^2 \times Red+256 \times Green+Blue
$$
で値を計算している
![stanford_bunny](https://github.com/a20083/interp_gan/blob/master/image/bunny_1.png)  

---
#### png_to_mask(dir_path)
util/draw_twisty_line.pyを使って作成したジグザグのマスクをnumpyの型にして返す関数

tqdm という関数があるが、これはプログレスバーを表示するためなので、気にしなくても良い


---

## mask.py
マスクを作る関数を集めたモジュール
ganで補間する部分を1として他の部分を0にしている　

---
#### slit_mask(image_size,batch_size)

10pixelごとに値を0にするリスト
```
zero_col_idx = [col for col in range(image_size) if col%10<2]
```

---

#### \_detect\_corner(img,axis=0)

ここではlambda式(ラムダ式)を使っている。
lambda式については、[ここ](https://www.sejuku.net/blog/23677)をみてほしい
```
if axis == 0:
        isDepth_1 = lambda idx: np.any(img[idx,])
        isDepth_2 = lambda idx: np.any(img[iter-idx-1,])
    else:
        isDepth_1 = lambda idx: np.any(img[:,idx])
        isDepth_2 = lambda idx: np.any(img[:,iter-idx-1])
```

この_detect_cornerがやっていることは、例えば  
このようなdepth画像があったとすると  
0 0 0 0 0   
0 0 3 8 0  
0 2 0 7 0  
0 5 0 0 0　　

このようにdepthを包含するmaskを生成する関数  
0 0 0 0 0   
0 1 1 1 0  
0 1 1 1 0  
0 1 1 1 0　　

---
#### make_detected_mask(image_data)

先ほどの_detected_maskを使ってN枚の画像に対してmaskを生成する関数

---
#### save_bin_mask(mask)
入力は (N,H,W,1)
maskの次元は4でないとエラーがでる  
mask.ndim == 4

あくまでこの関数は正しくmaskを生成できているか確認するため

---
## output_result.py

---
#### plot_loss_1(save_path,save_name,loss_list,ylabel_name)
train.pyで計算したlossをloss_listに渡している    
この関数を呼び出すと下図のようなグラフが描画される  
仮引数のylabel_nameは下図左のdepth_lossにあたる  
![depth_loss](https://github.com/a20083/interp_gan/blob/master/image/depth_loss.png)


---
#### plot_loss_2(save_path,save_name,loss_list1,loss_list2,ylabel_name)
plot_loss_1と同じ  
この関数を呼び出すと下図のようなグラフが描画される  
ちなみにdiscriminatorの学習は失敗である  
損失は普通0にならない
![d_loss](https://github.com/a20083/interp_gan/blob/master/image/d_loss.png)

---
#### save_images(save_path,images,save_name)
色のついた(rgbがある)画像を保存するための関数  
以前につくった名残  
特に使ってない

---
#### draw_heatmap(save_path,save_name,depth_array,vmin,vmax)
この関数を呼び出すと下図のようなグラフが描画される  
右側のcolorbarのスケールは仮引数のvmin,vmaxで指定  
今回使った胃の形状データのdepth画像は最大が約3200なので、vmin=0,vmax=3500で設定している  

![c_result](https://github.com/a20083/interp_gan/blob/master/image/c_result30000_1.png)

---

#### depth_to_rgb(depth_array)

名前の通りdepth画像をrgbに変換する関数である  
rgbをdepthにする関数(png_to_depth)を作ったので、これも作ったが、使わない
気にしなくてもいい

---

## random_crop.py

##### random_crop(dir_path,H,W,multiple_num)
一枚の画像から複数枚画像をランダムでcropする

元の画像
![stomach](https://github.com/a20083/interp_gan/blob/master/image/0000-3b.bmp)

cropされた画像  
![crop1](https://github.com/a20083/interp_gan/blob/master/image/crop_0.png)  
![crop2](https://github.com/a20083/interp_gan/blob/master/image/crop_1.png)

#### remove_not_covered_img(dir_path)
上の元画像をrandom_cropをすると、周りの黒い部分のみの画像も得られてしまう
なので、cropした画像の中から８割以上depthで埋められた画像を採用し、  
保存する関数である


----
## util.py

#### crop_local.py

画像をランダムでcropする関数で、学習で使う  
Local discriminatorに画像を入れるために使う
![network](https://github.com/a20083/interp_gan/blob/master/image/glcic_network.png)
(参照)[Globally and Locally Consistent Image Completion](http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf)

src/train.py
```
# point1,point2はcropする始点の(x,y)
# l_size : local_size 

fake_local = crop_local(fake_global,point1,self.l_size)
real_local = crop_local(real_batch,point2,self.l_size)
```
