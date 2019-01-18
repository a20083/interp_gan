import numpy as np
import os

#自作関数
from util.utils import crop_local
from util.output_result import plot_loss_1,plot_loss_2,draw_heatmap
from util.mask import rand_mask

class Train:
    # バイナリマスクに関して 穴の部分が1、それ以外が0
    def __init__(self,netC, netD,c_train,\
                    image_data,slit_mask,obj_mask,\
                        cons):
        '''
        # Arguments:
            netC(Model):            generator model
            netD(Model):            discriminator
            c_train:                train_function
            image_data(ndarray):    (N,H,W,channel) 正解画像
            slit_mask(ndarray):     (,H,W,channel) 穴は1
            obj_mask(ndarray):      (N,H,W,channel) 穴は1
            cons(dict):             from config.py
        '''
        self.netC = netC
        self.netD = netD
        self.c_train = c_train
        self.image_data = image_data
        self.slit_mask = slit_mask
        self.obj_mask = obj_mask
        self.g_size = cons['GLOBAL_SIZE']
        self.l_size = cons['LOCAL_SIZE']
        self.batch_size = cons['BATCH_SIZE']
        self.T_c =cons['T_c']
        self.T_d = cons['T_d']

    def set_trainable(self,model,trainable=True):
        model.trainable = trainable

        for layer in model.layers:
            layer.trainable = trainable

    def train_completion_net(self,c_save_path):
        depth_loss = [] #損失保存用
        for t in range(self.T_c):
            # 正解データとオブジェクトマスクの抽出
            rand_int = np.random.randint(0,self.image_data.shape[0],self.batch_size)
            real_batch = self.image_data[rand_int]
            obj_batch_mask = self.obj_mask[rand_int]

            # スリットマスクをランダムで抽出
            rand_int_2 = np.random.randint(0,self.slit_mask.shape[0],self.batch_size)
            slit_batch_mask = self.slit_mask[rand_int_2]

            c_loss = self.c_train([real_batch,obj_batch_mask,slit_batch_mask])
            print(f'======iteration : {t+1}/{self.T_c}========')
            print('depth_loss : ',c_loss)

            #損失の記録
            if t%100==0:
                depth_loss.append(float(c_loss[0]))
                #損失をグラフにして保存
                plot_loss_1(c_save_path,'depth_loss',depth_loss,'depth_loss')
                #学習した重みを保存
                self.netC.save_weights(os.path.join(c_save_path,'Completion_net.hdf5'))

            #画像の保存
            if t % 1000 ==0:
                # 訓練データの作成
                train_batch = real_batch * (1-slit_batch_mask)
                fake_global = self.netC.predict([train_batch , obj_batch_mask])
                draw_heatmap(c_save_path,f'c_result{t+1}',fake_global[:2,],0,3500)

        #損失をグラフにして保存
        plot_loss_1(c_save_path,'depth_loss',depth_loss,'depth_loss')
        #学習した重みを保存
        self.netC.save_weights(os.path.join(c_save_path,'Completion_net.hdf5'))
        #学習結果を保存
        draw_heatmap(c_save_path,f'c_result{t+1}',fake_global[:2,],0,3500)

    def train_discriminator(self,d_save_path,glcic):
        '''
        # Arguments:
            glcic(Model): completion net + discriminator
        '''
        #損失を保存する空のリストを用意
        d_fake_loss = []
        d_real_loss = []
        g_binary_loss = []

        for t in range(self.T_d):
            # 正解データとオブジェクトマスクの作成
            rand_int = np.random.randint(0,self.image_data.shape[0],self.batch_size)
            real_batch = self.image_data[rand_int]
            obj_batch_mask = self.obj_mask[rand_int]

            # 訓練データの作成
            rand_int_2 = np.random.randint(0,self.slit_mask.shape[0],self.batch_size)
            slit_batch_mask = self.slit_mask[rand_int_2]
            train_batch = real_batch * (1-slit_batch_mask)

            # local部分の座標を取得
            point1, l_masks = rand_mask(1,self.g_size,self.l_size,1)
            point2, _       = rand_mask(1,self.g_size,self.l_size,1)

            # point1 = (x,y)をbatch_size個複製する
            point1 = np.tile(point1,(self.batch_size,1))
            l_masks = np.tile(l_masks,(self.batch_size,1,1,1))
            point2 = np.tile(point2,(self.batch_size,1))

            fake_global = self.netC.predict([train_batch ,obj_batch_mask])

            fake_local = crop_local(fake_global,point1,self.l_size)
            real_local = crop_local(real_batch,point2,self.l_size)
            # 学習
            d_f_loss = self.netD.train_on_batch([fake_global, fake_local],[0]*self.batch_size)
            d_t_loss = self.netD.train_on_batch([real_batch, real_local],[1]*self.batch_size)

            print(f'=====iteration : {t+1}/{self.T_d}========')
            print('d_fake_loss : ',d_f_loss)
            print('d_real_loss : ',d_t_loss)

            #損失のログをとる
            if t%100==0:
                d_real_loss.append(d_t_loss)
                d_fake_loss.append(d_f_loss)

                #学習した重みを保存
                self.netC.save_weights(os.path.join(d_save_path,'Completion_net.hdf5'))
                self.netD.save_weights(os.path.join(d_save_path,'Discriminator.hdf5'))

                #損失をグラフにして保存
                plot_loss_1(d_save_path,'g_binary_loss',g_binary_loss,'binary_crossentropy')
                plot_loss_2(d_save_path,'d_loss',d_real_loss,d_fake_loss,'binary_crossentropy')

            if d_f_loss + d_t_loss < 1.0: # ここの値は任意
                self.set_trainable(self.netD,False)
                g_b_loss = glcic.train_on_batch([train_batch, obj_batch_mask,l_masks] , [1]*self.batch_size)
                self.set_trainable(self.netD,True)
                print('g_binary_loss : ',g_b_loss)
                if t%10==0:
                    g_binary_loss.append(g_b_loss)

                #学習結果を保存
                if t%100 == 0:
                    draw_heatmap(d_save_path,f'd_result{t+1}',fake_global[:2,],0,3500)

        #学習した重みを保存
        self.netC.save_weights(os.path.join(d_save_path,'Completion_net.hdf5'))
        self.netD.save_weights(os.path.join(d_save_path,'Discriminator.hdf5'))

        #損失をグラフにして保存
        plot_loss_1(d_save_path,'g_binary_loss',g_binary_loss,'binary_crossentropy')
        plot_loss_2(d_save_path,'d_loss',d_real_loss,d_fake_loss,'binary_crossentropy')

        draw_heatmap(d_save_path,f'd_result{t+1}',fake_global[:2,],0,3500)
