'''
Globally and Locally Consistent Image Completion
ディープネットワークによるシーンの大域的かつ局所的な整合性を考慮した画像補完
http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf
'''

from keras.models import Model
from keras.layers import Input,Dense,Reshape,Concatenate,Lambda,Flatten,Activation
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.losses import mean_absolute_error,binary_crossentropy
from keras import backend as K
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self,slit_mask,cons):
        '''
        # Args
            slit_mask: binary mask shape=(1,H,W,1)
            cons(dictionary): from config.py
        '''
        self.slit_mask = slit_mask
        self.G_SHAPE = cons['GLOBAL_SHAPE']
        self.L_SHAPE = cons['LOCAL_SHAPE']
        self.dp = cons['DEPTH_PENALTY']
        self.half = cons['half']

        self.netG = self.build_completion_net()

    # 自作の損失関数
    #########
    # マスクをスリットの部分だけではなく全体でかける
    # スリットの部分のみの差を損失関数に入れる λ|x-comp_net(x)|
    #########
    def _g_conv(self,filters=64,kernel=1,strides=1,dilate=1):
        def f(input):
            x = Conv2D(filters,kernel,strides=strides,padding='same',dilation_rate=1)(input)
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
            return x
        return f

    def _d_conv(self,filters=64,kernel=1,strides=1,dilate=1):
        def f(input):
            x = Conv2D(filters,kernel,strides=strides,padding='same',dilation_rate=1)(input)
            x = LeakyReLU(0.2)(x)
            x = BatchNormalization()(x)
            return x
        return f

    def _deconv(self,filters=64,kernel=1,strides=1):
        def f(input):
            x = Conv2DTranspose(filters,kernel,strides=strides,padding='same')(input)
            x = LeakyReLU(0.2)(x)
            x = BatchNormalization()(x)
            return x
        return f

    def build_completion_net(self):
        cnum = 64
        input_images = Input(shape=self.G_SHAPE,name='global_image')
        obj_mask = Input(shape=self.G_SHAPE,name='obj_mask')

        x = Lambda(lambda k: (k-self.half)/self.half)(input_images) # depth to [-1~1]
        x = self._g_conv(cnum,5,1)(x)
        x = self._g_conv(cnum*2,3,2)(x)
        x = self._g_conv(cnum*2,3,1)(x)
        x = self._g_conv(cnum*4,3,2)(x)
        x = self._g_conv(cnum*4,3,1)(x)
        x = self._g_conv(cnum*4,3,1)(x)
        x = self._g_conv(cnum*4,3,1,2)(x)
        x = self._g_conv(cnum*4,3,1,4)(x)
        x = self._g_conv(cnum*4,3,1,8)(x)
        x = self._g_conv(cnum*4,3,1,16)(x)
        x = self._g_conv(cnum*4,3,1)(x)
        x = self._g_conv(cnum*4,3,1)(x)
        x = self._deconv(cnum*2,4,2)(x)
        x = self._g_conv(cnum*2,3,1)(x)
        x = self._deconv(cnum,4,2)(x)
        x = self._g_conv(int(cnum/2),3,1)(x)
        x = Conv2D(1,3,strides=1,padding='same',activation='tanh')(x)
        x = Lambda(lambda k: k*self.half+self.half)(x) # [-1~1] to depths
        out = Lambda(lambda k: k[0]*k[1])([x,obj_mask])
        model = Model(inputs=[input_images,obj_mask], outputs=out,name='completion_net')
        return model

    def build_generator_with_own_loss(self):
        # returns: generator model and trainer
        real_global = Input(shape=self.G_SHAPE,name='global_image')
        obj_mask = Input(shape=self.G_SHAPE,name='obj_mask')
        slit_mask = Input(shape=self.G_SHAPE,name='slit_mask')

        train_data = real_global * (1-slit_mask)
        fake_global = self.netG([train_data,obj_mask])
        abs_error = K.abs(real_global - fake_global)
        loss =  K.mean(abs_error + self.dp*(1-slit_mask)*abs_error)

        g_updates = \
            Adam(lr=1e-5,beta_1=0.1)\
                .get_updates(loss,self.netG.trainable_weights)

        g_train = K.function(inputs = [real_global,obj_mask,slit_mask],\
                             outputs = [loss],\
                             updates = g_updates)

        return self.netG , g_train

    def build_local_discriminator(self):
        cnum = 64
        input = Input(shape=self.L_SHAPE)
        x = Lambda(lambda k: (k-self.half)/self.half)(input) #depth to [-1~1]
        x = self._d_conv(cnum,5,2)(x)
        x = self._d_conv(cnum*2,5,2)(x)
        x = self._d_conv(cnum*4,5,2)(x)
        x = self._d_conv(cnum*8,5,2)(x)
        x = self._d_conv(cnum*8,5,2)(x)
        x = Flatten()(x)
        x = Dense(1024,activation='sigmoid')(x)
        model = Model(inputs=input, outputs=x,name='local_discriminator')
        return model

    def build_global_discriminator(self):
         cnum = 64
         input = Input(shape=self.G_SHAPE)
         x = Lambda(lambda k: (k-self.half)/self.half)(input) #depth to [-1~1]
         x = self._d_conv(cnum,5,2)(x)
         x = self._d_conv(cnum*2,5,2)(x)
         x = self._d_conv(cnum*4,5,2)(x)
         x = self._d_conv(cnum*8,5,2)(x)
         x = self._d_conv(cnum*8,5,2)(x)
         x = self._d_conv(cnum*8,5,2)(x)
         x = Flatten()(x)
         x = Dense(1024,activation='sigmoid')(x)
         model = Model(inputs=input, outputs=x,name='global_discriminator')
         return model

    def build_discriminator_with_own_loss(self):
        '''
        # Returns:
            discriminator(model) : global,localの二つを合わせたモデル
        '''
        global_images = Input(shape=self.G_SHAPE,name='global_images')
        local_images = Input(shape=self.L_SHAPE,name='local_images')

        g_dis_out = self.build_global_discriminator()(global_images)
        l_dis_out = self.build_local_discriminator()(local_images)

        concat = Concatenate()([g_dis_out,l_dis_out])
        out = Dense(1,activation='sigmoid')(concat)

        combined_dis = Model(inputs=[global_images,local_images],outputs=out,name='g_l_discriminator')
        optimizer = Adam(lr=1e-5,beta_1=0.1)
        combined_dis.compile(loss='binary_crossentropy',optimizer=optimizer)
        return combined_dis

    def build_combined_model(self,comp_net,combined_dis,batch_size=32):
        '''
        # Aeguments:
            comp_net(Model) :completion_net
            combined_dis(Model) :(global + local) discriminator

        # Returns:
            model(model) : completion , discriminatorの二つを合わせたモデル
        '''

        l_size = self.L_SHAPE[1]
        g_size = self.G_SHAPE[1]

        #discriminatorの重みは固定する
        combined_dis.trainable = False
        for layer in combined_dis.layers:
            layer.trainable = False

        input_images = Input(shape=self.G_SHAPE,name='global_images')
        obj_mask = Input(shape=self.G_SHAPE,name='obj_mask')
        local_mask = Input(shape=self.G_SHAPE,name='local_mask')

        '''
        def crop(imgs,l_mask):
            imgs = K.cast(imgs,'float32')
            g_fake_m = tf.boolean_mask(imgs,l_mask)
            l_fake = K.reshape(g_fake_m,(batch_size,)+self.L_SHAPE)
            return l_fake
        '''
        g_fake = comp_net([input_images,obj_mask])
        '''
        l_fake = Lambda(lambda k: \
            crop(k[0],k[1]),name='crop_imgs')([g_fake,local_mask])
        '''
        l_fake = Lambda(lambda k:k[0]*k[1])([g_fake,local_mask])
        out = combined_dis([g_fake,l_fake])

        model = Model([input_images,obj_mask,local_mask],out,name='model_all')
        optimizer = Adam(lr=1e-5,beta_1=0.1)
        model.compile(loss='binary_crossentropy',optimizer=optimizer)
        return model
