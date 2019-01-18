import argparse
import os
import tensorflow as tf
from keras import backend as K
import numpy as np
import json

#
# ランダムで作成したスリットマスクを使う
#

#自作関数
from network import Network
from train import Train
from eval_model import EvalModel
from config import cons # constant
from util.load_data import png_to_depth,png_to_mask
from util.mask import obj_slit_mask,slit_mask,object_mask,make_detected_mask

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="train glcic")
    parser.add_argument('-c','--completion_net',
                        help="train completion net",
                        action='store_true')
    parser.add_argument('-d','--discriminator',
                        help="train discriminator",
                        action='store_true')
    parser.add_argument('-t','--test',
                        help="test glcic",
                        action='store_true')
    parser.add_argument('-i','--interp',
                        help="interpolation data",
                        action='store_true')
    args = parser.parse_args()

    #GPUの指定
    #nvidia-smiコマンドでデバイスの確認
    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(
            visible_device_list = "2",
            allow_growth = True
        )
    )

    K.set_session(tf.Session(config=config))

    # 学習結果の保存先
    c_save_path = os.path.join(cons['SAVE_PATH'], 'completion_net')
    d_save_path = os.path.join(cons['SAVE_PATH'], 'discriminator')

    #結果の保存先ファイルを作成する
    os.makedirs(c_save_path,exist_ok=True)
    os.makedirs(d_save_path,exist_ok=True)

    # 重みの保存先
    c_weight_path = os.path.join(c_save_path,'Completion_net.hdf5')

    # 画像データの読み込み
    image_data = png_to_depth(cons['TRAIN_DATA_PATH'])
    image_data = image_data.astype(np.float32)
    data_max = np.max(image_data)

    # テストデータの読み込み
    test_data = png_to_depth(cons['TEST_DATA_PATH'])
    test_data = test_data.astype(np.float32)
    test_max = np.max(test_data)

    # scailingするための変数
    half = max(data_max,test_max)/2.0
    cons['half'] = half

    slit_masks = png_to_mask(cons['MASK_DATA_PATH'])

    # モデルを取得
    net = Network(slit_masks, cons)
    netC,c_train = net.build_generator_with_own_loss()
    netD = net.build_discriminator_with_own_loss()

    # from util.output_result import draw_heatmap
    # draw_heatmap(cons['SAVE_PATH'],'test',np.squeeze(real_data,axis=-1))

    ########
    # 学習  #
    ########
    if args.completion_net or args.discriminator:
        # 訓練データにスリットが入った画像を作成
        # スリットの入ったバイナリマスクの取得 穴の部分が1
        obj_masks = object_mask(image_data.copy()) # (n,H,W,channel)
        train = Train(netC, netD,c_train,\
                        image_data ,slit_masks,obj_masks,\
                            cons)

    # オプションに応じて学習
    if args.completion_net:
        # 重みがある場合は重みの読み込み
        if os.path.exists(cons['C_TRAINED_WEIGHT']):
            netC.load_weights(cons['C_TRAINED_WEIGHT'])

        train.train_completion_net(c_save_path)

    if args.discriminator:
        #重みの読み込み
        netC.load_weights(c_weight_path)

        if os.path.exists(cons['D_TRAINED_WEIGHT']):
            netD.load_weights(cons['D_TRAINED_WEIGHT'])

        glcic = net.build_combined_model(netC,netD,cons['BATCH_SIZE'])
        train.train_discriminator(d_save_path,glcic)

    ##############
    # モデルの評価 #
    ##############
    if args.test:
        if os.path.exists(cons['C_TRAINED_WEIGHT']):
            netC.load_weights(cons['C_TRAINED_WEIGHT'])
        else:
            raise FileNotFoundError('not exist completion net weight')

        obj_masks = object_mask(test_data.copy()) # (n,H,W,channel)
        train_data = test_data * (1-slit_masks[:1])
        fake_data = netC.predict([train_data,obj_masks])
        eval_m = EvalModel(cons['SAVE_PATH'], test_data, fake_data, obj_masks)
        eval_m.complement_data()
        eval_m.draw_abs_error_map()
        print(eval_m.mean_depth_error())
        eval_m.make_xyz_file()

    ####################
    # 実際の測定データ復元 #
    ####################
    if args.interp:
        # 実際の測定されたデータの読み込み
        # すでに縦線検出せれた状態なので、スリットマスクはない
        observed_data = png_to_depth(cons['REAL_DATA'])
        observed_data = observed_data.astype(np.float32)
        observed_data_max = np.max(observed_data)

        cons['half'] = observed_data_max/2.0

        # モデルを取得
        net = Network(None, cons)
        netC,_ = net.build_generator_with_own_loss()
        obj_masks = make_detected_mask(np.squeeze(observed_data.copy()))

        netC.load_weights(cons['C_TRAINED_WEIGHT'])

        reconstructed_data = netC.predict([observed_data,obj_masks])
        reconst_path = os.path.join(cons['SAVE_PATH'],'reconst_data')
        eval_m = EvalModel(reconst_path, observed_data, reconstructed_data, obj_masks)
        eval_m.complement_data()
        eval_m.make_xyz_file()
