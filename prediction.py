#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction script using keras with python3
Reference: http://aidiary.hatenablog.com/entry/20170131/1485864665
"""

__author__ = "Haruyuki Ichino <mail@icchi.me>"
__version__ = "1.0"
__date__    = "2017/08/21"

import os
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import argparse
import glob
import sys
import numpy as np

from fine_tuning_keras import get_classes_img_count


# Command line arg setting
if len(sys.argv) != 3:
    print("Usage: python predict.py [target file path] [model path]")
    sys.exit(1)
target_path = sys.argv[1]
model_path = sys.argv[2]
print('Target file:', target_path)
print('Model file:', model_path)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path",
    "-d",
    type=str,
    default="./dataset/train_images/",
    help="Directory path of train dataset"
)
FLAGS, unparsed = parser.parse_known_args()


# Set parameters
classes, _ = get_classes_img_count(FLAGS.dataset_path)
nb_classes = len(classes)

# 入力画像のサイズはモデルの学習時のサイズと統一
img_height, img_width = 150, 150
channels = 3


# Load model
model = load_model(model_path)

# 画像を読み込んで4次元テンソルへ変換
img = image.load_img(target_path, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！
# これを忘れると結果がおかしくなるので注意
x = x / 255.0

# print(x)
# print(x.shape)

# クラスを予測
# 入力は1枚の画像なので[0]のみ
pred = model.predict(x)[0]

# 予測確率が高いトップ5を出力
top = 5
top_indices = pred.argsort()[-top:][::-1]
result = [(classes[i], pred[i]) for i in top_indices]
for x in result:
    print(x)
