#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction script using keras with python3
Reference: http://aidiary.hatenablog.com/entry/20170131/1485864665
"""

__author__ = "Haruyuki Ichino <mail@icchi.me>"
__version__ = "2.0"
__date__    = "2017/09/09"

import os
from keras.models import load_model
from keras.preprocessing import image
from sklearn import metrics
import glob
import sys
import numpy as np



# Command line arg setting
if len(sys.argv) != 4:
    print("Usage: python predict.py <model file> <label file> <target file/directory>")
    sys.exit(1)
model_file = sys.argv[1]
label_file = sys.argv[2]
target = sys.argv[3]
print('Model file:', model_file)
print('Lodel file:', label_file)
print('Target:', target)


def get_classes_from_label_file(file):
    try:
        f = open(file)
        classes = f.read().strip("\n").split("\n")
        f.close()
    except:
        print("Error: Not found label file.", file)
        sys.exit(0)

    return classes

def imgpath2imgarray(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)

    # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要
    img_array = img_array / 255.0

    return img_array

def get_images_and_label(path, label_str):
    images = []
    labels = []

    for tclass in sorted(os.listdir(path)):
        # .DS_Storeのチェック
        if tclass == ".DS_Store":
            continue

        class_path = os.path.join(path, tclass)

        # ディレクトリじゃない場合はスキップ
        if not os.path.isdir(class_path):
            continue

        image_paths = np.sort(glob.glob(os.path.join(class_path, '*.*[gG]')))
        class_image_count = len(image_paths)

        for image_path in image_paths:
            # get image data
            image_array = imgpath2imgarray(image_path)
            images.append(image_array)

        labels += [label_str.index(tclass)] * class_image_count

    return np.array(images), labels

def predict_classes(model, images):
    pred_labels = []
    results = model.predict(images, verbose=1)
    for result in results:
        pred_labels.append(np.argmax(result))

    return pred_labels

# Set parameters
classes = get_classes_from_label_file(label_file)
nb_classes = len(classes)
print("classes:", nb_classes)
print(classes)

# 入力画像のサイズはモデルの学習時のサイズと統一
img_height, img_width = 224, 224
channels = 3


# Load model
model = load_model(model_file)

# ファイルの場合、各クラスの確率を出力
if os.path.isfile(target):
    # 画像を読み込んで4次元テンソルへ変換
    image = imgpath2imgarray(target)
    # print(image)
    print(target, image.shape)

    # クラスを予測
    images = np.expand_dims(image, axis=0)
    pred = model.predict(images)[0]

    # 予測確率が高いトップ5を出力
    top = 10
    top_indices = pred.argsort()[-top:][::-1]
    results = [(classes[i], pred[i]) for i in top_indices]
    for result in results:
        print(result)

# ディレクトリの場合、混合行列を表示
elif os.path.isdir(target):

    images, labels = get_images_and_label(target, classes)
    print(images.shape)
    print("label:", len(labels))

    # predict
    pred_labels = predict_classes(model, images)
    print(pred_labels)

    # Output result
    print("classification report:")
    print(metrics.classification_report(labels, pred_labels, target_names=classes))
    print("confusion matrix:")
    print(metrics.confusion_matrix(labels, pred_labels))

    pass

else:
    print("Error: Wrong target type. Set image file or directory.")
