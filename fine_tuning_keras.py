#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tuning script using keras with python3
Reference: http://aidiary.hatenablog.com/entry/20170131/1485864665
"""

__author__ = "Haruyuki Ichino <mail@icchi.me>"
__version__ = "1.1"
__date__    = "2017/08/21"

import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, callbacks
import argparse
import glob
import sys
from datetime import datetime


#from smallcnn import save_history
def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))

def get_classes_img_count(path):
    """
    引数のディレクトリ内にあるディレクトリ(クラス)のlistと各list内に含まれる画像ファイルの合計を返す関数
    """

    # get classes
    classes = []
    for item in os.listdir(path):
        if os.path.isdir(path + item):
            classes.append(item)

    # get img count each class
    img_counts = []
    for c in classes:
        images = glob.glob(path + c + '/*.*g')
        img_counts.append(len(images))

    return classes, img_counts

def get_image_size(path):
    """
    引数のディレクトリ内で最初に見つかった画像ファイルのサイズを返却する関数
    """

    images = glob.glob(path + '*.*g', recursive=True)


if __name__ == '__main__':

    # Command line arg setting
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        "-t",
        type=str,
        default="./dataset/train_images/",
        help="Directory path of training data"
    )
    parser.add_argument(
        "--val_data_path",
        "-v",
        type=str,
        default="./dataset/test_images/",
        help="Directory path of validation data"
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default="./dataset/results/",
        help="Output path"
    )
    parser.add_argument(
        "--output_model_name",
        type=str,
        default="my-finetuning-model.h5",
        help="Output model name"
    )
    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.exists(FLAGS.output_path):
        os.mkdir(FLAGS.output_path)


    # Set Parameter
    classes, train_img_counts = get_classes_img_count(FLAGS.train_data_path)
    _, val_img_counts = get_classes_img_count(FLAGS.val_data_path)
    nb_classes = len(classes)
    nb_train_samples = sum(train_img_counts)
    nb_val_samples = sum(val_img_counts)
    nb_epoch = 50
    nb_epoch = 2
    min_nb_epoch = 10
    batch_size = 50
    batch_size = 1
    if (batch_size > nb_train_samples):
        print("Error: バッチサイズが学習サンプル数よりも大きくなっています")
        sys.exit(1)

    img_rows, img_cols = 150, 150
    channels = 3

    start_time = datetime.now().strftime("%Y%m%d%H%M%S")


    # Set callbacks
    cb_earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=min_nb_epoch)
    # cb_tensorboard = callbacks.TensorBoard(
    #     log_dir=FLAGS.output_path+'tb-logs'+start_time,
    #     histogram_freq=1,
    #     write_graph=True,
    #     write_grads=True,
    #     write_images=True,
    #     embeddings_freq=1,
    #     embeddings_layer_names=None,
    #     embeddings_metadata=None)

    # VGG16モデルと学習済み重みをロード
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    input_tensor = Input(shape=(img_rows, img_cols, channels))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    # vgg16.summary()

    # FC層を構築
    # Flattenへの入力指定はバッチ数を除く
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # VGG16とFCを接続
    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    # 最後のconv層の直前までの層をfreeze
    for layer in model.layers[:15]:
        layer.trainable = False

    # Fine-tuningのときはSGDの方がよい？
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        FLAGS.train_data_path,
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        FLAGS.val_data_path,
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    # Fine-tuning
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        callbacks=[cb_earlystopping])
        # callbacks=[cb_earlystopping, cb_tensorboard])

    # Output model
    # model.save_weights(os.path.join(FLAGS.output_path, 'my-finetuning.h5'))
    model.save(os.path.join(FLAGS.output_path, FLAGS.output_model_name))
    print("Saved: model file '" + (FLAGS.output_path+FLAGS.output_model_name) + "'")

    # Output history
    history_name_epoch = 'history_finetuning' + start_time + '_epoch.txt'
    save_history(history, os.path.join(FLAGS.output_path, history_name_epoch))
    print("Saved: learning log file ", history_name_epoch)
