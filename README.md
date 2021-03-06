# Fine-tuning & prediction script using keras with Python3
Kerasを使ってVGG16モデルを元にFine-tuning

* fine_tuning_keras.py  
kerasのVGGモデルをベースにdataset内の学習データをFine-tuningして、model(.h5)・labelファイルを出力

* prediction.py  
上で生成したmodel・labelファイルを使って入力画像・ディレクトリ(画像を格納)を識別


以下の記事で公開しているコードを改良。  
> 人工知能に関する断創録 - VGG16のFine-tuningによる17種類の花の分類
> <http://aidiary.hatenablog.com/entry/20170131/1485864665>

* 識別クラスやサンプル数などの各種パラメータを学習データのディレクトリ構造から取得
* 識別クラスを記述したラベルファイルを出力
* 各種パラメータはオプションやコマンドライン引数で指定できるように
* 重みだけでなくモデル構造も含めて出力する仕様に


## 使い方
### kerasの実行環境を用意

```bash
pip install Pillow
pip install h5py
pip install keras
```

### ディレクトリ構造
学習データと検証データをdatasetディレクトリ内に配置。
各クラスのディレクトリ名をクラス名として扱う。

画像ファイル名に指定はありません。
好きなフォーマットで大丈夫です。

```
.
├── README.md
├── dataset
│   ├── test_images
│   │   ├── class1
│   │   │   ├── class1_1.jpg
│   │   │   ├── class1_2.jpg
│   │   │   ├── class1_3.jpg
│   │   │    ...
│   │   ├── class2
│   │   │   ├── class2_1.jpg
│   │   │   ├── class2_2.jpg
│   │   │   ├── class2_3.jpg
│   │   │    ...
│   │   ├── class3
│   │   │   ├── class3_1.jpg
│   │   │   ├── class3_2.jpg
│   │   │   ├── class3_3.jpg
│   │   │    ...
│   │    ...
│   └── train_images
│        ├── class1
│        │   ├── class1_51.jpg
│        │   ├── class1_52.jpg
│        │   ├── class1_53jpg
│        │    ...
│        ├── class2
│        │   ├── class2_51.jpg
│        │   ├── class2_52.jpg
│        │   ├── class2_53jpg
│        │    ...
│        ├── class3
│        │   ├── class3_51.jpg
│        │   ├── class3_52.jpg
│        │   ├── class3_53jpg
│        │    ...
│         ...
├── fine_tuning_keras.py
└── prediction.py
```

### 実行

#### fine_tuning_keras.py

```bash
python fine_tuning_keras.py [options]
```

##### Options

| Option                | Default parameter                  |
|:----------------------|:-----------------------------------|
| --train_data_path     | `./dataset/train_images/`          |
| --val_data_path       | `./dataset/test_images/`           |
| --output_path         | `./results/`               |
| --output_model_name   | `my-finetuning-model_<date>.h5`    |
| --output_label_name   | `my-finetuning-label_<date>.txt`    |
| --output_history_name | `my-finetuning-history_<date>.txt`  |

##### output label format
改行区切りでクラスを記述したtxtファイル

```
$ cat label.txt
class1
class2
class3
```



#### prediction.py

```bash
python predict.py <model file> <label file> <target file/directory>
```

認識対象がディレクトリの場合は、ディレクトリ構造を学習時の`test_images`と同様にする。
