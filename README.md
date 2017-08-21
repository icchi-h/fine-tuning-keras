# Fine-tuning & prediction script using keras with Python3

* fine_tuning_keras.py  
kerasのVGGモデルをベースにdataset内の学習データをFine-tuningして、modelファイル(.h5)を出力

* prediction.py  
上で生成したmodelを使って入力画像を識別


以下の記事で公開しているコードを改良。  
> 人工知能に関する断創録 - VGG16のFine-tuningによる17種類の花の分類
> <http://aidiary.hatenablog.com/entry/20170131/1485864665>

* 識別クラスやサンプル数などの各種パラメータを学習データのディレクトリ構造から取得


## 使い方
### kerasの実行環境を用意

```bash
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
# sample
python fine_tuning_keras.py --output_model_name my-finetuning-model.h5
```

##### Options

| Option              | Default parameter         |
|:--------------------|:--------------------------|
| --train_data_path   | `./dataset/train_images/` |
| --val_data_path     | `./dataset/test_images/`  |
| --output_path       | `./dataset/results/`      |
| --output_model_name | `my-finetuning-model.h5`  |



#### prediction.py

```bash
python prediction.py <target image> <model file>

# e.g.
python prediction.py ~/hoge.png dataset/results/my-finetuning-model.h5
```

##### Options

| Option              | Default parameter         |
|:--------------------|:--------------------------|
| --dataset_path      | `./dataset/train_images/` |
