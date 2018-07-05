# VGG16モデルを動かす

## ILSVRC(ImageNet LargeScale Visual Recognition Challenge)
ディープラーニングが現在のように注目を集めるようになったきっかけが2012年に開催された大規模画像認識コンペティションのILSVRCである。その年のコンペでAlexNetが圧倒的な成績を残したことからディープラーニングが主役に躍り出た。

#### ImageNet
ImageNetは100万枚を超える画像のデータセットであり多種多様な種類の画像が含まれている。その画像一枚一枚にラベル(クラス名)が紐づけられている。この巨大なデータセットを用いて、ILSVRCという画像認識コンペティションが毎年開催される。コンペティションにはいくつかのテスト項目があり、その一つが「クラス分類(1000クラスの分類を行なって認識精度を競う)」である。

## VGG
VGG16は2014年度のILSVRCの大会で優勝した畳み込み13層とフル結合3層の計16層から成る畳み込みニューラルネットワークである。層が多いだけで一般的な畳み込みニューラルネットワークと大きな違いがなく、シンプルでわかりやすいのが特徴。ImageNetを使って訓練したモデルが公開されている。

VGGの出力層は1000ユニットあり、1000クラスを分類するニューラルネットである。1000クラスのリストは[1000 synsets for Task 2](http://image-net.org/challenges/LSVRC/2014/browse-synsets)にある。

### kerasのVGG16モデル
kerasではVGG16モデルがkeras.applications.vgg16モジュールに実装されていて簡単に使用可能。ImageNetの大規模画像セットで学習済みのモデルな為、自分で画像を集めて学習する必要がない。

```python
from keras.applications.vgg16 import VGG16
model = VGG16(include_top=True, weights='imagenet',
              input_tensor=None, input_shape=None)
```

VGG16クラスは4つの引数を取る

- include_topはVGG16のトップにある1000クラス分類するフル結合層（FC）を含むか含まないかを指定する。今回は画像分類を行いたいためFCを含んだ状態で使う。FCを使わない場合、VGG16を特徴抽出器として使うこともできる。

- weightsはVGG16の重みの種類を指定する。VGG16は単にモデル構造であるため必ずしもImageNetを使って学習しなければいけないわけではない。Noneにするとランダムな重みに成る為、自分で画像学習させる人間が使える。

- input_tensorは自分でモデルに画像を入力したいときに使う。後のFine-tuning時に使う。

- input_shapeは入力画像の形状を指定する。include_top=Trueにして画像分類器として使う場合は (224, 224, 3) で固定なのでNoneにする。


### 実装

PIL, Numpy, kerasがインストールされていればこのコードをそのままコピペするだけで動く

```python

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image
import numpy as np
import sys

'''
ImageNetで学習済みのVGG16モデルを使って入力画像のクラスを予測する
'''

if len(sys.argv) != 2:
    print("usage: python vgg16.py [image file]")
    sys.exit(1)

filename = sys.argv[1]

# 学習済みVGG16と学習済み重みを読み込む
model = VGG16(weights='imagenet')

# 引数で指定した画像ファイルを読み込む。サイズはVGG16にリサイズ
img = image.load_img(filename, target_size=(224, 224))

# 読み込んだPIL形式の画像をarrayに変換
x = image.img_to_array(img)

# 3次元テンソル(rows, cols, channels)を4次元テンソル(samples, rows, cols, channels)に変換
x = np.expand_dims(x, axis=0)

# 予測部分。VGG16の100クラスをdecode_predictions()で文字列に変換
preds = model.predict(preprocess_input(x))
results = decode_predictions(preds, top=5)[0]
for result in results:
    print(result)
    
```


へびっぽい画像が返ってくれば御の字だと高を括っていたら


> ('n01729322', 'hognose_snake', 0.27550206)
('n01740131', 'night_snake', 0.20833224)
('n01756291', 'sidewinder', 0.17194305)
('n01755581', 'diamondback', 0.1598247)
('n01744401', 'rock_python', 0.086609922)


ガチガチのオタク感溢れた解答がVGG16から返ってくる。
本当は日本の毒蛇の代表格、日本マムシヘビなのだけれども、北欧由来のブタハナヘビ(ホグノーズ)が一番上に来た。検索するとどちらも素人目には判断つかない。

<br>

ImageNet内の画像ではなく、完全にただのネットの画像検索したヘビの画像を入力してこの精度がでる。

|入力画像|出力画像1|出力画像2|
|---|---|---|
|![snake](https://user-images.githubusercontent.com/28590220/28920066-34a888a8-788b-11e7-9ebe-d7cb7f5a328d.jpg)|![hognose_snake](https://user-images.githubusercontent.com/28590220/28920067-34aa9206-788b-11e7-9e0e-c3ed3085296c.jpg)|![image](https://user-images.githubusercontent.com/28590220/28920175-a9ea2fc2-788b-11e7-88ec-31c644c4eba0.png)|

<br>

もちろん、ImageNetから取って来た適当な画像の認識結果を見てもこの精度の高さを誇る。


>('n02009912', 'American_egret', 0.9778074)
('n02012849', 'crane', 0.020496963)
('n02009229', 'little_blue_heron', 0.0014358116)
('n02007558', 'flamingo', 0.00011946301)
('n02006656', 'spoonbill', 5.8570407e-05)

<br>

|入力画像|出力画像|
|---|---|
|![american_egret](https://user-images.githubusercontent.com/28590220/28920347-570f5894-788c-11e7-8392-6f52b3cc4ece.gif)|![american-egret](https://user-images.githubusercontent.com/28590220/28920348-571fd5a2-788c-11e7-88f7-9ab80ee6509d.jpg)|

<br>

しかし、VGG16を学習した際に選ばれた1000クラス以外はどうやっても認識されない。そのため、次はFine-tuningと言う技術で1000クラスに含まれていない画像も認識させる。



