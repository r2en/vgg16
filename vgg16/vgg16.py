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
