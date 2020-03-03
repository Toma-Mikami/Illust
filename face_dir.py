import os
import numpy as np
import glob
import cv2

# ディレクトリを作成
output_dir = 'faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# globモジュール　引数に指定されたパターンにマッチする「ファイルパス名を取得」することが出来る
# ディレクトリ内のファイルに処理を加える際に使用する

# アスタリスクはフォルダ内のファイル名が代入される
files =glob.glob(r"C:\Users\Popono\PycharmProjects\Illust\IData\*")

# 特徴量ファイルをもとに分類
classifier = cv2.CascadeClassifier(r"C:\Users\Popono\PycharmProjects\Illust\lbpcascade_animeface-master\lbpcascade_animeface.xml")


for fname in files:
    # 画像読み込み
    image = cv2.imread(fname, cv2.IMREAD_COLOR)

    # グレースケールで処理を高速化 cvtColor(画像,オプション）
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #　顔検出　detectMultiscale(画像,ベクトル,縮小量,最低矩形,フラグ,最小サイズ,最大サイズ)
    face = classifier.detectMultiScale(gray_image)

    print(face)
    if len(face) > 0:
        for i, (x,y,w,h) in enumerate(face):
            # 一人ずつ顔を切り抜く
            face_image = image[y:y+h, x:x+w]
            #output_path = os.path.join(output_dir, '{0}.jpg'.format(i))
            output_path = os.path.join(output_dir, '{0}-{1}.jpg'.format(len(fname),i))
            cv2.imwrite(output_path,face_image)
        # cv2.imwrite(fname,image)

cv2.destroyAllWindows()