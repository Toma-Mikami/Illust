import os
import cv2

# 特徴量ファイルをもとに分類
classifier = cv2.CascadeClassifier(r"C:\Users\Popono\PycharmProjects\Illust\lbpcascade_animeface-master\lbpcascade_animeface.xml")

# 顔の検出
image = cv2.imread(r"C:\Users\Popono\PycharmProjects\Illust\IData\EQpZoyfUwAYbj_1.jpg")

# グレースケールで処理を高速化
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces = classifier.detectMultiScale(gray_image)

print(faces)

# ディレクトリを作成
output_dir = 'faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i, (x,y,w,h) in enumerate(faces):
    # 一人ずつ顔を切り抜く
    face_image = image[y:y+h, x:x+w]
    output_path = os.path.join(output_dir, '{0}.jpg'.format(i))
    cv2.imwrite(output_path,face_image)

cv2.imwrite('face2.jpg',image)