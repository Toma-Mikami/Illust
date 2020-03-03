import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\Popono\PycharmProjects\Illust\face.jpg")
cv2.imwrite('face2.jpg', img)

#このcvtColorで変換ができます
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()