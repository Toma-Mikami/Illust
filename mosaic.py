import cv2

src = cv2.imread(r"C:\Users\Popono\PycharmProjects\Illust\face.jpg")

def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

dst_05 = mosaic(src, ratio=0.5)
cv2.imwrite(r"C:\Users\Popono\PycharmProjects\Illust\mosaic\mosaic_05.jpg", dst_05)

dst_03 = mosaic(src, ratio=0.3)
cv2.imwrite(r"C:\Users\Popono\PycharmProjects\Illust\mosaic\mosaic_03.jpg", dst_03)

dst_01 = mosaic(src)
cv2.imwrite(r"C:\Users\Popono\PycharmProjects\Illust\mosaic\mosaic_01.jpg", dst_01)

dst_005 = mosaic(src, ratio=0.05)
cv2.imwrite(r"C:\Users\Popono\PycharmProjects\Illust\mosaic\mosaic_005.jpg", dst_005)

dst_001 = mosaic(src, ratio=0.01)
cv2.imwrite(r"C:\Users\Popono\PycharmProjects\Illust\mosaic\mosaic_001.jpg", dst_001)

