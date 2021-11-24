import sys
sys.path.append("../")

import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("../dataset/labelled/black_caps/1.jpg")
annotation = np.load("../dataset/labelled/black_caps/1.npy")
cv2.imwrite("image_sample.png", image)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print(annotation.shape)
print(annotation.squeeze())

for (x,y,d) in annotation.squeeze():
    labelled_image = cv2.circle(image, (int(x), int(y)), int(d), (0,0,255), -1)

while True:
    cv2.namedWindow("Labelled Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Labelled Image", 2048, 1536)
    cv2.imshow("Labelled Image", labelled_image)
    cv2.imwrite("labelled_image_sample.png", labelled_image)
    k = cv2.waitKey(400)

    if k == 27:
        cv2.destroyAllWindows()
        break