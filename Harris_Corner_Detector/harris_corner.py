import cv2
import numpy as np

img = cv2.imread('img/exemple.png')
imgOrigin = cv2.imread('img/example.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

dst = cv2.dilate(dst,None)

img[dst>0.01*dst.max()]=[0,0,255]

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imgOrigin, 'Origin', (45, 320), font, 3, (255, 0, 0), 3, cv2.LINE_AA)
cv2.putText(img, 'Harris Corner', (80, 320), font, 3, (255, 0, 0), 3, cv2.LINE_AA)

finalImage = np.concatenate((imgOrigin, img), axis=0)

cv2.imwrite('img/Harris_Corner_Detector.png',finalImage)
