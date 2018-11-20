import cv2
import numpy as np

image = cv2.imread('img/raposa.png')

kernel = np.ones((6, 6), np.float32) / 25
filter2D = cv2.filter2D(image, -1, kernel)
blur = cv2.blur(image, (20, 20))
gaussianBlur = cv2.GaussianBlur(image, (5, 5), 0)
median = cv2.medianBlur(image, 5)
bilateralFilter = cv2.bilateralFilter(image, 9, 75, 75)
shift = cv2.pyrMeanShiftFiltering(image, 80, 35)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(filter2D, 'Filter 2D', (45, 320), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
cv2.putText(blur, 'Blur', (80, 320), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
cv2.putText(gaussianBlur, 'Gaussian Blur', (10, 320), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
cv2.putText(median, 'Median', (60, 320), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
cv2.putText(bilateralFilter, 'Bilateral Filter', (10, 320), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
cv2.putText(shift, 'MeanShift Filter', (10, 320), font, 2, (255, 0, 0), 3, cv2.LINE_AA)

temporariaImage1 = np.concatenate((shift, median, gaussianBlur), axis=1)
temporariaImage2 = np.concatenate((bilateralFilter, blur, filter2D), axis=1)
finalImage = np.concatenate((temporariaImage1, temporariaImage2), axis=0)

cv2.imwrite('img/filters.jpg',finalImage)