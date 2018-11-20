import cv2
import numpy as np
import pymeanshift as pms

image = cv2.imread('img/example.jpg')

filter = cv2.pyrMeanShiftFiltering(image, 70, 27)
(segmented, labels, number_regions) = pms.segment(filter, spatial_radius=8,
                                                              range_radius=4.5, min_density=30)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, 'Origin', (45, 320), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
cv2.putText(filter, 'Filter', (10, 320), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
cv2.putText(segmented, 'Segmentation', (1, 320), font, 2, (255, 0, 0), 3, cv2.LINE_AA)

finalImage = np.concatenate((image, filter, segmented), axis=1)

cv2.imwrite('img/MeanShiftGirafa.jpg',finalImage)
