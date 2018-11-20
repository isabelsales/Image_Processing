import numpy as np
import cv2

cap = cv2.VideoCapture('video/por-do-sol.mp4')
output_file = "video/video_klt_tracker_por_do_sol.avi"
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 3,
                       blockSize = 3 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)
y = 0
is_begin = True

while True:
    ret,frame = cap.read()
    if frame is None:
        break
    processed = frame

    if is_begin:
        h, w, _ = processed.shape
        out = cv2.VideoWriter(output_file, fourcc, 30, (w, h), True)
        is_begin = False

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(32)
    if k == 32:
        cv2.imwrite('img/p{0:05d}.jpg'.format(y), img)
        y += 1
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(old_frame)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), (0,255,0), 1)
        frame = cv2.circle(frame,(a,b),2,(0,0,255), -1)
    img = cv2.add(frame,mask)
    out.write(img)
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.imshow('', img)
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()
