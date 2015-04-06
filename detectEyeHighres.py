# this one is for detecting better iris using high res eye image from webcam

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import cv2.cv as cv

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.9/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.9/share/OpenCV/haarcascades/haarcascade_eye.xml')

#imgpath = '/Users/z001c3k/work/lens/testImages/'
#for phone
#nr = 640
#nc = 480
scaleDownFactor = 1.5
c = 0
#for i in [26]:
cap = cv2.VideoCapture(0)
ret, b_img = cap.read()
siz = np.shape(b_img)
nr =int(siz[0]/scaleDownFactor)
nc = int(siz[1]/scaleDownFactor)
while(True):
    c = c+1
    print c
    ret, b_img = cap.read()
    #b_img = cv2.imread(imgpath + fname)
    #b_img = cv2.imread(imgpath + flist[i])
    img = cv2.resize(b_img, (nc, nr))
    gray_hres = cv2.cvtColor(b_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    allfaces = face_cascade.detectMultiScale(gray, 1.1, 5)
    s = np.shape(allfaces)
    if s[0]<1:
        print 'no faces detected'
    else:
        if s[0]==1:
            face = allfaces
        else:
            for i in np.arange(s[0]):
                faceFraction = float(allfaces[i,2]*allfaces[i,3])/nr*nc
                if (faceFraction>0.2*nc*nr) and (faceFraction<0.8*nc*nr):
                    face = allfaces[i,:] # this ensures there is only one face if at all detected
                    break        
        (x,y,w,h) = np.squeeze(face)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        (xhres,yhres,whres,hhres) = (int(scaleDownFactor*x),int(scaleDownFactor*y),int(scaleDownFactor*w),int(scaleDownFactor*h)) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        if np.shape(eyes)[0] == 0:
            print 'eye not detected'
        else:
            for (ex,ey,ew,eh) in eyes:
                eyecx = ey+eh/2
                if np.squeeze([eyecx>0.2*h] and [eyecx<0.5*h]):
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    eye_det = roi_color[ey:ey+eh, ex:ex+ew]
                    (exhres,eyhres,ewhres,ehhres) = (int(scaleDownFactor*ex),int(scaleDownFactor*ey),int(scaleDownFactor*ew),int(scaleDownFactor*eh))
                    eye_det_gray_hres = gray_hres[yhres+eyhres:yhres+eyhres+ehhres, xhres+exhres:xhres+exhres+ewhres]
                    circles = cv2.HoughCircles(eye_det_gray_hres,cv.CV_HOUGH_GRADIENT,4,6,param1=70,param2=35,minRadius=int(ewhres/6.6),maxRadius=int(ewhres/5))
                    a = None
                    if (type(circles) == type(a)):
                        print ('one iris not detected')                            
                    else:
                        cv2.circle(eye_det, (np.float32(circles[0,0,0]/scaleDownFactor), np.float32(circles[0,0,1]/scaleDownFactor)), np.float32(circles[0,0,2]/scaleDownFactor), (255,0,0), 2)                        
                    #cv2.circle(eye_det, (circles[0,0,0], circles[0,0,1]), 3, (0,0,255), 2)
                else:
                    print 'bad eye detected'
    
    cv2.imshow('frame',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
################################################################################
    # Capture frame-by-frame    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()