import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import cv2.cv as cv

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.9/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.9/share/OpenCV/haarcascades/haarcascade_eye.xml')

#imgpath = '/Users/z001c3k/work/lens/testImages/'
nr = 480
nc = 853
#for phone
#nr = 640
#nc = 480

c = 0
#for i in [26]:
cap = cv2.VideoCapture(0)

while(True):
    c = c+1
    print c
    ret, b_img = cap.read()
    #b_img = cv2.imread(imgpath + fname)
    #b_img = cv2.imread(imgpath + flist[i])
    img = cv2.resize(b_img, (nc, nr))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    allfaces = face_cascade.detectMultiScale(gray, 1.3, 2)
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
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if np.shape(eyes)[0] == 0:
            print 'eye not detected'
        else:
            for (ex,ey,ew,eh) in eyes:
                eyecx = ey+eh/2
                if np.squeeze([eyecx>0.2*h] and [eyecx<0.5*h]):
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)                
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