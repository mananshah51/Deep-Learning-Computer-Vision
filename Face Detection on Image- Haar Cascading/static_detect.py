#The above Code is taken from docs.opencv.org
#The Detection is based on Haar Cascades

#OpenCV Program to detect faces
#import OpenCV Library where its functionality resides
import cv2

#The Features that need to be extracted from the image are present in the xml file
#The Cascade classifiers have been trained with a lot of postiive and negative images
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#Reading img from the folder we need to perform face detection
img = cv2.imread('test.jpg')

#Converting the orginal image to Gray image because we are detecting face
#and we not detecting colors

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Output the Gray Image that is converted 
cv2.imshow('Gray Image -Test',gray)

#Face Detection

faces = face_cascade.detectMultiScale(gray,1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),4)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[x:y+h, y:x+w]

    
cv2.imshow('Face Detected Image',img)
cv2.waitKey(25000)
cv2.destroyAllWindows()

