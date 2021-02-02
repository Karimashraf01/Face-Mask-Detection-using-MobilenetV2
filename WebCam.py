import numpy as np
import cv2
from keras.models import load_model

## Prepare image to be loaded to the model
def preprocessing(img):
	img=cv2.resize(img,(128,128))
	img = img/255
	img=np.reshape(img,[1,128,128,3])
	return img


def get_className(classNo):
	if classNo==0:
		return "Mask"
	elif classNo==1:
		return "No Mask"

## Preparing Camera and Loading Haarcascade
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
THRESHOLD=0.90
cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX

## Loading Our MobilenetV2 Model
model = load_model('masknet.h5')

## Opening webcam and loading the model
while True:
	_ , imgOrignal=cap.read()
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	for x,y,w,h in faces:
		crop_img=imgOrignal[y:y+h,x:x+h]
		img=preprocessing(crop_img)
		prediction=model.predict(img)
		classIndex=model.predict_classes(img)
		probabilityValue=np.amax(prediction) ## Taking Highest prediction
		if (probabilityValue>THRESHOLD):       ## Checking Condfidence against the Threshold
			if (classIndex==0):              ## if prediction is MASK
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
				cv2.putText(imgOrignal,f"{str(get_className(classIndex))} {str(np.round(probabilityValue*100,2))}%",(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif (classIndex==1):           ## if prediction is NOMASK
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
				cv2.putText(imgOrignal,f"{str(get_className(classIndex))} {str(np.round(probabilityValue*100,2))}%",(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
	cv2.imshow("Result",imgOrignal)
	k=cv2.waitKey(1)
	if k==27:           ## PRESS Esc to triminate the program
		break
cap.release()
cv2.destroyAllWindows()
