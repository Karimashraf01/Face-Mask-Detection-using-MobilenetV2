import numpy as np
import cv2
from keras.models import load_model
import time

## Prepare image to be loaded to the model
def preprocessing(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img=cv2.resize(img,(128,128))
	img = img/255
	img=np.reshape(img,[1,128,128,3])
	return img


def get_className(classNo):
	if classNo==0:
		return "Mask"
	elif classNo==1:
		return "No Mask"

## Loading Haarcascade
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
THRESHOLD=0.90
font=cv2.FONT_HERSHEY_COMPLEX


## Loading Our MobilenetV2 Model
model = load_model('masknet.h5')

## Prediction Function
def predict(imgOrignal):
    faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
    for x,y,w,h in faces:
        crop_img=imgOrignal[y:y+h,x:x+h]
        img=preprocessing(crop_img)
        prediction=model.predict(img)
        classIndex=model.predict_classes(img)
        probabilityValue=np.amax(prediction) ## Talking Highest prediction
        if (probabilityValue>THRESHOLD):       ## Checking Condfidence against the Threshold
            if (classIndex==0):              ## if prediction is MASK
                cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
                cv2.putText(imgOrignal,f"{str(get_className(classIndex))} {str(np.round(probabilityValue*100,2))}%",(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
            elif (classIndex==1):           ## if prediction is MASK
                cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
                cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
                cv2.putText(imgOrignal,f"{str(get_className(classIndex))} {str(np.round(probabilityValue*100,2))}%",(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
    return imgOrignal



if __name__ == '__main__':
    cap=cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        start = time.time()
        _ , imgOrignal=cap.read()
        imgOrignal = predict(imgOrignal)
        fps = 1.0 / (time.time() - start)
        print("FPS: %.2f" % fps)
        fps = int(fps)
        fps = str(fps)
        cv2.putText(imgOrignal, fps + " fps", (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("Result",imgOrignal)
        k=cv2.waitKey(1)
        if k==27:           ## PRESS Esc to triminate the program
            break
    cap.release()
    cv2.destroyAllWindows()
