import numpy as np
import cv2
import time
import classifier_module as cm

cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX

while True:
	start = time.time()
	_ , imgOrignal=cap.read()
	imgOrignal = cm.predict(imgOrignal)
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
