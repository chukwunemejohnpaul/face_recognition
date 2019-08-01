import cv2
import os
import numpy as np


net = cv2.dnn.readNetFromCaffe(os.getcwd()+"/deploy.prototxt.txt",os.getcwd()+"/res10_300x300_ssd_iter_140000.caffemodel")

name_list = []
for image in os.listdir(os.getcwd()+"/images"):
	if image.endswith("jpg") or image.endswith("jpeg"):
		name, ext = os.path.splitext(image)
		name_list.append(name)
		Frame = cv2.imread(os.getcwd()+"/images/"+image)
		print(os.getcwd()+"/images/"+image)
		if Frame.shape:
			(h,w) = Frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(Frame, (300, 300)), 1.0,(300,300), (104,177.0,123.0))
			net.setInput(blob)
			detections = net.forward()
			for i in range(0, detections.shape[2]):
				confidence = detections[0,0,i,2]
				if confidence > 0.8:
					box = detections[0,0,i,3:7] * np.array([w,h,w,h])
					startx, starty, endx, endy = box.astype("int")
					face = Frame[starty:endy,startx:endx]
					cv2.imwrite(os.getcwd()+"/faces/"+name_list[-1]+ ext,face)
					print("Read and save a face to disk")
print("___________DONE_______________")