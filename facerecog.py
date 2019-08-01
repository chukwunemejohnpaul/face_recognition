import cv2
import numpy as np
import os
from scipy.spatial.distance import cosine
os.environ['KERAS_BACKEND']='theano'
from keras_vggface.utils import preprocess_input
import keras_vggface
from keras_vggface.vggface import VGGFace



def extract_face_embeddings(path):
	i = 0
	im = len(os.listdir(path))
	images = np.zeros((im-1,224,224,3))
	for image in os.listdir(path):
		if image.endswith("jpg") or image.endswith("jpeg"):
			image = cv2.imread(path + "/" + image)
			image = np.resize(image,(224,224,3))
			images[i] = np.array(image)
			i+=1
	samples = images.astype("float32")
	print("samples shape " ,samples.shape)
	samples = preprocess_input(samples, version=2)
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
	preds = model.predict(samples)
	return preds





def get_names(path):
	name_list = []
	for image in os.listdir(os.getcwd()+"/faces"):
		if image.endswith("jpg") or image.endswith("jpeg"):
			name, ext = os.path.splitext(image)
			name_list.append(name)
	return name_list



def reco_face(face,embeddings,model):
	scores = []
	images = np.zeros((1,224,224,3))
	face = np.resize(face,(224,224,3))
	images[0] = face
	faces = preprocess_input(images, version=2)
	print(faces.shape)
	pred = model.predict(faces)
	print("prediction ", pred)
	# calculate distance between embeddings
	num = 0
	for candidate_embedding in embeddings:
		print("candidate embeddding ", candidate_embedding)
		score = cosine(pred, candidate_embedding)
		scores.append(score)
		f = np.argmax(scores)
		scores = sorted(scores,reverse=True)
		num += 1
	return scores[0], f

name_list = get_names(os.getcwd()+"/faces")
print("name list ", name_list)
embeddings = extract_face_embeddings(os.getcwd()+"/faces")
net = cv2.dnn.readNetFromCaffe(os.getcwd()+"/deploy.prototxt.txt",os.getcwd()+"/res10_300x300_ssd_iter_140000.caffemodel")
model = VGGFace(model="resnet50",include_top=False,input_shape=(224,224,3))
video = cv2.VideoCapture(0)

print("embeddings")
print(embeddings)
while True:
	ret, Frame = video.read()
	if ret:
		(h,w) = Frame.shape[:2]
		blob  = cv2.dnn.blobFromImage(cv2.resize(Frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
		net.setInput(blob)
		detections = net.forward()
		for i in range(0, detections.shape[2]):
			confidence = detections[0,0,i,2]
			if confidence > 0.8:
				box  = detections[0,0,i,3:7] * np.array([w,h,w,h])
				(startx, starty, endx, endy) = box.astype("int")
				face = Frame[starty:endy,startx:endx]
				score, num = reco_face(face,embeddings,model)
				
				print(score)
				print("num ", num)
				if score <= 0.5:
					name = name_list[num]
				else:
					name = "unknown"
				cv2.rectangle(Frame, (startx, starty), (endx, endy),(0, 0, 255), 2)
				cv2.putText(Frame, name, (startx, starty),cv2.FONT_HERSHEY_SIMPLEX, 1.00, (0, 0, 255), 2)
		cv2.imshow("video",Frame)
		k = cv2.waitKey(1)
		if k == ord("q") & 0xff:
			break
video.release()
cv2.destroyAllWindows()