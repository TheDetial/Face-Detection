# -*- coding: utf-8 -*-
import cv2
import dlib
import sys
import numpy as np
# 人脸框和关键点检测
def face_dect(image):
	img = cv2.imread(image)
	# 1.检测人脸框
	detector = dlib.get_frontal_face_detector()
	dets = detector(img, 1)
	for i, d in enumerate(dets):
		print("Detection {}: left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
		left = d.left()
		top = d.top()
		right = d.right()
		bottom = d.bottom()
		cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
	print("Number of faces detected: {}".format(len(dets)))
	# 2.检测人脸5或68关键点
	predictor_path = './shape_predictor_5_face_landmarks.dat'
	predictor = dlib.shape_predictor(predictor_path)
	for k, d in enumerate(dets):
		shape = predictor(img, d)
		landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
		print("face_landmark: ")
		print(landmark.shape)
		print(landmark)
		for idx, point in enumerate(landmark):
			pos = (point[0, 0], point[0, 1])
			cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.3, color=(0, 255, 0))
	cv2.imwrite('img1_landmarks5.jpg', img)

image_path = 'img1.jpg'
face_dect(image_path)