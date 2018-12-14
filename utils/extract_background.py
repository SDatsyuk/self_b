# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import glob
import os
import numpy as np

def capture_background(cap):
	print('Press `f` to create background image')
	while True:
		_, frame = cap.read()
		frame = frame[:,:440]


		cv2.imshow('background', frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('f'):
			back = frame
			break
	cap.release()
	cv2.destroyAllWindows()
	return frame


def extract_background(back, image):
	
	grayBase = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
	grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# compute the Structural Similarity Index (SSIM) between the two
	# images, ensuring that the difference image is returned
	(score, diff) = compare_ssim(grayBase, grayImage, full=True)
	diff = (diff * 255).astype("uint8")
	print("SSIM: {}".format(score))

	# threshold the difference image, followed by finding contours to
	# obtain the regions of the two input images that differ
	thresh = cv2.threshold(diff, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	structuringElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50, 50))
	closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, structuringElement)
	cnts = cv2.findContours(closing, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	# print(cnts[:2])

	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	# val = [boundingBoxes
	idx = 1
	# (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][idx], reverse=False))
	# cnts = [c for i, c in enumerate(cnts) if boundingBoxes[i][2] > 10]
	# print(sorted(zip(cnts, boundingBoxes), key=lambda c: c[idx][2])[0])
	if cnts:
		(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
			key=lambda b: b[1][idx], reverse=False))
	# print(cnts)
	# cnts = [cnts[0]]
	# boundingBoxes = [boundingBoxes[0]]
	# im2, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.imshow("Original", image)
	cv2.imshow("Modified", back)
	cv2.imshow("Diff", diff)
	cv2.imshow("Thresh", thresh)

	return cnts[0]

	# loop over the contours
	# for c in cnts:
	# 	# compute the bounding box of the contour and then draw the
	# 	# bounding box on both input images to represent where the two
	# 	# images differ
	# 	(x, y, w, h) = cv2.boundingRect(c)
	# 	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	# 	# cv2.rectangle(base, (x, y), (x + w, y + h), (0, 0, 255), 2)
	 
	# # show the output images
	
	# key = cv2.waitKey(1) & 0xFF
	# if key == 27:
	# 	break
	# # base = image