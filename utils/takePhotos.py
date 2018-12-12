import cv2
import argparse
import uuid
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="images", help="path to store image")

args = vars(ap.parse_args())

def main():

	if not os.path.exists(args['output']):
		os.makedirs(args['output'])

	camera = cv2.VideoCapture(0)
	while True:
		res, frame = camera.read()

		cv2.imshow('ss', frame)

		key = cv2.waitKey(30)
		if key & 0xFF == ord("f"):
			filename = str(uuid.uuid4()) + '.jpg'
			cv2.imwrite(os.path.join(args['output'], filename), frame)

		if key & 0xFF == ord("q"):
			break

if __name__ == "__main__":
	main()