from pyzbar import pyzbar
import argparse
import cv2
import sys

def open_camera(idx=0):
	return cv2.VideoCapture(idx)

def read_frame(camera):
	return camera.read()

def find_barcode(image):
	return pyzbar.decode(image)

def viz_barcodes(image, barcodes):
	for barcode in barcodes:
		(x, y, w, h) = barcode.rect
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

		barcodeData = barcode.data.decode("utf-8")
		barcodeType = barcode.type

		print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
	return image

if __name__ == "__main__":
	camera = open_camera(0)
	while True:
		status, frame = read_frame(camera)
		if not status:
			sys.exit(0)
		barcodes = find_barcode(frame)

		viz_boxes = viz_barcodes(frame.copy(), barcodes)

		cv2.imshow("ss", viz_boxes)
		
		if cv2.waitKey(30) & 0xFF == ord("q"):
			break
	
