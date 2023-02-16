#!/usr/bin/env python3

import os
import cv2
import numpy
import pyzed.sl as sl
import pandas as pd

# Open the ZED Camera
cam = sl.Camera()
image = sl.Mat()
#depth = sl.Mat()

# Initial Parameters
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD1080

init.camera_fps = 30 # Set Camera FPS
#init.depth_mode = sl.DEPTH_MODE.ULTRA # Use ultra Depth Mode
#init.coordinate_units = sl.UNIT.METER # Use m to measure Unit depth

status = cam.open(init)
if status != sl.ERROR_CODE.SUCCESS:
	print(repr(status))
	exit(1)

runtime = sl.RuntimeParameters()
#runtime.sensing_mode = sl.SENSING_MODE.STANDARD # To configure depth sensing in FILL mode, set SENSING_MODE_FILL in RuntimeParameters
print('ZED Camera is Recording, use Ctrl-C to stop.')
frames_recorded = 0

image_dir = r'D:\\ZED\\street_image\\original\\'
process_dir = r'D:\\ZED\\street_image\\preprocessed\\'

while True:
	if frames_recorded >= 15:
		exit(1)
	# Get new frame from camera
	if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:

		cam.retrieve_image(image, sl.VIEW.LEFT)
		#cam.retrieve_image(depth, sl.VIEW.DEPTH)
		
		image_ocv = image.get_data()
		#depth_ocv = depth.get_data()
		
		cv2.imwrite(os.path.join(image_dir, f'image_{frames_recorded:03}.png'), image_ocv)
		#cv2.imwrite(os.path.join(image_dir, f'depth_{frames_recorded:03}.png'), depth_ocv)
		
		dim = (2048, 1024)

		image_rs = cv2.resize(image_ocv, dim, interpolation = cv2.INTER_AREA)
		#depth_rs = cv2.resize(depth_ocv, dim, interpolation = cv2.INTER_AREA)

		cv2.imwrite(os.path.join(process_dir, f'image_{frames_recorded:03}.png'), image_rs)
		#cv2.imwrite(os.path.join(process_dir, f'depth_{frames_recorded:03}.png'), depth_rs)

		frames_recorded += 1
