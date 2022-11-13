#!/usr/bin/env python3

import os
import cv2
import numpy
import pyzed.sl as sl

# Open the ZED Camera
cam = sl.Camera()
image = sl.Mat()
depth_fill = sl.Mat()
depth_standard = sl.Mat()
point_cloud = sl.Mat()


# Initial Parameters
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD1080

init.camera_fps = 30 # Set Camera FPS
init.depth_mode = sl.DEPTH_MODE.ULTRA # Use ultra Depth Mode
init.coordinate_units = sl.UNIT.METER # Use m to measure Unit depth

status = cam.open(init)
if status != sl.ERROR_CODE.SUCCESS:
	print(repr(status))
	exit(1)

runtime = sl.RuntimeParameters()
runtime.sensing_mode = sl.SENSING_MODE.STANDARD # To configure depth sensing in FILL mode, set SENSING_MODE_FILL in RuntimeParameters

# Setup Directories
image_dir = '/home/admin/Documents/ZED/depth/'

# Resize Images
dim = (2048, 1024)

# Get new frame from camera
if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
	print('Starting to capture images.')
	# Retrieve frame
	cam.retrieve_image(image, sl.VIEW.LEFT)
	cam.retrieve_image(depth_standard, sl.VIEW.DEPTH)
	cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

	# Make frame usable by OpenCV
	image_ocv = image.get_data()
	depth_s_ocv = depth_standard.get_data()

	# Resize images
	image_rs = cv2.resize(image_ocv, dim, interpolation = cv2.INTER_AREA)
	depth_s_rs = cv2.resize(depth_s_ocv, dim, interpolation = cv2.INTER_AREA)

	# Save Preprocessed images
	cv2.imwrite(os.path.join(image_dir, 'image.png'), image_rs)
	cv2.imwrite(os.path.join(image_dir, 'depth_standard.png'), depth_s_rs)
	point_cloud.write(os.path.join(image_dir, 'point_cloud') + '.ply')	

print('Changing depth sensing modes to fill.')
runtime.sensing_mode = sl.SENSING_MODE.FILL # To configure depth sensing in FILL mode, set SENSING_MODE_FILL in RuntimeParameters

# Get new frame from camera
if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
	
	# Retrieve frame
	cam.retrieve_image(depth_fill, sl.VIEW.DEPTH)

	# Make frame usable by OpenCV
	depth_f_ocv = depth_fill.get_data()
	
	# Resize images
	depth_f_rs = cv2.resize(depth_f_ocv, dim, interpolation = cv2.INTER_AREA)

	# Save Preprocessed images	
	cv2.imwrite(os.path.join(image_dir, 'depth_fill.png'), depth_f_rs)

print('All images saved.')
cam.close()
