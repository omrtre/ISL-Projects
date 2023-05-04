import numpy as np
import cv2 as cv
import sys
def analyze(img):
    img = cv.imread(img)

    if img is None:
        sys.exit("Could not read the image")

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower = np.array([24,50,50])
    upper = np.array([170,255,255])

    mask = cv.inRange(hsv,lower,upper)


    has_blue = np.sum(mask)
    print(has_blue)

    if has_blue > 0:
        print("Blue Detected!\nPress 's' to exit.")

    cv.imshow('mask',mask)
    cv.imshow('img',img)
    cv.imshow('hsv',hsv)

    k = cv.waitKey(0)

    if  k == ord("s"):
        cv.imwrite("colors.jpeg",img)
    return 0

analyze("photo.jpg")