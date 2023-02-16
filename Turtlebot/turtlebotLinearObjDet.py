#! /usr/bin/env python3

from queue import Empty
from re import X
from shutil import move
from tkinter import Y
from turtle import distance, forward, position, speed

from numpy import empty
import rospy
#from geometry_msgs.msg import Twist, Point
#from nav_msgs.msg import Odometry
from zed_interfaces.msg import Object
from zed_interfaces.msg import ObjectsStamped

objects = ObjectsStamped()
postObjx = Object()

def callback(msg):
    while not rospy.is_shutdown():
    	objXThreshold = 2.0
    	
    	for i in objects:
    		if (objects[i].position[0] >= objThreshold):
    			print("Object" + str(objects[i].label) + "detected, position close by.")

def listenerObjX():

    rospy.init_node('listenerObjx', anonymous = True)
    subObjlist = rospy.Subscriber('/zed2i/zed_node/obj_det/objects', ObjectsStamped, callback)
    #subZedOdom = rospy.Subscriber('/zed2i/zed_node/odom', callback)
    rospy.spin()

if __name__ == '__main__':
    listenerObjX()
