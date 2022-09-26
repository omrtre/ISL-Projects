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

postObjx = zed_interfaces.msg.Object().position[0]

def callback(msg):
    while not rospy.is_shutdown():
        objXThreshold = 2
        # If objects position (X) is greater than this, then sotp moving.
        if (postObjx >= objXThreshold):
            print("Object detected, position close by.")

def listenerObjX():

    rospy.init_node('listenerObjx', anonymous = True)
    rospy.Subscriber('/zed2i/zed_node/obj_det/objects', Object, callback)

    rospy.spin()

if __name__ == '__main__':
    listenerObjX()

#def stop():
#    var.linear.x = 0.0
#    var.angular.z = 0.0
#    pub.publish(var)

#def turn_left():
#    var.linear.x = .2
#    var.angular.z = 0.3
#    pub.publish(var)

#def forward_one_meter():
#    one_meter = 1
#    var.linear.x = .2
#    i = 1
#    time1 = rospy.Time.now().to_sec()
#    #while i == True:
#
#    start_dist = 0
#    while start_dist < one_meter:
#        pub.publish(var)
#        time2 = rospy.Time.now().to_sec()
#        start_dist = .2 * (time2 - time1)

#    var.linear.x = 0
#    pub.publish(var)
#    #   i == False

#def turn_90():
#    degree = 1.5
#    var.angular.z = .22
#    i = 1
#    time1 = rospy.Time.now().to_sec()
#    #while i == True:
#
#    start_dist = 0
#    while start_dist < degree:
#        pub.publish(var)
#        time2 = rospy.Time.now().to_sec()
#        start_dist = .2 * (time2 - time1)
#    #i == False
#    var.angular.z = 0
#    pub.publish(var)

#pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
#sub  = rospy.Subscriber('/odom', Odometry, callback)
#rate = rospy.Rate(5)
#rospy.spin()