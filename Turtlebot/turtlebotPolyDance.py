#! /usr/bin/env python3

from queue import Empty
from re import X
from shutil import move
from tkinter import Y
from turtle import distance, forward, position, speed

from numpy import empty
import rospy
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan 
from nav_msgs.msg import Odometry
rospy.init_node('square')
var = Twist()
odom = Odometry()
goal = Point()

def forward_one_meter():
    one_meter = 1
    var.linear.x = .2
    i = 1
    time1 = rospy.Time.now().to_sec()

    start_dist = 0
    while start_dist < one_meter:
        pub.publish(var)
        time2 = rospy.Time.now().to_sec()
        start_dist = .2 * (time2 - time1)

    var.linear.x = 0
    pub.publish(var)


def turn_angle(degree):
    var.angular.z = .22
    time1 = rospy.Time.now().to_sec()

    start_dist = 0
    while start_dist < degree:
        pub.publish(var)
        time2 = rospy.Time.now().to_sec()
        start_dist = .2 * (time2 - time1)
    var.angular.z = 0
    pub.publish(var)

def callback(msg):
    iterations = int(input("How many sides? "))
    i = 0
    degree = float(input("What angle? "))
    while not rospy.is_shutdown():
        while i < iterations: 
        #for i in range(iterations): 
            turn_angle(degree)
            forward_one_meter()
            print("Side completed, on to the next!")
            i += 1
            if i == iterations:
        	    print("Sides completed!")
        	    rospy.is_shutdown()

pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
sub  = rospy.Subscriber('/odom', Odometry, callback)
rate = rospy.Rate(5)
rospy.spin()
