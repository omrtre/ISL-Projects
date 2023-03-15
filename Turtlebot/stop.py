#! /usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

def callback(msg):
    print("==========================")
    print('s1 [270]')
    print (msg.ranges[270])
    print('s2 [0]')
    print (msg.ranges[0])
    print('s3 [90]')
    print(msg.ranges[90])
    if msg.ranges[0] > 1:
        move_cmd.linear.x = 0.2
        move_cmd.angular.z = 0.0
    if msg.ranges[0] < 1:
        move_cmd.linear.x = 0.0
        move_cmd.angular.z = 0.2
    if msg.ranges[270] < 0.3:
        move_cmd.linear.x = 0.1
        move_cmd.angular.z = 0.2
    if msg.ranges[315] < 0.3:
        move_cmd.linear.x = 0.0
        move_cmd.angular.z = 0.2

    pub.publish(move_cmd)
rospy.init_node('laser_data')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
sub = rospy.Subscriber('scan',LaserScan, callback)
move_cmd = Twist()
rospy.spin()


