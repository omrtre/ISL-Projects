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
    if msg.ranges[0] > 0.5:
        move_cmd.linear.x = 0.2
        move_cmd.angular.z = 0.0
    else:
        move_cmd.linear.x = 0.0
        move_cmd.angular.z = 0.0
    pub.publish(move_cmd)
rospy.init_node('laser_data')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
sub = rospy.Subscriber('scan',LaserScan, callback)
move_cmd = Twist()
rospy.spin()


