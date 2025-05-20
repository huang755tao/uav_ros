#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

# 初始化全局变量，用于存储路径
path = Path()

def odom_callback(odom_msg):
    global path
    # 创建一个新的 PoseStamped 消息
    pose_stamped = PoseStamped()
    # 设置 PoseStamped 消息的头部信息
    pose_stamped.header = odom_msg.header
    # 设置 PoseStamped 消息的姿态信息
    pose_stamped.pose = odom_msg.pose.pose
    # 更新路径消息的头部信息
    path.header = odom_msg.header
    # 将新的姿态信息添加到路径中
    path.poses.append(pose_stamped)
    # 发布路径消息
    path_pub.publish(path)

if __name__ == "__main__":
    # 初始化 ROS 节点
    rospy.init_node('odom_to_path_converter')
    # 订阅 Odometry 消息
    odom_sub = rospy.Subscriber('/drone_odom', Odometry, odom_callback)
    # 发布 Path 消息
    path_pub = rospy.Publisher('/drone_path', Path, queue_size=10)
    # 进入 ROS 循环
    rospy.spin()

