#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
from nav_msgs.msg import Path

class ViconDataParser:
    def __init__(self):
        # ==================== 参数初始化 ====================
        self.drone_name = rospy.get_param("~drone_name")  # 修正：使用launch传递的drone_name参数
        self.pose_topic = rospy.get_param("~pose_topic")    # 姿态话题
        self.twist_topic = rospy.get_param("~twist_topic")  # 速度话题
        self.path_topic = rospy.get_param("~path_topic")    # 轨迹话题
        self.child_frame = rospy.get_param("~child_frame")  # TF子帧名称
        
        # ==================== 初始化发布者 ====================
        self.path_pub = rospy.Publisher(self.path_topic, Path, queue_size=10)
        self.twist_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=10)  # 新增速度发布者
        
        # ==================== TF广播器 ====================
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # ==================== 状态变量 ====================
        self.path = Path()
        self.path.header.frame_id = "world"  # 显式设置frame_id
        self.max_path_points = 1000
        self.last_pose = None
        self.last_stamp = None
        
        # ==================== 订阅数据 ====================
        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback)
        rospy.Subscriber(self.twist_topic, TwistStamped, self.twist_callback)  # 新增速度订阅

    def pose_callback(self, msg):
        """位姿数据回调：更新轨迹和TF"""
        # ==================== 1. 更新并发布轨迹 ====================
        self.path.header.stamp = msg.header.stamp  # 更新时间戳
        self.path.poses.append(msg)
        # 限制轨迹点数
        if len(self.path.poses) > self.max_path_points:
            self.path.poses.pop(0)
        self.path_pub.publish(self.path)

        # ==================== 2. 广播TF变换 ====================
        tf_msg = TransformStamped()
        tf_msg.header = msg.header
        tf_msg.child_frame_id = self.child_frame
        tf_msg.transform.translation = msg.pose.position
        tf_msg.transform.rotation = msg.pose.orientation
        self.tf_broadcaster.sendTransform(tf_msg)

    def twist_callback(self, msg):
        """速度数据回调：直接转发或处理"""
        # 示例：直接转发速度数据（可根据需求添加滤波等处理）
        self.twist_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("vicon_data_parser")
    parser = ViconDataParser()
    rospy.spin()
