#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
from nav_msgs.msg import Path

class ViconDataParser:
    def __init__(self):
        # ==================== 参数初始化 ====================
        self.drone_name = rospy.get_param("~drone_name")  # 无人机名称（用于标识）
        self.input_topic = rospy.get_param("~input_topic")  # VRPN原始输入话题
        self.pose_topic = rospy.get_param("~pose_topic")  # 姿态输出话题
        self.twist_topic = rospy.get_param("~twist_topic")  # 速度输出话题
        self.path_topic = rospy.get_param("~path_topic")  # 轨迹输出话题
        self.child_frame = rospy.get_param("~child_frame")  # TF子帧名称（无人机坐标系）
        
        # ==================== 初始化发布者 ====================
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=10)
        self.twist_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=10)
        self.path_pub = rospy.Publisher(self.path_topic, Path, queue_size=10)
        
        # ==================== TF广播器 ====================
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # ==================== 状态变量 ====================
        self.last_pose = None  # 上一时刻位姿（用于速度计算）
        self.last_time = None  # 上一时刻时间戳（秒）
        self.path = Path()  # 轨迹消息（累积位姿）
        self.max_path_points = 1000  # 轨迹最大保留点数（避免内存溢出）
        
        # ==================== 订阅VRPN原始数据 ====================
        rospy.Subscriber(self.input_topic, PoseStamped, self.vrpn_callback)

    def vrpn_callback(self, msg):
        """VRPN数据回调函数（核心逻辑）"""
        # ==================== 1. 发布标准化姿态话题 ====================
        # 直接转发VRPN的位姿数据（可根据需要添加坐标变换）
        pose_msg = PoseStamped()
        pose_msg.header = msg.header  # 继承时间戳和参考帧（默认world）
        pose_msg.pose = msg.pose  # 位置和方向
        self.pose_pub.publish(pose_msg)

        # ==================== 2. 计算并发布速度 ====================
        if self.last_pose is not None:
            # 计算时间差（秒）
            current_time = msg.header.stamp.to_sec()
            dt = current_time - self.last_time
            if dt <= 0:
                rospy.logwarn("时间差异常（dt ≤ 0），跳过速度计算")
                return

            # 计算线速度（Δ位置 / Δ时间）
            dx = msg.pose.position.x - self.last_pose.position.x
            dy = msg.pose.position.y - self.last_pose.position.y
            dz = msg.pose.position.z - self.last_pose.position.z
            vx = dx / dt
            vy = dy / dt
            vz = dz / dt

            # 发布速度消息
            twist_msg = TwistStamped()
            twist_msg.header = msg.header
            twist_msg.twist.linear.x = vx
            twist_msg.twist.linear.y = vy
            twist_msg.twist.linear.z = vz
            # 角速度需通过方向变化计算（可选，此处省略）
            self.twist_pub.publish(twist_msg)

        # 更新上一状态
        self.last_pose = msg.pose
        self.last_time = msg.header.stamp.to_sec()

        # ==================== 3. 更新并发布轨迹 ====================
        # 初始化轨迹消息的header（仅首次设置）
        if not self.path.poses:
            self.path.header = msg.header
        # 添加当前位姿到轨迹
        self.path.poses.append(pose_msg)
        # 限制轨迹点数（保留最近max_path_points个点）
        if len(self.path.poses) > self.max_path_points:
            self.path.poses.pop(0)
        # 发布轨迹
        self.path_pub.publish(self.path)

        # ==================== 4. 广播TF变换 ====================
        tf_msg = TransformStamped()
        tf_msg.header = msg.header  # 时间戳和父帧（world）
        tf_msg.child_frame_id = self.child_frame  # 子帧（无人机坐标系）
        tf_msg.transform.translation = msg.pose.position  # 位置
        tf_msg.transform.rotation = msg.pose.orientation  # 方向
        self.tf_broadcaster.sendTransform(tf_msg)

if __name__ == "__main__":
    rospy.init_node("vicon_data_parser")
    parser = ViconDataParser()
    rospy.spin()
    