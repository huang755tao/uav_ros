import rospy
from mavros_msgs.msg import State, PositionTarget, AttitudeTarget, ParamValue
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, ParamSet, ParamSetRequest

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import BatteryState, Imu

import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion

import numpy as np
import pandas as pd
from pathlib import Path as Filepath
from datetime import datetime

from utils.Rk4 import rk4_step

class AgentInit:
    def __init__(self, dt=0.01, use_gazebo=True):
        # 基础参数
        self.dt = dt
        self.use_gazebo = use_gazebo
        self.thrust_scale = 0.704 / 1.5 / 9.8 if use_gazebo else 0.35 / 0.72 / 9.8
                
        # 状态标志
        # self.is_record = False
        self.is_show_rviz = False
        
        # 控制参数
        self.set_target_position = (1.0, 1.0, 1.5)
        self.pos_tolerance = 0.2
        self.vel_tolerance = 0.2

        # 实时状态存储
        self.current_state = State()            # 存储当前无人机系统的状态信息
        self.current_pose = PoseStamped()       # 存储无人机当前位置
        self.target_pose = PoseStamped()        # 存储无人机目标位置
        self.uav_odom = Odometry()              # 实时传递无人机的位姿（位置与姿态）、速度（线速度与角速度）及状态置信度信息
        self.uav_states = np.zeros(12)          # 无人机实施状态反馈
        self.battery_voltage = 0.0              # 电池电压

        # 轨迹显示配置（Rviz专用）
        self.trajectory = Path()
        self.trajectory.header.frame_id = "map"
        self.trajectory_queue = []  # 维护最近1000个轨迹点

        # 数据记录
        self.data_log = []
        self.prev_velocity = np.zeros(3)
        self.prev_time = None

        # 初始化一ros节点
        rospy.init_node("px4_attitude_controller")
        self.rate = rospy.Rate(1/self.dt)
        self.t0 = rospy.Time.now().to_sec()
        self.current_time = rospy.Time.now()

        # 订阅器
        self.state_sub = rospy.Subscriber("mavros/state", State, callback=self.state_cb)
        self.odom_sub = rospy.Subscriber("mavros/local_position/odom", Odometry, callback=self.uav_odom_cb)
        self.battery_sub = rospy.Subscriber("mavros/battery", BatteryState, callback=self.uav_battery_cb)
        # self.uav_rate_sub = rospy.Subscriber("mavros/imu/data", Imu, callback=self.uav_rate_cb)

        # 发布器
        self.att_cmd_pub = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)   # 传输姿态控制指令
        self.pos_cmd_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)    # 传输定点位置
        # self.odom_pub = rospy.Publisher('drone_odom', Odometry, queue_size=10)                              # 初始化轨迹消息发布器
        self.trajectory_pub = rospy.Publisher('reference_path', Path, queue_size=10)                        # 初始化参考轨迹发布器

        # 服务客户端
        rospy.wait_for_service("/mavros/cmd/arming", timeout=10)
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool) # 创建一个服务客户端的 “代理对象”，允许客户端像调用本地函数一样调用远程服务。
    
        rospy.wait_for_service("/mavros/set_mode", timeout=10)
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        # 控制指令初始化
        self.att_target = AttitudeTarget()
        self.att_target.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | \
                                    AttitudeTarget.IGNORE_PITCH_RATE | \
                                    AttitudeTarget.IGNORE_YAW_RATE

        # 注册关闭回调（处理数据保存）
        rospy.on_shutdown(self.shutdown_handler)

    def state_cb(self, msg: State):
        self.current_state = msg
    
    def uav_battery_cb(self, msg: BatteryState):
        self.battery_voltage = msg.voltage

    def uav_odom_cb(self, msg: Odometry):
        self.uav_odom = msg
        self.current_pose.header = msg.header
        self.current_pose.pose = msg.pose.pose

        position = msg.pose.pose.position
        velocity = msg.twist.twist.linear
        angular_velocity = msg.twist.twist.angular
        orientation = msg.pose.pose.orientation
        euler = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        self.uav_states = np.array([position.x, position.y, position.z,
                                    velocity.x, velocity.y, velocity.z,
                                    euler[0], euler[1], euler[2],
                                    angular_velocity.x,angular_velocity.y,angular_velocity.z])

        if self.is_show_rviz:
            # RIVI显示
            self.rviz_show_path_limit()  

    def rviz_show_path(self):
        # 显示完整轨迹
        if rospy.is_shutdown():  # 新增节点状态检查
            return
        _odom = Odometry()
        _odom.header.stamp = rospy.Time.now()
        _odom.header.frame_id ='map'
        _odom.child_frame_id = 'base_link'
        _odom.pose.pose.position = self.current_pose.pose.position
        # 假设无人机没有旋转
        quaternion = quaternion_from_euler(0, 0, 0)
        _odom.pose.pose.orientation.x = quaternion[0]
        _odom.pose.pose.orientation.y = quaternion[1]
        _odom.pose.pose.orientation.z = quaternion[2]
        _odom.pose.pose.orientation.w = quaternion[3]
        # self.odom_pub.publish(_odom)

    def rviz_show_path_limit(self):
        # 显示完整轨迹
        if rospy.is_shutdown():
            return
        
        # 创建轨迹点
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose = self.current_pose.pose
        
        # 维护轨迹队列（限制长度）
        self.trajectory_queue.append(pose_stamped)
        if len(self.trajectory_queue) > 1000:
            self.trajectory_queue.pop(0)
        
        # 发布轨迹
        self.trajectory.poses = self.trajectory_queue
        self.trajectory.header.stamp = rospy.Time.now()
        self.trajectory_pub.publish(self.trajectory)

    def arm_safe(self, max_attempts=50, wait_sec=2.0) -> bool:
        """解锁无人机（带节点状态检查和基础重试）"""
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True
        rospy.loginfo("Attempting to arm...")

        for attempt in range(max_attempts):
            if rospy.is_shutdown():
                rospy.logwarn("Node shutdown - aborting arming")
                return False  # 节点已关闭，直接退出
            
            if self.current_state.armed:
                rospy.loginfo("Drone already armed")
                return True
            
            try:
                if not self.arming_client(arm_cmd).success:  # 检查服务调用是否成功
                    rospy.logwarn(f"Arm attempt {attempt+1} failed (service error)")
                    self.pub_position_keep()
                    continue  # 跳过等待，直接重试
                
                # 等待状态更新（同时检查节点状态）
                for _ in range(int(wait_sec / self.dt)):
                    if rospy.is_shutdown():
                        return False
                    self.rate.sleep()  # 使用类内定义的rate（更精准）
                
                if self.current_state.armed:
                    rospy.loginfo("Drone armed successfully")
                    return True
                else:
                    if rospy.is_shutdown():
                        return False
                    for _ in range(50):
                        self.pub_position_keep()
                        self.rate.sleep()
                    rospy.logwarn(f"Arm attempt {attempt+1} failed (status not updated)")
            
            except Exception as e:
                if not rospy.is_shutdown():  # 仅在节点运行时记录错误
                    rospy.logwarn(f"Arm attempt {attempt+1} failed: {str(e)}")
        
        rospy.logerr("Max arm attempts reached - please check connection")
        return False

    def arm(self):
        """解锁无人机"""
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True
        rospy.loginfo("3,Attempting to arm...")
        while not rospy.is_shutdown() and not self.current_state.armed:
            if self.arming_client.call(arm_cmd).success:
                rospy.loginfo("Drone armed successfully")
                self.rate.sleep()
                break
            self.pub_position_keep()
            self.rate.sleep()

    def set_offboard_mode_backup(self, max_attempts=10):
        """带重试次数限制的OFFBOARD模式切换"""
        offboard_mode = SetModeRequest()
        offboard_mode.custom_mode = "OFFBOARD"
        attempts = 0
        
        rospy.loginfo("2,Attempting to Switch OFFBOAD Mode...")
        while not rospy.is_shutdown() and self.current_state.mode != "OFFBOARD":
            if attempts >= max_attempts:
                rospy.logerr("Max OFFBOARD attempts reached - aborting")
                return False
            
            self.set_mode_client(offboard_mode)
            # rospy.loginfo(f"Switching to OFFBOARD (attempt {attempts+1})")
            self.pub_position_keep()
            self.rate.sleep()
            attempts += 1
        
        rospy.loginfo("OFFBOARD mode activated")
        return True
    
    def set_offboard_mode(self, max_attempts=10) -> bool:
        """切换OFFBOARD模式（增强错误处理）"""
        offboard_req = SetModeRequest()
        offboard_req.custom_mode = "OFFBOARD"
        rospy.loginfo("2,Attempting to switch to OFFBOARD mode...")
        
        for attempt in range(max_attempts):
            if rospy.is_shutdown():
                return False
            
            if self.current_state.mode == "OFFBOARD":
                rospy.loginfo("Already in OFFBOARD mode")
                return True
            
            try:
                response = self.set_mode_client(offboard_req)
                if response.mode_sent:
                    rospy.loginfo(f"OFFBOARD mode request sent (attempt {attempt+1})")
                    rospy.sleep(0.5)  # 等待模式切换
                else:
                    rospy.logwarn(f"Mode switch failed (attempt {attempt+1})")
                
                self.pub_position_keep()  # 维持模式要求
            
            except rospy.ServiceException as e:
                rospy.logerr(f"Set mode service failed: {str(e)}")
        
        rospy.logerr("Max OFFBOARD attempts reached")
        return False
    
    def thrust_to_throttle(self, thrust):
        """推力转换为油门值（0-2.0范围）"""
        throttle = np.clip(self.thrust_scale * thrust, 0.01, 3.0)
        return throttle
    
    def pub_position_keep(self):
        """发送当前位置以保持悬停（满足OFFBOARD模式要求）"""
        self.current_pose.header.stamp = rospy.Time.now()
        self.pos_cmd_pub.publish(self.current_pose)

    def publish_attitude_command(self, phi_d, theta_d, psi_d, thrust):
        """
        发布期望姿态和油门指令
        :param phi_d: 期望滚转角（弧度）
        :param theta_d: 期望俯仰角（弧度）
        :param psi_d: 期望偏航角（弧度）
        :param thrust: 期望推力（N）
        """
        # 生成四元数
        q = quaternion_from_euler(phi_d, theta_d, psi_d, axes='sxyz')
        # 填充姿态目标消息
        self.att_target.header.stamp = rospy.Time.now()
        self.att_target.orientation.x = q[0]
        self.att_target.orientation.y = q[1]
        self.att_target.orientation.z = q[2]
        self.att_target.orientation.w = q[3]
        self.att_target.thrust = self.thrust_to_throttle(thrust)


        # 发布指令
        self.att_cmd_pub.publish(self.att_target)
    
    def pub_target_position(self):
        self.target_pose.header.frame_id = "map"
        self.target_pose.pose.position.x = self.set_target_position[0]
        self.target_pose.pose.position.y = self.set_target_position[1]
        self.target_pose.pose.position.z = self.set_target_position[2]
        if not rospy.is_shutdown():
            self.target_pose.header.stamp = rospy.Time.now()
            self.pos_cmd_pub.publish(self.target_pose)

    
    def reach_target_position(self):
        rospy.loginfo(f"Moving to target position: {self.set_target_position}")
        self.target_pose.header.frame_id = "map"
        self.target_pose.pose.position.x = self.set_target_position[0]
        self.target_pose.pose.position.y = self.set_target_position[1]
        self.target_pose.pose.position.z = self.set_target_position[2]

        while not rospy.is_shutdown():
            self.target_pose.header.stamp = rospy.Time.now()
            self.pos_cmd_pub.publish(self.target_pose)

            current_pos = self.uav_odom.pose.pose.position
            distance = np.sqrt((current_pos.x - self.set_target_position[0])**2 +
                               (current_pos.y - self.set_target_position[1])**2 +
                               (current_pos.z - self.set_target_position[2])**2)

            if distance < self.pos_tolerance:
                rospy.loginfo("Reached target position!")
                return True

            self.rate.sleep()

    def shutdown_handler(self):
        # 终止程序后，询问是否保存程序
        if len(self.data_log) == 0:
            rospy.loginfo("No data to save")
            return

        user_input = input("\nData recording completed. Enter '1' to save data, '0' to discard: ")
        while user_input not in ['0', '1']:
            user_input = input("Invalid input. Please enter '1' to save or '0' to discard: ")
        
        if user_input == '1':
            data_dir = Filepath("scripts/data")
            data_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = data_dir / f"uav_data_{timestamp}.csv"
            df = pd.DataFrame(self.data_log)
            df.to_csv(filename, index=False)
            rospy.loginfo(f"Data saved to {filename}")
        else:
            rospy.loginfo("Data discarded")

    def initialize_system(self):
        """系统初始化流程"""
        # 等待飞控连接
        rospy.loginfo("initialization...")

        # 等待飞控连接
        rospy.loginfo("1,Waiting for FCU connection...")
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()
        
        # 发送初始位置指令（OFFBOARD模式需要）
        for _ in range(100):
            if rospy.is_shutdown():
                break
            self.pub_position_keep()
            self.rate.sleep()
        
        # 切换模式并解锁
        if not self.set_offboard_mode():
            rospy.signal_shutdown("Failed to switch to OFFBOARD mode")
            return

        # 输入1，arm电机
        user_input = input("Please enter '1' to arm and start flight: ")
        while user_input != '1':
            user_input = input("Invalid input. Please enter '1' to continue: ")
        self.arm_safe()
        
        rospy.loginfo("System initialization complete")

    def run(self, attitude_controller, simulation_time: float):
        """
        主控制循环
        :param attitude_controller: 外环控制器函数，返回(phi_d, theta_d, psi_d, thrust)
        :simulation_time: 仿真时常
        """
        
        self.initialize_system()
        rospy.loginfo(f"Control loop started at {1/self.dt:.1f}Hz")
        rospy.loginfo("Starting position guidance...")

        self.reach_target_position()
        rospy.loginfo('Wait 1 sec')
        for _ in range(100):
            self.pub_position_keep()
            self.rate.sleep()
        
        rospy.loginfo("Switching to attitude control...")

        # self.is_record = True
        self.is_show_rviz =  True

        start_time = rospy.Time.now()
        current_time = start_time
        self.t0 = start_time.to_sec()

        while (not rospy.is_shutdown()) and ((current_time - start_time).to_sec() < simulation_time):
            current_time = rospy.Time.now()            
            
            phi_d, theta_d, psi_d, thrust = attitude_controller()           # 获取控制器输出
            self.publish_attitude_command(phi_d, theta_d, psi_d, thrust)    # 发布控制指令
            # 发送位置保持指令（确保OFFBOARD模式维持）
            # rospy.loginfo("Control")
            self.rate.sleep()
        # self.is_record = False
        self.is_show_rviz = False
        self.set_target_position = (.0, .0, 2.5)
        self.reach_target_position()

def example_controller():
    # 这里应包含实际的外环控制算法（如SMC/PID/RL等）
    # 返回值：(滚转角, 俯仰角, 偏航角, 推力)
    return (0.0, 0.0, 0.0, 1.5*9.8)  # 示例：悬停指令

if __name__ == "__main__":
    controller = AgentInit()
    controller.run(example_controller,simulation_time=10.)
    # minimal_run()
