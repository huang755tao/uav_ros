import rospy
import numpy as np

from control.dual_PID import PidControl
from control.PdTBC_backup import PdTBC  
from utils.ref_create import TrajectoryGenerator 

from env.uav_ros import UAV_ROS


def attitude_controller():
    # 这里应包含实际的外环控制算法（如SMC/PID/RL等）
    # 返回值：(滚转角, 俯仰角, 偏航角, 推力)
    return (0.0, 0.0, 0.0, 12.2)  # 示例：悬停指令


if __name__ == "__main__":
    dt = 0.01

    approch_time = 5
    simulation_time = 30
    uav = UAV_ROS(dt=dt, use_gazebo=True)

    # 实例化控制器
    dual_pid = PidControl(kp_pos=np.array([0.45, 0.45, 0.6]),
                ki_pos=np.array([0.002, 0.002, 0.001]),
                kd_pos=np.array([0., 0., 0.0]),
                kp_vel=np.array([3., 3., 3.5]),
                ki_vel=np.array([0.01, 0.01, 0.]),
                kd_vel=np.array([0., 0., 0.0]))
    

    # 实例化参考轨迹
    # 绘制圆形轨迹（半径1.5m，高度2m，频率0.2Hz）
    # circle_gen = TrajectoryGenerator(
    #     trajectory_type="circular",
    #     total_time=simulation_time,  # 总时长10秒（约3个周期，因频率0.3Hz周期≈3.33秒）
    #     dt=dt,          # 时间步长0.01秒（轨迹平滑）
    #     radius=1.5,
    #     z_offset=2.0,
    #     freq=0.2
    # )

    # circle_gen = TrajectoryGenerator(
    #     trajectory_type="sine_cosine",
    #     total_time=simulation_time,
    #     dt=dt,
    #     amplitudes=(1.0, 1.0, 0.5),  # X/Y/Z轴振幅
    #     freqs=(0.2, 0.2, 0.1)         # X/Y/Z轴频率
    # )

    circle_gen = TrajectoryGenerator(
        trajectory_type="figure8",
        total_time=simulation_time,  # 总时长10秒（约5个周期，频率0.5Hz周期2秒）
        dt=dt,
        radius=1.0,
        z_offset=1.5,
        freq=0.1
    )

    red_p0 = np.zeros(3)
    ref = np.zeros(3)
    ref_d = np.zeros(3)
    ref_dd = np.zeros(3)

    rospy.loginfo('system init')
    uav.initialize_system()

    rospy.loginfo(f"Control loop started at {1/uav.dt:.1f}Hz")
    rospy.loginfo("Starting position guidance...")
    # uav.target_position = (1.0, 1.0, 2.0)
    # uav.reach_target_position()

    rospy.loginfo('Wait 1 sec')
    for _ in range(100):
        uav.pub_position_keep()
        uav.rate.sleep()

    rospy.loginfo('Record ref')
    uav.is_record_ref = True
    rospy.loginfo('Show in rviz')
    uav.is_show_rviz =  True

    rospy.loginfo("Switching to attitude control...")
    rospy.loginfo("Use pid to approch...")
    distance = 1
    distance_tolerant = 0.1
    while (not rospy.is_shutdown()) and (distance > distance_tolerant):
        current_x = uav.uav_states[0]
        current_y = uav.uav_states[1]
        current_z = uav.uav_states[2]

        ref_state = np.zeros(9)
        ref_state[0:3]=np.array([1,1,1])
        
        uav_state = uav.uav_states[:6]
        dual_pid.update_dual(state=uav_state, ref_state=ref_state)
        uo = dual_pid.control

        phi_d, theta_d, thrust = uav.u_to_angle_dir(uo)
        psi_d = 0

        uav.publish_attitude_command(phi_d, theta_d, psi_d, thrust)

        distance = np.linalg.norm(uav.uav_states[0:3]-ref_state[0:3])
        uav.rate.sleep()

    rospy.loginfo("Approched...")
    rospy.loginfo("Start to run controller...")

    circle_gen.p0 = uav.uav_states[0:3]

    start_time = rospy.Time.now()
    current_time = start_time
    uav.t0 = start_time.to_sec()
    sim_t = (current_time - start_time).to_sec()

    while (not rospy.is_shutdown()) and (sim_t < simulation_time):  
        current_time = rospy.Time.now()
        sim_t = (current_time - start_time).to_sec()

        current_x = uav.uav_states[0]
        current_y = uav.uav_states[1]
        current_z = uav.uav_states[2]

        traject = circle_gen.get_point(t=sim_t)

        ref_state = np.array([traject.pos[0],traject.pos[1],traject.pos[2],
                               traject.vel[0],traject.vel[1],traject.vel[2],
                               traject.acc[0],traject.acc[1],traject.acc[2]])
        
        # ref_state = np.zeros(9)
        # ref_state[0:3]=np.array([1,1,3])
        
        uav_state = uav.uav_states[:6]
        dual_pid.update_dual(state=uav_state, ref_state=ref_state)
        uo = dual_pid.control

        phi_d, theta_d, thrust = uav.u_to_angle_dir(uo)
        psi_d = 0

        uav.publish_attitude_command(phi_d, theta_d, psi_d, thrust)

        uav.data_record(sim_t,ref_state)
        uav.rate.sleep()

    rospy.loginfo('Simulation is finished')
    uav.is_show_rviz = False
    uav.target_position = (.0, .0, .5)
    uav.reach_target_position()
        

