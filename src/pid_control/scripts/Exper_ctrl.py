import rospy
import numpy as np

from control.dual_PID import PidControl
from control.BC import BC
from control.FTBC import FTBC
from control.NFTBC import NFTBC
from control.PdTBC import PdTBC  

from utils.ref_create import TrajectoryGenerator 
from utils.EKF import UAV_EKF
from utils.utilis import data_record_to_file

from observer.PdTDO import PTDO

from env.uav_ros import UAV_ROS


def attitude_controller():
    # 这里应包含实际的外环控制算法（如SMC/PID/RL等）
    # 返回值：(滚转角, 俯仰角, 偏航角, 推力)
    return (0.0, 0.0, 0.0, 12.2)  # 示例：悬停指令


if __name__ == "__main__":
    dt = 0.01

    approch_time = 5
    simulation_time = 30.

    control_name = "PTBC_figure8_D_noDO"

    uav = UAV_ROS(m=0.72, dt=dt, use_gazebo=False, control_name=control_name)
    
    # ekf = UAV_EKF(dt=uav.dt)
    observer = PTDO(dt=uav.dt)

    # 实例化控制器
    # 双闭环的参数
    pid_ctrl = PidControl(kp_pos=np.array([0.5, 0.5, 1.1]),
                ki_pos=np.array([0.001, 0.001, 0.00]),
                kd_pos=np.array([0., 0., 0.0]),
                kp_vel=np.array([3., 3., 4.]),
                ki_vel=np.array([0.00, 0.00, 0.00]),
                kd_vel=np.array([0., 0., 0.0]))
    
    # 反步控制
    bc_ctrl = BC(dt=dt,
                 m=uav.m,
                 k1=np.array([0.3, 0.3, 1.0]),
                 k2=np.array([3., 3., 3.]))   
    
    # 固定时间控制
    ft_ctrl = FTBC(dt=dt,                     
                   m=uav.m,
                   k_eta12=np.array([0.15, 0.15, 0.1]),
                   l_eta1=np.array([20, 20., 10.]),

                   k_eta22=np.array([0.1, 0.1, 0.1]),
                   l_eta2=np.array([1., 1., 1.]), 

                   l_eta3=np.array([0.5, 0.5, 0.5]),
                   l_eta4=np.array([0.5, 0.5, 0.5]),          
                   k_eta31=np.array([.5, .5, .5]),  
                   k_eta32=np.array([1.0, 1.0, 1.0]),
                   p_eta= 0.2,
                   T_eta= 10.)
    ft_ctrl.k11 = np.array([10., 12., 15.])
    ft_ctrl.k21 = np.array([35., 35., 45.])

    # 非奇异固定时间控制
    nft_ctrl = NFTBC(dt=dt,                     
                   m=uav.m,
                   k_eta12=np.array([0.15, 0.15, 0.1]),
                   l_eta1=np.array([20, 20., 10.]),

                   k_eta22=np.array([0.1, 0.1, 0.1]),
                   l_eta2=np.array([1., 1., 1.]), 

                   l_eta3=np.array([0.5, 0.5, 0.5]),
                   l_eta4=np.array([0.5, 0.5, 0.5]),          
                   k_eta31=np.array([.5, .5, .5]),  
                   k_eta32=np.array([1.0, 1.0, 1.0]),
                   p_eta= 0.2,
                   T_eta= 10.)
    nft_ctrl.k11 = np.array([10., 12., 15.])
    nft_ctrl.k21 = np.array([35., 35., 45.])

    # 预设时间控制
    pdt_ctrl = PdTBC(dt=dt,
                     m=uav.m,
                     k_eta12=np.array([0.15, 0.15, 0.1]),
                     l_eta1=np.array([20, 20., 10.]),

                     k_eta22=np.array([0.1, 0.1, 0.1]),
                     l_eta2=np.array([1., 1., 1.]), 

                     l_eta3=np.array([0.5, 0.5, 0.5]),
                     l_eta4=np.array([0.5, 0.5, 0.5]),          
                     k_eta31=np.array([.5, .5, .5]),  
                     k_eta32=np.array([1.0, 1.0, 1.0]),
                     p_eta= 0.2,
                     T_eta= 10.)

    pdt_ctrl.k11 = np.array([10., 10., 15.])
    pdt_ctrl.k21 = np.array([35., 35., 45.])


    # 实例化参考轨迹
    # 绘制圆形轨迹（半径1.5m，高度2m，频率0.2Hz）
    # circle_gen = TrajectoryGenerator(
    #     trajectory_type="circular",
    #     total_time=simulation_time,  # 总时长10秒（约3个周期，因频率0.3Hz周期≈3.33秒）
    #     dt=dt,          # 时间步长0.01秒（轨迹平滑）
    #     radius=1.,
    #     z_offset=1.5,
    #     freq=0.1
    # )

    # freqs = 0.1
    # circle_gen = TrajectoryGenerator(
    #     trajectory_type="sine_cosine",
    #     total_time=simulation_time,
    #     dt=dt,
    #     amplitudes=(0., 1., 0.6),  # X/Y/Z轴振幅
    #     freqs=(0, freqs, freqs)         # X/Y/Z轴频率
    # )

    circle_gen = TrajectoryGenerator(
        trajectory_type="figure8",
        total_time=simulation_time,  # 总时长10秒（约5个周期，频率0.5Hz周期2秒）
        dt=dt,
        radius=1.0,
        z_offset=0.,
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

    rospy.loginfo('Record')
    uav.is_record_ref = True
    uav.is_record_obs = True
    rospy.loginfo('Show in rviz')
    uav.is_show_rviz = True
    if not uav.use_gazebo:
        uav.is_show_rviz = False

    rospy.loginfo("Switching to attitude control...")
    rospy.loginfo("Use pid to approch z...")
    distance = 1
    distance_tolerant = 0.2
    while (not rospy.is_shutdown()) and (distance > distance_tolerant):
        current_x = 0
        current_y = 0.

        ref_state = np.zeros(9)
        ref_state[0:3]=np.array([current_x, current_y, 2.])
        
        # uav_state = uav.uav_states[:6]
        # pid_ctrl.update_dual(state=uav_state, ref_state=ref_state)
        # uo = pid_ctrl.control
        # phi_d, theta_d, thrust = uav.u_to_angle_dir(uo)
        # psi_d = 0

        # uav.publish_attitude_command(phi_d, theta_d, psi_d, thrust)
        uav.set_target_position=(ref_state[0], ref_state[1], ref_state[2])
        uav.pub_target_position()

        distance = np.linalg.norm(uav.uav_states[0:3]-ref_state[0:3])
        uav.rate.sleep()

    rospy.loginfo("Approched z...")

    rospy.loginfo("Use pid to approch xy...")
    distance = 1
    distance_tolerant = 0.35
    while (not rospy.is_shutdown()) and (distance > distance_tolerant):

        ref_state = np.zeros(9)
        ref_state[0:3]=np.array([0, 1, 2.])
        
        uav_state = uav.uav_states[:6]
        pid_ctrl.update_dual(state=uav_state, ref_state=ref_state)
        uo = pid_ctrl.control

        phi_d, theta_d, thrust = uav.u_to_angle_dir(uo)
        psi_d = 0

        uav.publish_attitude_command(phi_d, theta_d, psi_d, thrust)

        distance = np.linalg.norm(uav.uav_states[0:3]-ref_state[0:3])
        uav.rate.sleep()

    rospy.loginfo("Approched xy...")

    rospy.loginfo("Start to run controller...")

    circle_gen.p0 = np.array([0, 0, 2.])

    start_time = rospy.Time.now()
    current_time = start_time
    uav.t0 = start_time.to_sec()
    sim_t = (current_time - start_time).to_sec()
    print(sim_t, 'sim_t')

    thrust_rate = 1.0 
    while (not rospy.is_shutdown()) and (sim_t < simulation_time):  

        current_time = rospy.Time.now()
        sim_t = (current_time - start_time).to_sec()
        if sim_t <= 30:
            current_x = uav.uav_states[0]
            current_y = uav.uav_states[1]
            current_z = uav.uav_states[2]
            uav_state = uav.uav_states[:6]

            # freqs += 0.1/5*uav.dt
            # freqs = min(freqs, 0.2)
            # circle_gen.params["freqs"] = (0, freqs, freqs)
            traject = circle_gen.get_point(t=sim_t)
            # traject.acc[2] += 0. * np.cos(3*np.pi/2*sim_t)
            ref_state = np.array([traject.pos[0],traject.pos[1],traject.pos[2],
                                traject.vel[0],traject.vel[1],traject.vel[2],
                                traject.acc[0],traject.acc[1],traject.acc[2]])
            
            sys_dynamic = - uav.kt / uav.m * uav_state[3:6] + pdt_ctrl.control
            obs = observer.observe(x=uav_state[3:6], system_dynamic=sys_dynamic, u=pdt_ctrl.control)

            # ekf.update(z=uav_state, current_time=sim_t)

            # ref_state = np.zeros(9)
            # ref_state[0:3]=np.array([0.8, 0.8, 1.5])
            # # 画正方形
            # if sim_t >= 8:
            #     ref_state[0:3]=np.array([0.8, -0.8, 1.5])
            # if sim_t >= 16:
            #     ref_state[0:3]=np.array([-0.8, -0.8, 1.5])
            # if sim_t >= 24:
            #     ref_state[0:3]=np.array([-0.8, 0.8, 1.5])
            
            # pid_ctrl.update(state=uav_state, ref_state=ref_state)
            pid_ctrl.update_dual(state=uav_state, ref_state=ref_state)
            bc_ctrl.update(state=uav_state, ref_state=ref_state)
            ft_ctrl.update(state=uav_state, ref_state=ref_state, obs= 0.*(obs.clip(-np.array([0.1, 0.15, 8.5]), np.array([0.1, 0.15, 8.5]))))
            nft_ctrl.update(state=uav_state, ref_state=ref_state, obs= 0.*(obs.clip(-np.array([0.1, 0.1, 8.5]), np.array([0.1, 0.1, 8.5]))))
            pdt_ctrl.update(state=uav_state, ref_state=ref_state, obs= 0.05*(obs.clip(-np.array([0.1, 0.15, 1*8.5]), np.array([0.1, 0.15, 1*8.5]))))
        
            # uo = pid_ctrl.control
            # uo = bc_ctrl.control
            # uo = ft_ctrl.control
            # uo = nft_ctrl.control
            uo = pdt_ctrl.control

            phi_d, theta_d, thrust = uav.u_to_angle_dir(uo, is_idea=True)
            psi_d = 0
            # print(pdt_ctrl.L)
            thrust_rate -= 0.1/10*uav.dt
            thrust_rate = max(thrust_rate, 0.75)
            thrust = thrust * thrust_rate
            
            uav.publish_attitude_command(phi_d, theta_d, psi_d, thrust)

            # uav.set_target_position=(ref_state[0], ref_state[1], ref_state[2])
            # uav.pub_target_position()

            uav.data_record(sim_t=sim_t, 
                            thrust=thrust, 
                            ref_state=ref_state,
                            obs_state=obs)            
        uav.rate.sleep()



    rospy.loginfo('Simulation is finished')
    # data_record_to_file(data_log=ekf.filter_date_log, control_name="PtD")
    uav.is_show_rviz = False
    uav.set_target_position = (.0, .0, 1.5)
    uav.reach_target_position()

    


        

