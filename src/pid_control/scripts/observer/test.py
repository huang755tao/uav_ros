import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple  # 添加类型模块导入
from observer.Filter import RK4DynamicFilter, FirstOrderFilter  # 假设路径正确

from observer.PdTDO import PTDO
from utils.TestModel import SecondOrderSystem


import numpy as np
from typing import Tuple


def generate_second_order_signal(
    t: np.ndarray, 
    a: float = 2.0,         # 基础加速度（二次项系数）
    A: float = 1.0,         # 正弦分量振幅
    omega: float = 5.0      # 正弦角频率（rad/s）
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成含正弦分量的二阶参考信号（位置+速度+加速度）
    信号形式：x(t) = 0.5*a*t² + A*sin(ωt)（二次项+正弦波动）
    """
    # 位置信号（二次项 + 正弦分量）
    x = 0.5 * a * t**2 + A * np.sin(omega * t)
    
    # 速度信号（二次项导数 + 正弦分量导数）
    v = a * t + A * omega * np.cos(omega * t)
    
    # 加速度信号（二次项导数的导数 + 正弦分量导数的导数）
    acc = a - A * (omega**2) * np.sin(omega * t)
    
    # 扩展为3维信号（各轴相同）
    x_3d = np.stack([x, x, x], axis=1)  # 形状：(len(t), 3)
    v_3d = np.stack([v, v, v], axis=1)
    acc_3d = np.stack([acc, acc, acc], axis=1)
    
    return x_3d, v_3d, acc_3d


def main():
    # 生成时间序列
    t_start, t_end, dt = 0.0, 25.0, 0.01
    t = np.arange(t_start, t_end, dt)
    num_steps = len(t)
    
    # 初始化参考信号（3维抛物线运动）
    x_ref, v_ref, acc_ref = generate_second_order_signal(t)
    
    # 初始化跟踪器
    # 二阶动态跟踪器（可估计位置和速度）
    dyn_filter = RK4DynamicFilter(dt=dt, r11=10.0, r21=100.0)
    # 一阶惯性跟踪器（仅估计位置和一阶导数）
    gain_vector = np.array([100.0, 100.0, 300.0])  # 各轴独立增益
    first_order_filter = FirstOrderFilter(dt=dt, kt=gain_vector)
    
    # 存储跟踪结果
    dyn_z1_list = []  # 动态跟踪器位置估计
    dyn_dot_z1_list = []  # 动态跟踪器速度估计
    dyn_z2_list = []  # 动态跟踪器二阶导数估计（需访问内部状态）
    
    fo_z1_list = []  # 一阶跟踪器位置估计
    fo_dot_z1_list = []  # 一阶跟踪器速度估计
    
    for x in x_ref:
        # 二阶跟踪器更新
        z1_dyn, dot_z1_dyn = dyn_filter.update(x)
        dyn_z1_list.append(z1_dyn)
        dyn_z2_list.append(dot_z1_dyn)
        
        # 一阶跟踪器更新
        z1_fo, dot_z1_fo = first_order_filter.update(x)
        fo_z1_list.append(z1_fo)
        fo_dot_z1_list.append(dot_z1_fo)
    
    # 转换为数组
    dyn_z1 = np.array(dyn_z1_list)
    dyn_z2 = np.array(dyn_z2_list) 
    
    fo_z1 = np.array(fo_z1_list)
    fo_dot_z1 = np.array(fo_dot_z1_list)
    
    # 绘制结果（以第一轴为例，3轴信号相同）
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    
    # 位置跟踪对比
    axes[0, 0].plot(t, x_ref[:, 0], 'r-', label='Reference')
    axes[0, 0].plot(t, dyn_z1[:, 0], 'b--', label='DynamicFilter')
    axes[0, 0].set_title('Position Tracking (Axis 1)')
    axes[0, 0].legend()
    
    axes[0, 1].plot(t, x_ref[:, 0], 'r-', label='Reference')
    axes[0, 1].plot(t, fo_z1[:, 0], 'g-.', label='FirstOrderFilter')
    axes[0, 1].legend()
    
    # 速度跟踪对比
    axes[1, 0].plot(t, v_ref[:, 0], 'r-', label='True Velocity')
    axes[1, 0].plot(t, dyn_z2[:, 0], 'b--', label='DynamicFilter Velocity')
    axes[1, 0].set_title('Velocity Tracking (Axis 1)')
    
    axes[1, 1].plot(t, v_ref[:, 0], 'r-', label='True Velocity')
    axes[1, 1].plot(t, fo_dot_z1[:, 0], 'g-.', label='FirstOrderFilter Velocity')
    
    # # 加速度跟踪对比（仅DynamicFilter有二阶估计）
    # axes[2, 0].plot(t, acc_ref[:, 0], 'r-', label='True Acceleration')
    # axes[2, 0].plot(t, dyn_z2[:, 0], 'b--', label='DynamicFilter Acceleration')
    # axes[2, 0].set_title('Acceleration Tracking (Axis 1)')
    # axes[2, 0].set_xlabel('Time (s)')
    
    # axes[2, 1].axis('off')  # 一阶跟踪器无二阶输出，隐藏右下图
    
    plt.tight_layout()
    plt.show()



# def main():
#     dt = 0.01          
#     total_time = 20.0  
#     steps = int(total_time / dt) + 1
#     t = np.linspace(0, total_time, steps)

#     def disturbance(t):
#         return 0.2 * np.sin(1.2 * t) + 0.1 * (1 if t > 5 else 0) + 0.05

#     system = SecondOrderSystem(omega_n=2.0, zeta=0.7, disturbance=disturbance)
#     observer = PTDO(dim=1, T=2., p=1.0, m1=10, m2=10, k_d2=5., k_l2=5., k_e2=1.)
#     observer.set_init(np.array([0.0]))

#     x_history = np.zeros((steps, 2))     
#     u_history = np.zeros(steps)          
#     delta_est_history = np.zeros(steps)  
#     delta_true_history = np.zeros(steps) 

#     for i in range(steps):
#         u = 0.5 if i*dt < 3 else 0.8  
#         system.step(u, dt)
#         x1, x2 = system.x
#         sys_dynamic = system.dynamic  

#         x_observe = np.array([x2])              
#         u_observe = np.array([u])               
#         sys_dynamic_observe = np.array([sys_dynamic])  

#         delta_est = observer.observe(
#             system_dynamic=sys_dynamic_observe,
#             x=x_observe,
#             u=u_observe,
#             config='PD'
#         )
        
#         x_history[i] = system.x
#         u_history[i] = u
#         delta_est_history[i] = delta_est[0]
#         delta_true_history[i] = system.disturbance(system.t)

#     plt.figure(figsize=(12, 8))
    
#     plt.subplot(2, 1, 1)
#     plt.plot(t, delta_true_history, label='True Disturbance', linewidth=2)
#     plt.plot(t, delta_est_history, label='Estimated Disturbance', linestyle='--', linewidth=2)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Disturbance Value')
#     plt.title('Disturbance Estimation Comparison')
#     plt.legend()
#     plt.grid(True)
    
#     plt.subplot(2, 1, 2)
#     plt.plot(t, x_history[:, 0], label='Displacement x1', linewidth=2)
#     plt.plot(t, x_history[:, 1], label='Velocity x2', linewidth=2)
#     plt.xlabel('Time (s)')
#     plt.ylabel('State Value')
#     plt.title('System State Evolution')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()


if __name__ == "__main__":
    main()
