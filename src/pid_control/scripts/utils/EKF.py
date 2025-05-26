import numpy as np
from scipy.linalg import block_diag

class UAV_EKF:
    def __init__(self, dt=0.01):
        """
        EKF初始化函数
        :param dt: 时间步长（与仿真频率匹配，PX4默认是100Hz即dt=0.01s）
        """
        # 状态向量维度：[x, y, z, vx, vy, vz]
        self.n = 6  
        # 观测向量维度：直接观测位置和 v（与状态维度相同）
        self.m = 6  

        # 初始化状态估计（首次更新前需要初始measure position）
        self.x = np.zeros((self.n, 1))  
        # 初始化协方差矩阵（根据经验设置初始不确定性）
        self.P = np.diag([1.1, 1.1, 1.1, 1.5, 1.5, 1.5])  

        # 状态转移矩阵F（连续时间模型离散化）
        self.F = np.eye(self.n)
        self.F[:3, 3:] = np.eye(3) * dt  # 位置 = 位置 +  v*dt

        # 过程噪声协方差矩阵Q（假设加 v噪声为白噪声）
        sigma_a = 0.1  # 加 v噪声标准差（m/s²）
        q = (dt**4)/4 * sigma_a**2
        qv = (dt**3)/2 * sigma_a**2
        q_a = dt**2 * sigma_a**2
        self.Q = np.array([
            [q, 0, 0, qv, 0, 0],
            [0, q, 0, 0, qv, 0],
            [0, 0, q, 0, 0, qv],
            [qv, 0, 0, q_a, 0, 0],
            [0, qv, 0, 0, q_a, 0],
            [0, 0, qv, 0, 0, q_a]
        ])

        # 观测矩阵H（直接观测位置和 v）
        self.H = np.eye(self.m)  

        # 测量噪声协方差矩阵R（根据传感器参数设置）
        # 假设位置噪声0.1m， v噪声0.05m/s
        self.R = block_diag(
            np.diag([0.1**2]*3),  # 位置测量噪声
            np.diag([0.05**2]*3)   #  v测量噪声
        )

        self.dt = dt
        self.last_time = None

        self.filter_date_log = []

    def predict(self):
        """执行预测步骤"""
        # 状态预测
        self.x = self.F @ self.x  
        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q  

    def update(self, z: np.ndarray, current_time: float, is_record: bool=True):
        """
        执行更新步骤
        :param z: 测量向量 [x, y, z, vx, vy, vz]（来自仿真/传感器）
        :param current_time: 当前时间戳（用于计算实际dt）
        :return: 滤波后的状态 [x, y, z, vx, vy, vz]
        """
        # 处理首次调用时的时间初始化
        if self.last_time is None:
            self.last_time = current_time
            self.x = np.array(z).reshape(-1, 1)  # 初始状态用首次measure position
            return self.x.flatten()

        # 计算实际时间步长（应对非固定频率情况）
        dt = current_time - self.last_time
        self.last_time = current_time

        # 更新状态转移矩阵中的dt
        self.F[:3, 3:] = np.eye(3) * dt

        # 执行预测
        self.predict()

        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 更新状态估计
        z = np.array(z).reshape(-1, 1)
        self.x = self.x + K @ (z - self.H @ self.x)

        # 更新协方差矩阵
        self.P = (np.eye(self.n) - K @ self.H) @ self.P
        
        if is_record:
            self.filter_date_log.append({
                "Timestamp": current_time,
                "filtered_x": self.x[0],
                "filtered_y": self.x[1],
                "filtered_z": self.x[2],
                "filtered_vx": self.x[3],
                "filtered_vy": self.x[4],
                "filtered_vz": self.x[5],
                "measure_x": z[0],
                "measure_y": z[1],
                "measure_z": z[2],
                "measure_vx": z[3],
                "measure_vy": z[4],
                "measure_vz": z[5]
            })

        return self.x.flatten()


    # def plot_data(self):
    #     measure_x = [state[0] for state in self.measure_states]
    #     measure_y = [state[1] for state in self.measure_states]
    #     measure_z = [state[2] for state in self.measure_states]
    #     measure_vx = [state[3] for state in self.measure_states]
    #     measure_vy = [state[4] for state in self.measure_states]
    #     measure_vz = [state[5] for state in self.measure_states]

    #     filtered_x = [state[0] for state in self.filter_states]
    #     filtered_y = [state[1] for state in self.filter_states]
    #     filtered_z = [state[2] for state in self.filter_states]
    #     filtered_vx = [state[3] for state in self.filter_states]
    #     filtered_vy = [state[4] for state in self.filter_states]
    #     filtered_vz = [state[5] for state in self.filter_states]

    #     import matplotlib.pyplot as plt
    #     fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    #     axs[0,0].plot(ekf.time_stamps, measure_x, 'g--', alpha=0.5, label='measure position')
    #     axs[0,0].plot(ekf.time_stamps, filtered_x, 'r', label='estimation')
    #     axs[0,0].set_title('X')

    #     axs[0,1].plot(measure_y, 'g--', alpha=0.5, label='measure position')
    #     axs[0,1].plot(filtered_y, 'r', label='estimation')
    #     axs[0,1].set_title('Y')

    #     axs[0,2].plot(measure_z, 'g--', alpha=0.5, label='measure position')
    #     axs[0,2].plot(filtered_z, 'r', label='estimation')
    #     axs[0,2].set_title('Z')

    #     axs[1,0].plot(measure_vx, 'g--', alpha=0.5, label='measure v')
    #     axs[1,0].plot(filtered_vx, 'r', label='estimation v')
    #     axs[1,0].set_title('Xv')

    #     axs[1,1].plot(measure_vy, 'g--', alpha=0.5, label='measure v')
    #     axs[1,1].plot(filtered_vy, 'r', label='estimation v')
    #     axs[1,1].set_title('Yv')

    #     axs[1,2].plot(measure_vz, 'g--', alpha=0.5, label='measure v')
    #     axs[1,2].plot(filtered_vz, 'r', label='estimation v')
    #     axs[1,2].set_title('Zv')

    #     for ax in axs.flat:
    #         ax.legend()
    #         ax.grid(True)
    #     plt.tight_layout()
    #     plt.show()


# ------------------------ 示例用法 ------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta

    # 初始化EKF（假设仿真频率100Hz）
    ekf = UAV_EKF(dt=0.01)

    # 生成仿真数据（模拟带有噪声的位置和 v）
    num_steps = 2000
    time = [datetime(2025, 5, 26) + timedelta(seconds=i*0.01) for i in range(num_steps)]
    
    # 真实轨迹（匀速运动示例）
    real_x = [0.1*i for i in range(num_steps)]
    real_y = [0.05*i for i in range(num_steps)]
    real_z = [0.2*i - 0.001*i**2 for i in range(num_steps)]  # 带小加 v的z轴运动
    real_vx = [0.1]*num_steps
    real_vy = [0.05]*num_steps
    real_vz = [0.2 - 0.002*i for i in range(num_steps)]

    # 生成含噪声的measure position（位置噪声±0.15m， v噪声±0.1m/s）
    np.random.seed(42)
    measure_x = [x + np.random.normal(0, 0.15) for x in real_x]
    measure_y = [y + np.random.normal(0, 0.15) for y in real_y]
    measure_z = [z + np.random.normal(0, 0.15) for z in real_z]
    measure_vx = [vx + np.random.normal(0, 0.1) for vx in real_vx]
    measure_vy = [vy + np.random.normal(0, 0.1) for vy in real_vy]
    measure_vz = [vz + np.random.normal(0, 0.1) for vz in real_vz]

    # 运行EKF滤波
    filtered_states = []
    for i in range(num_steps):
        z = [measure_x[i], measure_y[i], measure_z[i], 
             measure_vx[i], measure_vy[i], measure_vz[i]]
        filtered = ekf.update(z, time[i].timestamp())
        filtered_states.append(filtered)

    # 提取结果
    filtered_x = [state[0] for state in ekf.filter_states]
    filtered_y = [state[1] for state in ekf.filter_states]
    filtered_z = [state[2] for state in ekf.filter_states]
    filtered_vx = [state[3] for state in ekf.filter_states]
    filtered_vy = [state[4] for state in ekf.filter_states]
    filtered_vz = [state[5] for state in ekf.filter_states]

    # 绘制对比图
    # fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    # axs[0,0].plot(real_x, label='real')
    # axs[0,0].plot(measure_x, 'g--', alpha=0.5, label='measure position')
    # axs[0,0].plot(filtered_x, 'r', label='estimation')
    # axs[0,0].set_title('X')

    # axs[0,1].plot(real_y, label='real position')
    # axs[0,1].plot(measure_y, 'g--', alpha=0.5, label='measure position')
    # axs[0,1].plot(filtered_y, 'r', label='estimation')
    # axs[0,1].set_title('Y')

    # axs[0,2].plot(real_z, label='real position')
    # axs[0,2].plot(measure_z, 'g--', alpha=0.5, label='measure position')
    # axs[0,2].plot(filtered_z, 'r', label='estimation')
    # axs[0,2].set_title('Z')

    # axs[1,0].plot(real_vx, label='real v')
    # axs[1,0].plot(measure_vx, 'g--', alpha=0.5, label='measure v')
    # axs[1,0].plot(filtered_vx, 'r', label='estimation v')
    # axs[1,0].set_title('Xv')

    # axs[1,1].plot(real_vy, label='real v')
    # axs[1,1].plot(measure_vy, 'g--', alpha=0.5, label='measure v')
    # axs[1,1].plot(filtered_vy, 'r', label='estimation v')
    # axs[1,1].set_title('Yv')

    # axs[1,2].plot(real_vz, label='real v')
    # axs[1,2].plot(measure_vz, 'g--', alpha=0.5, label='measure v')
    # axs[1,2].plot(filtered_vz, 'r', label='estimation v')
    # axs[1,2].set_title('Zv')

    # for ax in axs.flat:
    #     ax.legend()
    #     ax.grid(True)

    # ekf.plot_data()

