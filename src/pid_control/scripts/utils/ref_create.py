import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 三维绘图支持
from dataclasses import dataclass


@dataclass
class TrajectoryPoint:
    """存储单个时刻的轨迹信息（时间、位置、速度、加速度）"""
    t: float          # 当前时间（秒，相对于轨迹起始时间）
    pos: np.ndarray   # [x, y, z] 位置（米）
    vel: np.ndarray   # [vx, vy, vz] 速度（米/秒）
    acc: np.ndarray   # [ax, ay, az] 加速度（米/秒²）


class TrajectoryGenerator:
    def __init__(self, trajectory_type: str = "circular", total_time: float = 30.0, dt: float = 0.01, **kwargs):
        """
        轨迹生成器初始化
        参数：
            trajectory_type: 轨迹类型（"circular", "figure8", "square", "sine_cosine"）
            total_time: 轨迹总时长（秒，超过后循环）
            dt: 时间步长（秒，控制轨迹平滑度，默认0.01s）
            **kwargs: 轨迹参数（如radius半径、freq频率等）
        """
        self.trajectory_type = trajectory_type
        self.total_time = total_time
        self.dt = dt
        self.params = kwargs
        self._validate_params()  # 参数校验（确保必要参数存在）
        print('Tracking types are circular, figure8, square, sine_cosine')

        # 初始化初始位置
        self.p0 = np.zeros(3)

    def _validate_params(self):
        """校验轨迹类型所需的参数是否完整"""
        required_params = {
            "circular": ["radius", "z_offset", "freq"],        # 圆形轨迹必要参数
            "figure8": ["radius", "z_offset", "freq"],         # 8字轨迹必要参数
            "square": ["side_length", "z_offset", "speed"],     # 方形轨迹必要参数
            "sine_cosine": ["amplitudes", "freqs"]             # 正余弦组合轨迹必要参数
        }.get(self.trajectory_type, [])
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"轨迹类型 '{self.trajectory_type}' 缺少必要参数: {param}")

    def get_point(self, t: float) -> TrajectoryPoint:
        """
        根据时间t获取轨迹点（支持循环轨迹）
        参数：
            t: 当前时间（秒，相对于轨迹起始时间）
        返回：
            TrajectoryPoint: 当前时刻的轨迹信息
        """
        t_clamped = t % self.total_time  # 超过总时长后循环
        if self.trajectory_type == "circular":
            return self._circular_point(t_clamped)
        elif self.trajectory_type == "figure8":
            return self._figure8_point(t_clamped)
        elif self.trajectory_type == "square":
            return self._square_point(t_clamped)
        elif self.trajectory_type == "sine_cosine":
            return self._sine_cosine_point(t_clamped)
        else:
            raise ValueError(f"未知轨迹类型: {self.trajectory_type}")

    def _circular_point(self, t: float) -> TrajectoryPoint:
        """圆形轨迹点计算（XY平面圆周运动，Z轴固定）"""
        radius = self.params["radius"]       # 圆半径（米）
        z_offset = self.params["z_offset"]   # Z轴高度（米）
        freq = self.params["freq"]           # 频率（Hz，决定圆周运动快慢）
        omega = 2 * np.pi * freq             # 角频率（rad/s）

        # 位置（X=半径*cos(ωt), Y=半径*sin(ωt), Z=固定高度）
        x = radius * np.cos(omega * t) + self.p0[0]
        y = radius * np.sin(omega * t) + self.p0[1]
        z = z_offset + self.p0[2]

        # 速度（位置的一阶导数）
        vx = -radius * omega * np.sin(omega * t)
        vy = radius * omega * np.cos(omega * t)
        vz = 0.0  # Z轴速度为0

        # 加速度（速度的一阶导数）
        ax = -radius * omega**2 * np.cos(omega * t)
        ay = -radius * omega**2 * np.sin(omega * t)
        az = 0.0  # Z轴加速度为0

        return TrajectoryPoint(t, np.array([x, y, z]), np.array([vx, vy, vz]), np.array([ax, ay, az]))

    def _figure8_point(self, t: float) -> TrajectoryPoint:
        """8字轨迹点计算（XY平面8字形运动，Z轴固定）"""
        radius = self.params["radius"]       # 轨迹幅度（米）
        z_offset = self.params["z_offset"]   # Z轴高度（米）
        freq = self.params["freq"]           # 基础频率（Hz）
        omega = 2 * np.pi * freq             # 角频率（rad/s）

        # 位置（X=半径*sin(ωt), Y=半径*sin(2ωt), Z=固定高度）
        x = radius * np.sin(omega * t) + self.p0[0]
        y = radius * np.sin(2 * omega * t) + self.p0[1] # Y轴频率是X轴的2倍（形成8字）
        z = z_offset + self.p0[2]

        # 速度（位置的一阶导数）
        vx = radius * omega * np.cos(omega * t)
        vy = 2 * radius * omega * np.cos(2 * omega * t)  # Y轴速度频率为2ω
        vz = 0.0

        # 加速度（速度的一阶导数）
        ax = -radius * omega**2 * np.sin(omega * t)
        ay = -4 * radius * omega**2 * np.sin(2 * omega * t)  # Y轴加速度频率为2ω
        az = 0.0

        return TrajectoryPoint(t, np.array([x, y, z]), np.array([vx, vy, vz]), np.array([ax, ay, az]))

    def _square_point(self, t: float) -> TrajectoryPoint:
        """方形轨迹点计算（XY平面方形运动，带加减速平滑拐点）"""
        side_length = self.params["side_length"]  # 方形边长（米）
        z_offset = self.params["z_offset"]        # Z轴高度（米）
        speed = self.params["speed"]              # 匀速段速度（米/秒）
        t_acc = 0.5                                # 加减速时间（秒，控制拐点平滑度）
        a_max = speed / t_acc                      # 最大加速度（米/秒²）

        # 分段时间点（覆盖4边运动：右→前→左→后）
        t_segment = [
            0, t_acc, (side_length/speed) + t_acc,                  # 第1边（右向）：加速-匀速-减速
            (side_length/speed) + 2*t_acc, 2*(side_length/speed) + 2*t_acc,  # 第2边（前向）：加速-匀速-减速
            2*(side_length/speed) + 3*t_acc, 3*(side_length/speed) + 3*t_acc,  # 第3边（左向）：加速-匀速-减速
            3*(side_length/speed) + 4*t_acc, 4*(side_length/speed) + 4*t_acc   # 第4边（后向）：加速-匀速-减速
        ]
        x, y, vx, vy, ax, ay = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 初始化位置、速度、加速度
        z, vz, az = z_offset, 0.0, 0.0                       # Z轴固定，无速度/加速度

        # 分段计算轨迹（右→前→左→后）
        if t < t_segment[1]:               # 右向加速段（+X）
            ax = a_max                     # 加速度为+a_max
            vx = a_max * t                 # 速度线性增加
            x = 0.5 * a_max * t**2         # 位置二次增加
        elif t < t_segment[2]:             # 右向匀速段
            ax = 0.0                       # 加速度为0
            vx = speed                     # 速度保持speed
            x = 0.5 * a_max * t_acc**2 + speed * (t - t_segment[1])  # 位置线性增加
        elif t < t_segment[3]:             # 右向减速段（-X方向加速度）
            ax = -a_max                    # 加速度为-a_max
            vx = speed - a_max * (t - t_segment[2])  # 速度线性减小
            x = side_length - 0.5 * a_max * (t - t_segment[2])**2  # 位置接近边长

        elif t < t_segment[4]:             # 前向加速段（+Y）
            ay = a_max                     # 加速度为+a_max
            vy = a_max * (t - t_segment[3])  # 速度线性增加
            y = 0.5 * a_max * (t - t_segment[3])**2  # 位置二次增加
        elif t < t_segment[5]:             # 前向匀速段
            ay = 0.0                       # 加速度为0
            vy = speed                     # 速度保持speed
            y = 0.5 * a_max * t_acc**2 + speed * (t - t_segment[4])  # 位置线性增加
        elif t < t_segment[6]:             # 前向减速段（-Y方向加速度）
            ay = -a_max                    # 加速度为-a_max
            vy = speed - a_max * (t - t_segment[5])  # 速度线性减小
            y = side_length - 0.5 * a_max * (t - t_segment[5])**2  # 位置接近边长

        elif t < t_segment[7]:             # 左向加速段（-X）
            ax = -a_max                    # 加速度为-a_max
            vx = -a_max * (t - t_segment[6])  # 速度线性减小（负方向）
            x = side_length - 0.5 * a_max * (t - t_segment[6])**2  # 位置从边长减小
        elif t < t_segment[8]:             # 左向匀速段
            ax = 0.0                       # 加速度为0
            vx = -speed                    # 速度保持-speed（左向）
            x = side_length - speed * (t - t_segment[7])  # 位置线性减小

        # 后向（-Y）运动可根据需要扩展，示例中简化为回到起点

        return TrajectoryPoint(t, np.array([x + self.p0[0], y + self.p0[1], z + self.p0[2]]), np.array([vx, vy, vz]), np.array([ax, ay, az]))

    def _sine_cosine_point(self, t: float) -> TrajectoryPoint:
        """正余弦组合轨迹点计算（XYZ轴独立正余弦运动）"""
        amplitudes = self.params["amplitudes"]  # 各轴振幅（Ax, Ay, Az）
        freqs = self.params["freqs"]            # 各轴频率（fx, fy, fz）
        Ax, Ay, Az = amplitudes
        fx, fy, fz = freqs
        omega_x = 2 * np.pi * fx  # X轴角频率
        omega_y = 2 * np.pi * fy  # Y轴角频率
        omega_z = 2 * np.pi * fz  # Z轴角频率

        # 位置（各轴独立正余弦运动）
        x = Ax * np.sin(omega_x * t) + self.p0[0]
        y = Ay * np.cos(omega_y * t) + self.p0[1]
        z = Az * np.sin(omega_z * t) + self.p0[2]

        # 速度（位置的一阶导数）
        vx = Ax * omega_x * np.cos(omega_x * t)
        vy = -Ay * omega_y * np.sin(omega_y * t)
        vz = Az * omega_z * np.cos(omega_z * t)

        # 加速度（速度的一阶导数）
        ax = -Ax * omega_x**2 * np.sin(omega_x * t)
        ay = -Ay * omega_y**2 * np.cos(omega_y * t)
        az = -Az * omega_z**2 * np.sin(omega_z * t)

        return TrajectoryPoint(t, np.array([x, y, z]), np.array([vx, vy, vz]), np.array([ax, ay, az]))

    def plot_trajectory(self, save_path: str = "trajectory_plot.png") -> None:
        """绘制三维轨迹路径及各轴时间序列曲线（位置、速度）"""
        fig = plt.figure(figsize=(18, 18))  # 调整画布高度以适应3行布局
        fig.suptitle(f"{self.trajectory_type.capitalize()} Trajectory", fontsize=16, y=0.95)  # 总标题

        # 预生成轨迹数据（用于绘图）
        self.times = np.arange(0, self.total_time, self.dt)  # 时间序列
        self.points = [self.get_point(t) for t in self.times]  # 所有时刻的轨迹点
        self.positions = np.array([p.pos for p in self.points])  # 位置数组（N,3）
        self.velocities = np.array([p.vel for p in self.points])  # 速度数组（N,3）
        self.accelerations = np.array([p.acc for p in self.points])  # 加速度数组（N,3）


        # 子图1：三维轨迹路径（3行3列，第1个位置）
        ax3d = fig.add_subplot(3, 3, 1, projection='3d')  # 改为3行3列
        ax3d.plot(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], 'b-', linewidth=2)
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.set_title('3D Trajectory Path')

        # 子图2-4：X/Y/Z轴位置-时间曲线（3行3列，第2-4个位置）
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            ax = fig.add_subplot(3, 3, i+2)  # 改为3行3列，位置2-4
            ax.plot(self.times, self.positions[:, i], 'r-', linewidth=2)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{labels[i]} Position (m)')
            ax.set_title(f'{labels[i]} Position vs Time')
            ax.grid(True)

        # 子图5-7：X/Y/Z轴速度-时间曲线（3行3列，第5-7个位置）
        for i in range(3):
            ax = fig.add_subplot(3, 3, i+5)  # 改为3行3列，位置5-7（i=0→5, i=1→6, i=2→7）
            ax.plot(self.times, self.velocities[:, i], 'g-', linewidth=2)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{labels[i]} Velocity (m/s)')
            ax.set_title(f'{labels[i]} Velocity vs Time')
            ax.grid(True)

        # 调整子图间距（留出总标题空间）
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"轨迹图表已保存至：{save_path}")
        plt.close()


# ------------------- 示例使用 -------------------
if __name__ == "__main__":
    # 示例1：绘制圆形轨迹（半径3m，高度2m，频率0.3Hz）
    circle_gen = TrajectoryGenerator(
        trajectory_type="circular",
        total_time=10.0,  # 总时长10秒（约3个周期，因频率0.3Hz周期≈3.33秒）
        dt=0.01,          # 时间步长0.01秒（轨迹平滑）
        radius=3.0,
        z_offset=2.0,
        freq=0.3
    )
    circle_gen.plot_trajectory("scripts/data/circular_trajectory.png")

    # 示例2：绘制8字轨迹（幅度2m，高度1.5m，频率0.5Hz）
    figure8_gen = TrajectoryGenerator(
        trajectory_type="figure8",
        total_time=10.0,  # 总时长10秒（约5个周期，频率0.5Hz周期2秒）
        dt=0.01,
        radius=2.0,
        z_offset=1.5,
        freq=0.5
    )
    figure8_gen.plot_trajectory("scripts/data/figure8_trajectory.png")

    # 示例3：绘制方形轨迹（边长4m，高度2m，速度1.2m/s）
    square_gen = TrajectoryGenerator(
        trajectory_type="square",
        total_time=15.0,  # 总时长需覆盖4边运动（约12秒，留3秒冗余）
        dt=0.01,
        side_length=4.0,
        z_offset=2.0,
        speed=1.2
    )
    square_gen.plot_trajectory("scripts/data/square_trajectory.png")

    # 示例4：绘制正余弦组合轨迹（XYZ轴独立运动）
    sine_cosine_gen = TrajectoryGenerator(
        trajectory_type="sine_cosine",
        total_time=10.0,
        dt=0.01,
        amplitudes=(2.0, 3.0, 1.0),  # X/Y/Z轴振幅
        freqs=(0.2, 0.4, 0.1)         # X/Y/Z轴频率
    )
    sine_cosine_gen.plot_trajectory("scripts/data/sine_cosine_trajectory.png")
