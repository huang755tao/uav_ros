import numpy as np
import matplotlib.pyplot as plt
from utils.Rk4 import rk4_step  # 导入独立RK4模块


class SecondOrderSystem:
    """
    带扰动的稳定二阶系统（状态空间形式），使用外部RK4模块积分
    """
    def __init__(self, omega_n=2.0, zeta=0.7, x0=None, disturbance=None):
        # 参数校验
        if omega_n <= 0 or zeta <= 0:
            raise ValueError("自然频率和阻尼比必须大于0")
        
        self.omega_n = omega_n  # 自然频率 (rad/s)
        self.zeta = zeta        # 阻尼比
        self.disturbance = disturbance if disturbance else lambda t: 0.0  # 扰动函数
        
        # 初始化状态
        self.x = np.array([0.0, 0.0]) if x0 is None else np.array(x0, dtype=float)
        self.dynamic = 0.0
        self.t = 0.0  # 当前时间

    def dynamics(self, t: float, x: np.ndarray, u: float) -> np.ndarray:
        """
        系统动力学方程（供RK4调用）
        
        Args:
            t: 当前时间（s）
            x: 当前状态[x₁, x₂]（位移, 速度）
            u: 控制输入
        
        Returns:
            dx/dt: 状态导数
        """
        x1, x2 = x
        d = self.disturbance(t)  # 获取当前扰动
        
        # 二阶系统微分方程
        dx1_dt = x2
        dx2_dt = -self.omega_n**2 * x1 - 2 * self.zeta * self.omega_n * x2 + self.omega_n**2 * u + d
        self.dynamic = -self.omega_n**2 * x1 - 2 * self.zeta * self.omega_n * x2 + self.omega_n**2 * u 
        return np.array([dx1_dt, dx2_dt])

    def step(self, u: float, dt: float) -> None:
        """
        使用外部RK4模块更新状态
        
        Args:
            u: 控制输入
            dt: 时间步长（s）
        """
        # 调用RK4模块（注意动力学函数需符合rk4_step的参数要求）
        self.x = rk4_step(
            dynamics=self.dynamics,  # 传入系统动力学函数
            x=self.x,                # 当前状态
            t=self.t,                # 当前时间
            u=u,                     # 控制输入
            dt=dt                    # 时间步长
        )
        self.t += dt


def simulate():
    # ==================== 系统参数 ====================
    omega_n = 2.0       # 自然频率（rad/s）
    zeta = 1.7          # 阻尼比（欠阻尼）
    dt = 0.005           # 仿真时间步长（s）
    total_time = 50.0   # 总仿真时间（s）
    steps = int(total_time / dt)  # 总步数
    
    # 初始状态和扰动定义
    x0 = np.array([0.0, 0.0])  # [初始位移, 初始速度]
    def disturbance(t):
        return 0.5 * np.sin(1.5 * t) if t > 5 else 0.0  # 5秒后正弦扰动

    # 初始化系统
    system = SecondOrderSystem(
        omega_n=omega_n,
        zeta=zeta,
        x0=x0,
        disturbance=disturbance
    )

    # 控制输入（阶跃输入：t≥1s时u=1.0）
    def control_input(t):
        return 1.0 if t >= 1.0 else 0.0

    # ==================== 仿真循环 ====================
    time = []
    x1_list = []  # 位移
    x2_list = []  # 速度
    u_list = []   # 控制输入
    d_list = []   # 扰动

    for _ in range(steps + 1):
        t = system.t
        u = control_input(t)
        
        # 记录数据
        time.append(t)
        x1_list.append(system.x[0])
        x2_list.append(system.x[1])
        u_list.append(u)
        d_list.append(system.disturbance(t))
        
        # 调用RK4模块更新状态
        system.step(u, dt)

    # 转换为numpy数组
    time = np.array(time)
    x1 = np.array(x1_list)
    x2 = np.array(x2_list)
    u = np.array(u_list)
    d = np.array(d_list)

    # ==================== 结果可视化 ====================
    plt.figure(figsize=(14, 8))
    plt.suptitle(f'Second-Order System Response (ωₙ={omega_n}, ζ={zeta})', fontsize=14)

    plt.subplot(2, 2, 1)
    plt.plot(time, x1, 'b-', label='Displacement (x₁)')
    plt.plot(time, u, 'r--', label='Control Input (u)')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('Displacement vs Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(time, x2, 'g-', label='Velocity (x₂)')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(time, d, 'm-', label='Disturbance (d)')
    plt.xlabel('Time (s)')
    plt.ylabel('Disturbance (N)')
    plt.title('Disturbance vs Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x1, x2, 'c-', alpha=0.7)
    plt.xlabel('Displacement (x₁)')
    plt.ylabel('Velocity (x₂)')
    plt.title('Phase Portrait')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate()
