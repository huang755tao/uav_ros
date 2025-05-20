import numpy as np
from typing import Tuple  # 添加类型模块导入
from utils.Rk4 import rk4_step


def sig(x: np.ndarray, a: float, kt: float = 5.0) -> np.ndarray:
    """自定义饱和函数"""
    return np.fabs(x) ** a * np.tanh(kt * x)


class DynamicFilter:
    """二阶动态微分跟踪器（固定增益结构）"""
    
    def __init__(self, 
                 dt: float, 
                 r11: float = 20.0, 
                 r21: float = 50.0) -> None:
        self.r11 = r11
        self.r21 = r21
        self.dt = dt
        self.z1 = np.zeros(3, dtype=np.float32)
        self.dot_z1 = np.zeros(3, dtype=np.float32)
        self.z2 = np.zeros(3, dtype=np.float32)
        self.e = np.zeros(3, dtype=np.float32)

    def _compute_first_derivative(self, x: np.ndarray) -> np.ndarray:
        e = x - self.z1
        return self.z2 + self.r11 * (np.sign(e) * np.sqrt(np.fabs(e)) ** 0.5 + 2 * np.sign(e) * (np.fabs(e) ** 1.5))

    def _compute_second_derivative(self, x: np.ndarray) -> np.ndarray:
        e = x - self.z1
        return self.r21 * (0.5 * np.tanh(20 * e) + 4 * 2 * e + 4 * 1.5 * np.fabs(e) ** 2 * np.tanh(100*e))

    def update(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # 修改此处
        """返回类型改为typing.Tuple"""
        self.dot_z1 = self._compute_first_derivative(x)
        dot_z2 = self._compute_second_derivative(x)

        self.z1 += self.dot_z1 * self.dt
        self.z2 += dot_z2 * self.dt

        self.e = x - self.z1
        # print(self.e, 'td e')
        return self.z1.copy(), self.z2.copy()
    
class RK4DynamicFilter:
    """基于RK4积分的二阶动态跟踪器"""
    def __init__(self, 
                 dt: float, 
                 r11: float = 20.0,  # 保持原参数
                 r21: float = 50.0) -> None:
        self.r11 = r11       # 位置误差增益（原变量）
        self.r21 = r21       # 速度误差增益（原变量）
        self.dt = dt         # 时间步长（原变量）
        self.z1 = np.zeros(3, dtype=np.float32)  # 位置估计（原变量）
        self.dot_z1 = np.zeros(3, dtype=np.float32)  # 速度估计（原变量）
        self.z2 = np.zeros(3, dtype=np.float32)  # 二阶状态变量（原变量）
        self.e = np.zeros(3)

    def dynamics(self, t: float, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        跟踪器动力学模型（与rk4_step接口兼容）
        Args:
            t: 当前时间（未使用，保留接口）
            state: 状态向量 [z1_x, z1_y, z1_z, z2_x, z2_y, z2_z]
            u: 控制输入（目标位置 [x_ref, y_ref, z_ref]）
        Returns:
            d_state: 状态导数 [dz1_x/dt, dz1_y/dt, dz1_z/dt, dz2_x/dt, dz2_y/dt, dz2_z/dt]
        """
        # 从状态向量中解包原变量
        z1 = state[:3]  # 位置估计（原变量）
        z2 = state[3:]  # 二阶状态变量（原变量）
        
        # 计算误差（与原逻辑一致）
        e = u - z1
        print(e, 'TD error')
        # 计算一阶导数（dz1/dt = dot_z1，与原逻辑一致）
        dot_z1 = self._compute_first_derivative(z2, e)  # 注意：调整了参数传递方式
        
        # 计算二阶导数（dz2/dt，与原逻辑一致）
        dot_z2 = self._compute_second_derivative(e)
        
        # 合并导数向量（与state结构对应）
        d_state = np.concatenate([dot_z1, dot_z2])
        return d_state

    def _compute_first_derivative(self, z2: np.ndarray, e: np.ndarray) -> np.ndarray:
        """原一阶导数计算逻辑"""
        return z2 + self.r11 * (np.sign(e) * np.sqrt(np.fabs(e)) ** 0.5 + 2 * np.sign(e) * (np.fabs(e) ** 1.5))

    def _compute_second_derivative(self, e: np.ndarray) -> np.ndarray:
        """原二阶导数计算逻辑"""
        return self.r21 * (0.5 * np.tanh(50 * e) + 4 * 2 * e + 4 * 1.5 * np.fabs(e) ** 2 * np.sign(e))

    def update(self, x_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用通用rk4_step更新状态
        Args:
            x_ref: 目标位置
        Returns:
            (z1, dot_z1): 位置估计和速度估计
        """
        # 封装状态向量（[z1, z2]）
        state = np.concatenate([self.z1, self.z2])
        
        # 调用通用RK4积分函数
        new_state = rk4_step(
            dynamics=self.dynamics,
            x=state,
            t=0,          # 时间（未使用）
            u=x_ref,      # 控制输入：目标位置
            dt=self.dt    # 时间步长
        )
        
        # 解包更新后的状态向量
        self.z1 = new_state[:3]  # 更新位置估计（原变量）
        self.z2 = new_state[3:]  # 更新二阶状态变量（原变量）
        
        # 重新计算速度估计（与原逻辑一致）
        e = x_ref - self.z1
        self.e = e
        self.dot_z1 = self._compute_first_derivative(self.z2, e)
        
        return self.z1.copy(), self.z2.copy()


class FirstOrderFilter:
    """一阶惯性微分跟踪器（可变增益结构）"""
    
    def __init__(self, 
                 dt: float, 
                 dim: int = 3, 
                 kt: np.ndarray = np.zeros(3, dtype=np.float32)) -> None:
        self.dt = dt
        self.k1 = kt.astype(np.float32) if kt.size else np.zeros(dim, dtype=np.float32)
        self.z1 = np.zeros(dim, dtype=np.float32)
        self.dot_z1 = np.zeros(dim, dtype=np.float32)

    def update(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # 修改此处
        """返回类型改为typing.Tuple"""
        self.dot_z1 = self.k1 * (x - self.z1)
        self.z1 += self.dot_z1 * self.dt
        return self.z1.copy(), self.dot_z1.copy()
