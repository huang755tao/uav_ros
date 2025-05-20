import numpy as np
from typing import Callable


def rk4_step(
    dynamics: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    t: float,
    u: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    四阶龙格-库塔（RK4）积分单步更新
    
    Args:
        dynamics: 动力学函数，形式为 dx/dt = dynamics(t, x, u)
        x: 当前状态（n维向量）
        t: 当前时间（s）
        u: 控制输入（m维向量）
        dt: 时间步长（s）
    
    Returns:
        x_next: 更新后的状态（n维向量）
    """
    # 计算四个斜率项
    k1 = dynamics(t, x, u)
    k2 = dynamics(t + dt/2, x + dt/2 * k1, u)
    k3 = dynamics(t + dt/2, x + dt/2 * k2, u)
    k4 = dynamics(t + dt, x + dt * k3, u)
    
    # RK4积分公式
    x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x_next
