import numpy as np
from typing import Union, Optional


class PTDO:
    """
    优化版Prescribed-Time Disturbance Observer（PTDO）
    核心改进：增加低通滤波强度、限制非线性项变化率、优化状态更新逻辑
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        T: float = 2.5,
        p: float = 0.2,
        m1: int = 8,
        m2: int = 8,
        k_d2: float = 10.,
        k_l2: float = 10.,
        k_e2: float = 10.,
        dim: int = 3,
        # 新增抗突变参数
        delta_filter_tau: float = 0.05,  # Delta低通滤波时间常数（s）
        dz_rate_limit: float = 30.0,    # dz变化率限制（1/s）
        z_delta_sat: float = 5.0        # z_delta饱和阈值（防止过饱和）
    ) -> None:
        # 参数校验
        if any([param <= 0 for param in [dt, T, p, m1, m2, k_d2, k_l2, k_e2, delta_filter_tau]]):
            raise ValueError("所有正参数必须大于0")
        if dim < 1:
            raise ValueError("维度参数必须≥1")
        
        # 公共参数
        self.dt = dt
        self.T = T
        self.p = p
        self.m1 = m1
        self.m2 = m2
        self.dim = dim
        self.z_delta_sat = z_delta_sat  # 新增：状态误差饱和阈值
        self.dz_rate_limit = dz_rate_limit  # 新增：状态微分变化率限制

        # 低通滤波系数计算（根据时间常数tau）
        self.delta_filter_alpha = np.exp(-dt / delta_filter_tau)  # 指数滤波系数（更合理的滤波参数）

        # 计算衍生增益
        self.k_d2 = k_d2
        self.k_d1 = self._compute_a(0.5, self.p, self.k_d2)
        
        self.k_l2 = k_l2
        self.k_l1 = self._compute_a(0.5, self.p, self.k_l2)
        
        self.k_e2 = k_e2
        self.k_e1 = self._compute_a(0.5, self.p, self.k_e2)
       
        # 观测器内部状态（初始化更合理）
        self.L = np.zeros(self.dim)  # 初始化为0避免初始突变
        self.Delta = np.zeros(self.dim)
        self.z = np.zeros(self.dim)  # 初始观测状态
        self.prev_dz = np.zeros(self.dim)  # 新增：上一时刻dz用于变化率限制
        self.z_delta = np.zeros(self.dim)  # 状态误差

    @staticmethod
    def _compute_a(k1: float, k2: float, b: float) -> float:
        numerator = (1 - k1) + k2 * np.log(1 + 1/b)
        denominator = (1 - k1) * k2
        if np.isclose(denominator, 0):
            raise ValueError("分母不能为0（k1不能等于1或k2不能为0）")
        return numerator / denominator

    def _saturate(self, x: Union[float, np.ndarray], sat: float) -> Union[float, np.ndarray]:
        """饱和函数：防止输入过饱和导致非线性项突变"""
        return np.clip(x, -sat, sat)

    def _h1(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """优化后的非线性函数：限制输入范围+平滑指数"""
        x_sat = self._saturate(x, self.z_delta_sat)  # 限制输入范围
        exponent = 1 + self.p + self.p * np.sign(np.abs(x_sat) - 1)
        # exponent = np.clip(exponent, 1.0, 3.0)  # 限制指数范围（防止指数过大）
        return np.tanh(x_sat) * np.abs(x_sat) ** exponent

    def _h2(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """优化后的非线性函数：限制输入范围+避免负数幂"""
        x_sat = self._saturate(x, self.z_delta_sat)  # 限制输入范围
        try:
            exponent = 1 + self.p + self.p * np.sign(np.abs(x_sat) - 1)
            # exponent = np.clip(exponent, 1.0, 3.0)  # 限制指数范围
            return np.sign(x_sat) * (np.abs(x_sat) ** exponent)
        except:
            return x_sat  # 异常时返回饱和后的值

    def _rate_limit(self, current: np.ndarray, previous: np.ndarray, limit: float) -> np.ndarray:
        """变化率限制函数：防止微分信号突变"""
        max_inc = limit * self.dt
        return np.clip(current, previous - max_inc, previous + max_inc)

    def update(self, syst_dynamic: np.ndarray, x: np.ndarray, u: np.ndarray) -> None:
        # 1. 计算状态误差（增加饱和限制）
        self.z_delta = self._saturate(x - self.z, self.z_delta_sat)
        
        # 2. 扰动估计更新（增强低通滤波）
        raw_delta = (
            self.L * np.tanh(self.m1 * self.z_delta)
            + (self.k_d1 / self.T) * (self.k_d2 * np.tanh(self.m1 * self.z_delta) + self._h1(self.z_delta))
        )
        # 指数加权低通滤波（比原0.99/0.01更合理）
        self.Delta = self.delta_filter_alpha * self.Delta + (1 - self.delta_filter_alpha) * raw_delta
        
        # 3. 观测器状态微分计算（增加变化率限制）
        raw_dz = syst_dynamic + self.Delta
        self.dz = self._rate_limit(raw_dz, self.prev_dz, self.dz_rate_limit)  # 限制dz变化率
        self.prev_dz = self.dz  # 保存当前dz用于下一时刻限制
        
        # 4. 非线性积分项更新（限制输入范围）
        self.dL = (
            self.z_delta * np.tanh(self.m1 * self.z_delta) 
            - (self.k_l1 / self.T) * (self.k_l2 * np.tanh(self.m2 * self._saturate(self.L, self.z_delta_sat)) + (2 + 2 * self.p) * self._h2(self.L))
        )

    def normal_update(self, sys_dynamic: np.ndarray, x: np.ndarray, u: np.ndarray) -> None:
        # 1. 计算状态误差（增加饱和限制）
        delta_z = self._saturate(self.z - x, self.z_delta_sat)
        
        # 2. 扰动估计直接计算（增加低通滤波）
        raw_delta = -self.L * np.tanh(10 * delta_z) - (np.tanh(delta_z) + self._h1(delta_z))
        self.Delta = self.delta_filter_alpha * self.Delta + (1 - self.delta_filter_alpha) * raw_delta
        
        # 3. 观测器状态微分计算（增加变化率限制）
        raw_dz = self.Delta + sys_dynamic
        self.dz = self._rate_limit(raw_dz, self.prev_dz, self.dz_rate_limit)
        self.prev_dz = self.dz
        
        # 4. 非线性积分项更新（限制输入范围）
        self.dL = np.abs(delta_z)  # 简化形式保留，但输入已饱和

    def observe(self, system_dynamic: np.ndarray, x: np.ndarray, u: np.ndarray, config: str = 'PD') -> np.ndarray:
        # 输入维度校验
        if x.shape != (self.dim,) or u.shape != (self.dim,) or system_dynamic.shape != (self.dim,):
            raise ValueError(f"输入维度需为({self.dim},)，当前x={x.shape}, u={u.shape}, system_dynamic={system_dynamic.shape}")
        
        # 修正配置判断逻辑（原代码可能写反了）
        if config == 'PD':
            self.update(system_dynamic, x, u)  # PD模式使用改进的update方法
        elif config == 'normal':
            self.normal_update(system_dynamic, x, u)  # normal模式使用简化方法
        else:
            raise ValueError(f"无效的观测器类型config='{config}'，仅支持'PD'或'normal'")
        
        # 离散时间积分更新状态（增加积分限幅）
        self.z = self._saturate(self.z + self.dz * self.dt, 10.0 * self.z_delta_sat)  # 限制z的范围
        self.L = self._saturate(self.L + self.dL * self.dt, 10.0 * self.z_delta_sat)  # 限制L的范围
        
        return self.Delta

    def set_init(self, x: np.ndarray) -> None:
        """优化初始化：将观测状态初始化为实际状态，减少初始误差"""
        if x.shape != (self.dim,):
            raise ValueError(f"初始状态维度需为({self.dim},)，当前x={x.shape}")
        self.z = x.copy()  # 初始观测状态设为实际状态值
        self.L = np.zeros(self.dim)  # 积分项保持0初始化
        self.Delta = np.zeros(self.dim)  # 扰动估计初始化为0
        self.prev_dz = np.zeros(self.dim)  # 初始微分状态
    