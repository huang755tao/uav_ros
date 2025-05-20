import numpy as np
from observer.Filter import FirstOrderFilter
from observer.Filter import RK4DynamicFilter

import numpy as np

class BC:
    def __init__(self,
                 
                 dt: float = 0.01,                                  # Control period (s)
                 ctrl0: np.ndarray = np.array([0.0, 0.0, 0.0]),      # Initial control value

                 k1: np.ndarray = np.array([0.5, 0.5, 0.5]), 
        
                 k2: np.ndarray = np.array([2.0, 2.0, 2.0]), 
                                 
                 uf_max: float = 35.0,                               # Maximum thrust limit (N)
                 g: float = 9.8,                                     # Gravitational acceleration (m/s²)
                 kt: float = 1e-3,                                  # Translational damping coefficient
                 m: float = 1.5):
        self.dt = dt
        self.m = m
        self.kt = kt
        self.g = g              # Gravitational acceleration
        self.control = ctrl0
        
        self.k1 = k1
        self.k2 = k2

        self.e = np.zeros(3)
        self.e_d = np.zeros(3)
        self.e2 = np.zeros(3)

    def vt(self, ref_vel: np.ndarray):
        return ref_vel - self.k1 * self.e
    
    def vt_dot(self, ref_acc: np.ndarray):
        return ref_acc - self.k1 * self.e_d
    
    def update(self, 
               state: np.ndarray,
               ref_state: np.ndarray,
               obs: np.ndarray = np.zeros(3)):
        eta = state[0:3]
        eta_d = state[3:6]

        ref = ref_state[0:3]
        ref_vel = ref_state[3:6]
        ref_acc = ref_state[6:9]

        self.e = (eta - ref).clip(-np.ones(3), np.ones(3))
        self.e_d = (eta_d - ref_vel).clip(-5*np.ones(3), 5*np.ones(3))

        alpha = self.vt(ref_vel=ref_vel)
        alpha_d = self.vt_dot(ref_acc=ref_acc)

        self.e2 = eta_d - alpha

        term1 = - self.e + self.kt / self.m * eta_d
        term2 = - obs + alpha_d
        term3 = - self.k2 * self.e2

        self.control = term1 + term2 + term3 