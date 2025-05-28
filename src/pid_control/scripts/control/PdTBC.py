import numpy as np
from observer.Filter import FirstOrderFilter
from observer.Filter import RK4DynamicFilter

import numpy as np

def sig(x, a, kt=100.):
    return np.fabs(x) ** a * np.sign(kt*x)

def calculate_k1(p1, p2, k2):
    numerator = (1 - p1) + p2 * np.log(1 + 1/k2)
    denominator = (1 - p1) * p2
    return 3 ** p2 * numerator / denominator

class PdTBC:
    def __init__(self,
                 
                 dt: float = 0.01,                                  # Control period (s)
                 ctrl0: np.ndarray = np.array([0.0, 0.0, 0.0]),      # Initial control value

                 # Virtual control law parameters (phase 1)
                 k_eta12: np.ndarray = np.array([0.5, 0.5, 0.5]),    # Saturation function gain (Equation 49)
                 l_eta1: np.ndarray = np.array([15., 15., 15.]),     # Saturation function parameter (Equation 49)
        
                 # Virtual control law parameters (phase 2)
                 k_eta22: np.ndarray = np.array([2.0, 2.0, 2.0]),    # Saturation function gain (Equation 52)
                 l_eta2: np.ndarray = np.array([5, 5, 5]),           # Saturation function parameter (Equation 52)
        
                 # Adaptive term parameters (Equation 53)
                 l_eta3: np.ndarray = np.array([5, 5, 5]),           # Adaptive term saturation parameter 1 (Equation 53)
                 l_eta4: np.ndarray = np.array([5, 5, 5]),           # Adaptive term saturation parameter 2 (Equation 53)
                 k_eta31: np.ndarray = np.array([1.0, 1.0, 1.0]),    # Adaptive gain 1 (Equation 53)
                 k_eta32: np.ndarray = np.array([1.0, 1.0, 1.0]),   # Adaptive gain 2 (Equation 53)
        
                 # Common parameters
                 p_eta: float = 0.3,                                 # Nonlinear term exponent (Equation 49/52)
                 T_eta: float = 2.0,                                 # Fixed time constant (Equation 49/52)

                 uf_max: float = 35.0,                               # Maximum thrust limit (N)
                 g: float = 9.8,                                     # Gravitational acceleration (m/s²)
                 k_t: float = 1e-3,                                  # Translational damping coefficient
                 m: float = 1.5):                                    # UAV mass (kg)
        
        self.k_t = k_t          # Damping coefficient
        self.m = m              # Mass

        self.dt = dt            # Control period
        self.g = g              # Gravitational acceleration
        self.control = ctrl0
        self.uf_max = uf_max    # Maximum thrust
        
        self.p = p_eta          # Nonlinear term exponent (abbreviated as p)
        self.T = T_eta          # Fixed time constant
        
        # Control law gains (maintain paper symbol system with annotation)
        self.k12 = k_eta12      # Phase 1 saturation gain (Equation 49: k_eta12)
        self.k11 = calculate_k1(p1=0.5, p2=self.p, k2=self.k12)
        self.l1 = l_eta1        # Phase 1 saturation parameter (Equation 49: l_eta1)
        print('k11:', self.k11)
        print('k12:', self.k12)
        print('l1:', self.l1)

        self.k22 = k_eta22      # Phase 2 saturation gain (Equation 52: k_eta22)
        self.k21 = calculate_k1(p1=0.5, p2=self.p, k2=self.k22)
        self.l2 = l_eta2        # Phase 2 saturation parameter (Equation 52: l_eta2)
        print('k21:', self.k21)
        print('k22:', self.k22)
        print('l2:', self.l2)

        self.l3 = l_eta3        # Adaptive term saturation parameter 1 (Equation 53: l_eta3)
        self.l4 = l_eta4        # Adaptive term saturation parameter 2 (Equation 53: l_eta4)
        self.k31 = k_eta31      # Adaptive gain 1 (Equation 53: k_eta3i)
        self.k32 = k_eta32    # Adaptive gain 2 (Equation 53: k_eta32i)

        # Controller states
        self.e_eta = np.zeros(3)
        self.e_eta_d = np.zeros(3)
        self.e_eta2 = np.zeros(3)

        self.alpha_data = []
        self.alpha_d_date = []
        
        # Adaptive parameter 
        self.L = np.ones(3)*0.2
        
        # Intergral compensator
        self.eta_i = np.zeros(3)
        
        # Differentiator initialization (for virtual control law filtering)
        self.alpah_obs = FirstOrderFilter(dt=dt,kt=np.ones(3)*100)  # Fixed time differentiator

    def vt(self, ref_vel):
        h_sig = 1 + self.p + self.p * np.sign(np.fabs(self.e_eta)-1)
        return ref_vel - self.k11 / self.T * (self.k12 * np.tanh(self.l1*self.e_eta) + sig(x=self.e_eta, a=h_sig, kt=100.))
    
    def vt_dot(self, ref_acc):
        h_sig = 1 + self.p + self.p * np.sign(np.fabs(self.e_eta)-1)
        return ref_acc - self.k11 / self.T *(self.k12 * self.l1 * (1 - np.tanh(self.l1*self.e_eta)**2)*self.e_eta_d
                                             + h_sig * sig(x=self.e_eta, a=h_sig, kt=100.) * self.e_eta_d)

    def vt2(self):
        h_sig = 1 + self.p + self.p * np.sign(np.fabs(self.e_eta2)-1)
        return -self.k21 / self.T *(self.k22 * np.tanh(self.l2 * self.e_eta2) + sig(x=self.e_eta2, a=h_sig, kt=100.))
    
    def update_L(self):
        h_sig = 1 + self.p + self.p * np.sign(np.fabs(self.L)-1)
        return (np.tanh(self.l3*self.e_eta2)*self.e_eta2 - self.k31/self.T * (self.k32*np.tanh(self.l4*self.L) + (2+2*self.p)*self.L**h_sig)) * self.dt
    
    def update(self, 
               state: np.ndarray, 
               ref_state: np.ndarray,
               obs: np.ndarray=np.zeros(3)):
        eta = state[0:3]
        eta_d = state[3:6]

        ref = ref_state[0:3]
        ref_vel = ref_state[3:6]
        ref_acc = ref_state[6:9]

        self.e_eta = (eta - ref).clip(-np.ones(3), np.ones(3))*0.7
        self.e_eta_d = (eta_d - ref_vel).clip(-7.5*np.ones(3), 7.5*np.ones(3))

        alpha = self.vt(ref_vel=ref_vel)
        alpha_hat, alpah_d_hat = self.alpah_obs.update(x=alpha)
        alpha_d = self.vt_dot(ref_acc=ref_acc).clip(-20*np.ones(3), 20*np.ones(3))

        self.e_eta2 = (eta_d - alpha)
        vt2 = self.vt2()
        self.L += self.update_L()

        self.eta_i = 1. * self.eta_i + (self.e_eta + 0.1*np.tanh(20*self.e_eta))*self.dt

        term1 = - self.e_eta + self.k_t / self.m * eta_d - 0.1* self.e_eta2
        term2 = - obs + alpha_d
        term3 = - self.L * np.tanh(self.l3*self.e_eta2) * 1. + self.eta_i*0.00

        # print(alpah_d, 'alpha_d')

        # print(term1, 'term1')
        # print(term2, 'term2')
        # print(term3, 'term3')
        
        self.control = (term1 + term2 + vt2 + term3)#.clip(-20*np.ones(3), 20*np.ones(3))
