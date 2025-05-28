import numpy as np

class PidControl(object):
    def __init__(self,
                 dt: float=0.01,
                 kp_pos: np.ndarray=np.zeros(3),
                 ki_pos: np.ndarray=np.zeros(3),
                 kd_pos: np.ndarray=np.zeros(3),
                 kp_vel: np.ndarray=np.zeros(3),
                 ki_vel: np.ndarray=np.zeros(3),
                 kd_vel: np.ndarray=np.zeros(3),
                 kp_att: np.ndarray=np.zeros(3),
                 ki_att: np.ndarray=np.zeros(3),
                 kd_att: np.ndarray=np.zeros(3),
                 p_v: np.ndarray=np.ones(3),
                 p_a: np.ndarray=np.ones(3),
                 p_r: np.ndarray=np.ones(3)):

        " init model "
        self.dt = dt
        " init model "
        
        " init control para "
        self.p_v = p_v
        self.p_a = p_a
        self.p_r = p_r
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos

        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.kd_vel = kd_vel

        " init model "
        self.kp_att = kp_att
        self.ki_att = ki_att
        self.kd_att = kd_att

        self.p_v = p_v
        self.p_a = p_a
        self.p_r = p_r
        " init control para "
        " lation state "
        self.err_p_pos = np.zeros(3)
        self.err_i_pos = np.zeros(3)
        self.err_d_pos = np.zeros(3)

        self.err_p_vel = np.zeros(3)
        self.err_i_vel = np.zeros(3)
        self.err_d_vel = np.zeros(3)

        self.err_p_att = np.zeros(3)
        self.err_i_att = np.zeros(3)
        self.err_d_att = np.zeros(3)

        self.err_control = np.zeros(3)
        self.control = np.zeros(3)
        
        self.att_control = np.zeros(3)
        " simulation state "
    
    def para(self):
        print('Para for fix PID',
            'kp_pos:', self.kp_pos,
            'ki_pos:', self.ki_pos,
            'kd_pos:', self.kd_pos,
            'kp_vel:', self.kp_vel,
            'ki_vel:', self.ki_vel,
            'kd_vel:', self.kd_vel,
            'alpha_pos:', self.p_v,
            'alpha_vel:', self.p_a)
        
    def update_att(self, 
                   att: np.ndarray,
                   pqr: np.ndarray,
                   ref_att: np.ndarray = np.zeros(3),
                   ref_att_v: np.ndarray = np.zeros(3)):
        " Attitude loop "
        # print(att, 'att')
        
        self.err_p_att = (ref_att - att)# .clip(np.array([-1.57/3, -1.57/3, -1.57/3]),np.array([1.57/3, 1.57/3, 1.57/3]))
        
        for i in range(3):
            self.err_i_att[i] += abs(self.err_p_att[i].clip(-0.1, 0.1))**self.p_r[i]*np.tanh(20*self.err_p_att[i])*self.dt
        
        self.err_d_att = ref_att_v - pqr

        att_v = np.zeros(3)
        for i in range(3):
            att_v[i] = self.kp_att[i] * abs(self.err_p_att[i])**(self.p_r[i])*np.tanh(20*self.err_p_att[i])
            + self.ki_att[i] * self.err_i_att[i] + self.kd_att[i] * abs(self.err_d_att[i])**(2*self.p_r[i]/(1+self.p_r[i]))*np.tanh(10*self.err_p_att[i])
            
        att_v = att_v.clip(np.array([-2, -2, -2]), np.array([2, 2, 2])) + 0.*ref_att_v
        self.att_control = att_v 
        return self.att_control[0], self.att_control[1], self.att_control[2]
        " Attitude loop "        
    
    def update_adpative(self,
               state: np.ndarray,
               ref_state: np.ndarray=np.zeros(9)):
        " position loop "
        self.err_p_pos = (state[0: 3] - ref_state[0: 3]) # .clip(np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]))
        alpha = - self.kp_pos * self.err_p_pos + ref_state[3: 6] 
        d_alpha = -self.kp_pos * self.err_p_pos + ref_state[6: 9]
        " position loop "
        " v "
        self.err_d_pos = state[3: 6] - alpha
        self.control = - self.err_p_pos - self.kp_pos * self.err_d_pos - self.kp_pos * alpha + self.kp_pos * ref_state[3: 6] + ref_state[6: 9] - self.kp_vel*self.err_d_pos - np.array([0.1, 0.1, 2])*np.tanh(50*self.err_d_pos) 
        " v "
    
    def update_dual(self,
               state: np.ndarray,
               ref_state: np.ndarray=np.zeros(9)):
        " position loop "
        self.err_p_pos = np.tanh(0.5*(ref_state[0: 3] - state[0: 3]))
        self.err_p_pos = (ref_state[0: 3] - state[0: 3]).clip(np.array([-1, -1, -1]), np.array([1, 1,1]))+0.1*np.tanh(20*(ref_state[0: 3] - state[0: 3]))#.clip(np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]))
        
        for i in range(3):
            self.err_i_pos[i] += abs(self.err_p_pos[i].clip(-0.5, 0.5))**self.p_v[i]*np.tanh(100*self.err_p_pos[i])*self.dt

        self.err_d_pos = 10 * np.tanh(0.1*(ref_state[3: 6] - state[3: 6])/self.dt)
        # self.err_d_pos = (ref_state[3: 6] - state[3: 6])+np.tanh(10*(ref_state[3: 6] - state[3: 6]))  

        pos_vel = np.zeros(3)
        # for i in range(3):
        #     pos_vel[i] = self.kp_pos[i] * abs(self.err_p_pos[i])**self.p_v[i]*np.tanh(100*self.err_p_pos[i]) \
        #           + self.ki_pos[i] * self.err_i_pos[i] \
        #           + self.kd_pos[i] * abs(self.err_d_pos[i])**(2*self.p_v[i]/(1+self.p_v[i]))*np.tanh(100*self.err_d_pos[i])

        for i in range(3):
            pos_vel[i] = self.kp_pos[i] * self.err_p_pos[i] \
                  + self.ki_pos[i] * self.err_i_pos[i]*0 \
                  + self.kd_pos[i] * abs(self.err_d_pos[i])**(2*self.p_v[i]/(1+self.p_v[i]))*np.tanh(100*self.err_d_pos[i])
        # print(pos_vel, 'pos_vel')
            
        pos_vel = pos_vel + ref_state[3: 6]
        " position loop "

        " Velocity loop "
        self.err_p_vel = (pos_vel - state[3: 6]).clip(np.array([-5., -5., -5.]), np.array([5., 5., 5.]))
        err_vel = pos_vel - state[3: 6]
        for i in range(3):
            self.err_i_vel[i] = self.err_i_vel[i] + 2*np.exp(-(err_vel[i]/0.5)**1) * err_vel[i] * self.dt
        self.err_d_vel = (pos_vel - state[3: 6])/self.dt

        vel_a = np.zeros(3)
        # for i in range(3):
        #     vel_a[i] = self.kp_vel[i] * abs(self.err_p_vel[i])**(self.p_a[i])*np.tanh(50*self.err_p_vel[i]) \
        #         + self.ki_vel[i] * self.err_i_vel[i] \
        #         + self.kd_vel[i] * abs(self.err_d_vel[i])**(2*self.p_a[i]/(1+self.p_a[i]))*np.tanh(100*self.err_p_vel[i])

        for i in range(3):
            vel_a[i] = self.kp_vel[i] * self.err_p_vel[i] \
                + self.ki_vel[i] * self.err_i_vel[i] \
                + self.kd_vel[i] * abs(self.err_d_vel[i])**(2*self.p_a[i]/(1+self.p_a[i]))*np.tanh(100*self.err_p_vel[i])
            
        vel_a += ref_state[6: 9]
        vel_a = vel_a.clip(np.array([-20, -20, -20]), np.array([20, 20, 20]))
        self.control = 0.*self.control + 1*vel_a 
        " Velocity loop "

    def update(self,
            state: np.ndarray,
            ref_state: np.ndarray=np.zeros(9)) -> np.ndarray:
        self.err_p_pos = (ref_state[0: 3] - state[0: 3])
        
        for i in range(3):
            self.err_i_pos[i] = self.err_i_pos[i] + self.err_p_pos[i]*self.dt
        # self.err_i_pos += self.err_p_pos * self.dt 

        # self.err_d_pos = 10 * np.tanh(0.1*(ref_state[3: 6] - state[3: 6])/self.dt)
        self.err_d_pos = (ref_state[3: 6] - state[3: 6])  

        a_vel = np.zeros(3)
        for i in range(3):
            a_vel[i] = self.kp_pos[i] * self.err_p_pos[i] 
            + self.ki_pos[i] * self.err_i_pos[i] 
            + self.kd_pos[i] * self.err_d_pos[i]
        # print(pos_vel, 'pos_vel')
            
        a_vel = a_vel.clip(np.array([-10, -10, -10]), np.array([10, 10, 10]))  
        # a_vel +=  ref_state[6: 9]
        self.err_control = 0.05 * self.err_control + 0.95 * a_vel
        self.control = self.err_control + ref_state[6: 9]
        
    def update_back(self, flag: int, state: np.ndarray,
                    ref_state: np.ndarray, v2: np.ndarray):
        self.err_p_pos = (ref_state[0: 3] - state[0: 3]).clip(np.array([-1.5, -1.5, -0.5]), np.array([1.5, 1.5, 0.5]))
        self.err_i_pos = self.err_i_pos*np.array([.999, .999, 1.]) + self.err_p_pos.clip(np.array([-0.2, -0.2, -0.25]), np.array([0.2, .2, .25]))*self.dt
        if flag==1:
            err_p_pos = self.err_p_pos + np.array([0.25, 0.25, 0.15])*np.tanh(np.array([15, 15, 5])*self.err_p_pos) + 0.02*np.tanh(np.array([50, 50, 10])*self.err_p_pos)
        else:
            err_p_pos = self.err_p_pos + np.array([0.1, 0.1, 0.15])*np.tanh(np.array([5, 5, 5])*self.err_p_pos) + 0.02*np.tanh(np.array([50, 50, 10])*self.err_p_pos)
            
        self.err_d_pos = (ref_state[3: 6] - state[3: 6])#+0.1*np.tanh(2*(ref_state[3: 6] - state[3: 6]))
        
        self.err_control = (1 + self.ki_pos+self.kp_pos**2*self.kd_pos+v2)*err_p_pos+(self.kp_pos**2+self.kd_pos)*self.err_d_pos+self.ki_pos*self.kd_pos*self.err_i_pos
        
        self.control = self.err_control + ref_state[6: 9]
        
        
        
    def reset(self):
        " simulation state "
        self.err_p_pos = np.zeros(3)
        self.err_i_pos = np.zeros(3)
        self.err_d_pos = np.zeros(3)

        self.err_p_vel = np.zeros(3)
        self.err_i_vel = np.zeros(3)
        self.err_d_vel = np.zeros(3)
        " simulation state "
