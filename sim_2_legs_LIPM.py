import numpy as np
import matplotlib.pyplot as plt
import time

from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from typing import List

g = 9.81

class Field:
    def __init__(self, time_width:float=10.0, resol:int=500, size:float=10.0):
        self.size   : float = size
        self.p_time : float = None

        self.t_span = (0, time_width)
        self.t_eval = np.linspace(*self.t_span, resol)

        self.init_field()
    
    def set_field_size(self):
        self.ax.set_aspect('equal')
        self.ax.set_xlim([-self.size, self.size])
        self.ax.set_ylim([-self.size, self.size])

    def update(self, frame):
        self.ax.clear()
        self.set_field_size()
        self.frame_callback(frame)
    
    def init_field(self):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        self.set_field_size()

    def start(self):
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.ani = FuncAnimation(
            self.fig, 
            lambda i : self.update(i),
            frames   = len(self.t_eval), 
            blit     = False, 
            interval = 0.01
        )

    def get_axis(self) -> Axes:
        return self.ax
    
    def spin(self):
        plt.show()

    def on_close(self, event):
        self.ani.event_source.stop()
        plt.close('all')

    def frame_callback(frame):
        pass

class DoublePendulum:
    def __init__(self):
        self.m1 = 1.0
        self.m2 = 1.0
        self.m3 = 1.0
        self.m4 = 1.0

        self.l1 = 1.0
        self.l2 = 1.0
        self.l3 = 1.0
        self.l4 = 1.0

        self.th   : np.ndarray = np.array([ [np.pi/6], [np.pi/6], [5*np.pi/6], [5*np.pi/6] ])
        self.dth  : np.ndarray = np.array([ [0.0], [0.0], [0.0], [0.0] ])
        self.ddth : np.ndarray = np.array([ [0.0], [0.0], [0.0], [0.0] ])

        self.x1   : np.ndarray = np.array([ [-self.l1*np.sin(self.th[0][0])], [self.l1*np.cos(self.th[0][0])] ])
        self.dx1  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddx1 : np.ndarray = np.array([ [0.0], [0.0] ])

        self.x2   : np.ndarray = np.array([ [self.x1[0][0] - self.l2*np.sin(self.th[1][0])], 
                                            [self.x1[1][0] + self.l2*np.cos(self.th[1][0])] ])
        self.dx2  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddx2 : np.ndarray = np.array([ [0.0], [0.0] ])


        self.x3   : np.ndarray = np.array([ [self.x2[0][0] - self.l3*np.sin(self.th[2][0])],
                                            [self.x2[1][0] + self.l3*np.cos(self.th[2][0])] ])
        self.dx3  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddx3 : np.ndarray = np.array([ [0.0], [0.0] ])

        self.x4   : np.ndarray = np.array([ [self.x3[0][0] - self.l4*np.sin(self.th[3][0])],
                                            [self.x3[1][0] + self.l4*np.cos(self.th[3][0])] ])
        self.dx4  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddx4 : np.ndarray = np.array([ [0.0], [0.0] ])

        self.lda1 : float = 0.0
        self.lda2 : float = 0.0

        self.is_control : bool = False
        self.tau  : np.ndarray = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        self.th_d : np.ndarray = np.zeros((4,1))

        self.k1 : float = 1.0
        self.k2 : float = 1.0

    def get_jacobian(self) -> np.ndarray:
        J = np.array(
            [
                [-self.l1*np.cos(self.th[0][0]), -self.l2*np.cos(self.th[1][0]), -self.l3*np.cos(self.th[2][0]), -self.l4*np.cos(self.th[3][0])],
                [-self.l1*np.sin(self.th[0][0]), -self.l2*np.sin(self.th[1][0]), -self.l3*np.sin(self.th[2][0]), -self.l4*np.sin(self.th[3][0])]
            ]
        )
        return J
    
    def get_foot_velcoity(self):
        J = self.get_jacobian()
        foot_vel = J @ self.dth
        return foot_vel
    
    def get_dth_gnd_contact(self):
        l1, l2, l3, l4         = self.l1, self.l2, self.l3, self.l4
        m1, m2, m3, m4         = self.m1, self.m2, self.m3, self.m4
        th1, th2, th3, th4     = self.th[0][0], self.th[1][0], self.th[2][0], self.th[3][0]

        J = self.get_jacobian()
        M = np.array([[1.0*l1**2*(m1 + m2 + m3 + m4), 1.0*l1*l2*(m2 + m3 + m4)*np.cos(th1 - th2), 1.0*l1*l3*(m3 + m4)*np.cos(th1 - th3), 1.0*l1*l4*m4*np.cos(th1 - th4)], [1.0*l1*l2*(m2 + m3 + m4)*np.cos(th1 - th2), 1.0*l2**2*(m2 + m3 + m4), 1.0*l2*l3*(m3 + m4)*np.cos(th2 - th3), 1.0*l2*l4*m4*np.cos(th2 - th4)], [1.0*l1*l3*(m3 + m4)*np.cos(th1 - th3), 1.0*l2*l3*(m3 + m4)*np.cos(th2 - th3), 1.0*l3**2*(m3 + m4), 1.0*l3*l4*m4*np.cos(th3 - th4)], [1.0*l1*l4*m4*np.cos(th1 - th4), 1.0*l2*l4*m4*np.cos(th2 - th4), 1.0*l3*l4*m4*np.cos(th3 - th4), 1.0*l4**2*m4]])
        M_inv = np.linalg.pinv(M)

        p = -np.linalg.pinv( (J @ M_inv @ J.T) ) @ J @ self.dth
        dth = self.dth + M_inv @ J.T @ p

        return dth
    
    def feedforward_control(self, rhs:np.ndarray):
        tau = -rhs
        return tau

    def system(self, dt:float, lda1:float=0.0, lda2:float=0.0):
        l1, l2, l3, l4         = self.l1, self.l2, self.l3, self.l4
        m1, m2, m3, m4         = self.m1, self.m2, self.m3, self.m4
        th1, th2, th3, th4     = self.th[0][0], self.th[1][0], self.th[2][0], self.th[3][0]
        dth1, dth2, dth3, dth4 = self.dth[0][0], self.dth[1][0], self.dth[2][0], self.dth[3][0]

        foot_vel = self.get_foot_velcoity()
        
        if self.x4[1][0] <= 0 and foot_vel[1][0] < 0:
            self.dth = self.get_dth_gnd_contact()
            
            for i in range(4):
                if abs(self.dth[i][0]) < 1e-5:
                    self.dth[i][0] = 0.0

        if self.is_control:
            tau_ff = np.zeros((6,1))
            # tau_ff[:4,:] = -self.k1 * (self.th - self.th_d) - self.k2*self.dth
        
        if self.x4[1][0] <= 0:
            
            A = np.array([[1.0*l1**2*(m1 + m2 + m3 + m4), 1.0*l1*l2*(m2 + m3 + m4)*np.cos(th1 - th2), 1.0*l1*l3*(m3 + m4)*np.cos(th1 - th3), 1.0*l1*l4*m4*np.cos(th1 - th4), 1.0*l1*np.cos(th1), 1.0*l1*np.sin(th1)], 
                          [1.0*l1*l2*(m2 + m3 + m4)*np.cos(th1 - th2), 1.0*l2**2*(m2 + m3 + m4), 1.0*l2*l3*(m3 + m4)*np.cos(th2 - th3), 1.0*l2*l4*m4*np.cos(th2 - th4), 1.0*l2*np.cos(th2), 1.0*l2*np.sin(th2)], 
                          [1.0*l1*l3*(m3 + m4)*np.cos(th1 - th3), 1.0*l2*l3*(m3 + m4)*np.cos(th2 - th3), 1.0*l3**2*(m3 + m4), 1.0*l3*l4*m4*np.cos(th3 - th4), 1.0*l3*np.cos(th3), 1.0*l3*np.sin(th3)], 
                          [1.0*l1*l4*m4*np.cos(th1 - th4), 1.0*l2*l4*m4*np.cos(th2 - th4), 1.0*l3*l4*m4*np.cos(th3 - th4), 1.0*l4**2*m4, 1.0*l4*np.cos(th4), 1.0*l4*np.sin(th4)], 
                          [-l1*np.cos(th1), -l2*np.cos(th2), -l3*np.cos(th3), -l4*np.cos(th4), 0, 0], 
                          [-l1*np.sin(th1), -l2*np.sin(th2), -l3*np.sin(th3), -l4*np.sin(th4), 0, 0]])

            B = np.array([[1.0*l1*(-dth2**2*l2*m2*np.sin(th1 - th2) - dth2**2*l2*m3*np.sin(th1 - th2) - dth2**2*l2*m4*np.sin(th1 - th2) - dth3**2*l3*m3*np.sin(th1 - th3) - dth3**2*l3*m4*np.sin(th1 - th3) - dth4**2*l4*m4*np.sin(th1 - th4) + g*m1*np.sin(th1) + g*m2*np.sin(th1) + g*m3*np.sin(th1) + g*m4*np.sin(th1))], 
                          [1.0*l2*(dth1**2*l1*m2*np.sin(th1 - th2) + dth1**2*l1*m3*np.sin(th1 - th2) + dth1**2*l1*m4*np.sin(th1 - th2) - dth3**2*l3*m3*np.sin(th2 - th3) - dth3**2*l3*m4*np.sin(th2 - th3) - dth4**2*l4*m4*np.sin(th2 - th4) + g*m2*np.sin(th2) + g*m3*np.sin(th2) + g*m4*np.sin(th2))], 
                          [1.0*l3*(dth1**2*l1*m3*np.sin(th1 - th3) + dth1**2*l1*m4*np.sin(th1 - th3) + dth2**2*l2*m3*np.sin(th2 - th3) + dth2**2*l2*m4*np.sin(th2 - th3) - dth4**2*l4*m4*np.sin(th3 - th4) + g*m3*np.sin(th3) + g*m4*np.sin(th3))], 
                          [1.0*l4*m4*(dth1**2*l1*np.sin(th1 - th4) + dth2**2*l2*np.sin(th2 - th4) + dth3**2*l3*np.sin(th3 - th4) + g*np.sin(th4))], 
                          [-dth1**2*l1*np.sin(th1) - dth2**2*l2*np.sin(th2) - dth3**2*l3*np.sin(th3) - dth4**2*l4*np.sin(th4)], 
                          [dth1**2*l1*np.cos(th1) + dth2**2*l2*np.cos(th2) + dth3**2*l3*np.cos(th3) + dth4**2*l4*np.cos(th4)]])
            
            ###################### numerical error compensation (Baumgarte stabilization) #################################
            # alpha = 1.0
            # beta = 1.0
            # C = np.array([
            #     self.x4[0][0],
            #     self.x4[1][0]
            # ])
            # v = self.get_foot_velcoity()
            # constraint_rhs = B[4:, :] - 2*alpha*v - beta**2*C

            # B[4] = constraint_rhs[0][0]
            # B[5] = constraint_rhs[1][0]
            ################################################################################################################

            if self.is_control:
                self.tau = self.feedforward_control(B)
                B += self.tau + A @ tau_ff

            sol = np.matmul(np.linalg.inv(A), B) 
            self.ddth = sol[:4,]
                    
            self.dth += self.ddth * dt

            # for i in range(4):
            #     if abs(self.dth[i][0]) < 1e-2:
            #         self.dth[i][0] = 0.0
            
            self.th  += self.dth * dt 

        else:
            A = np.array([[1.0*l1**2*(m1 + m2 + m3 + m4), 1.0*l1*l2*(m2 + m3 + m4)*np.cos(th1 - th2), 1.0*l1*l3*(m3 + m4)*np.cos(th1 - th3), 1.0*l1*l4*m4*np.cos(th1 - th4)], 
                          [1.0*l1*l2*(m2 + m3 + m4)*np.cos(th1 - th2), 1.0*l2**2*(m2 + m3 + m4), 1.0*l2*l3*(m3 + m4)*np.cos(th2 - th3), 1.0*l2*l4*m4*np.cos(th2 - th4)], 
                          [1.0*l1*l3*(m3 + m4)*np.cos(th1 - th3), 1.0*l2*l3*(m3 + m4)*np.cos(th2 - th3), 1.0*l3**2*(m3 + m4), 1.0*l3*l4*m4*np.cos(th3 - th4)], 
                          [1.0*l1*l4*m4*np.cos(th1 - th4), 1.0*l2*l4*m4*np.cos(th2 - th4), 1.0*l3*l4*m4*np.cos(th3 - th4), 1.0*l4**2*m4]])
            
            B = np.array([[1.0*l1*(-dth2**2*l2*m2*np.sin(th1 - th2) - dth2**2*l2*m3*np.sin(th1 - th2) - dth2**2*l2*m4*np.sin(th1 - th2) - dth3**2*l3*m3*np.sin(th1 - th3) - dth3**2*l3*m4*np.sin(th1 - th3) - dth4**2*l4*m4*np.sin(th1 - th4) + g*m1*np.sin(th1) + g*m2*np.sin(th1) + g*m3*np.sin(th1) + g*m4*np.sin(th1) - lda1*np.cos(th1) - lda2*np.sin(th1)) + self.tau[0][0]],
                          [1.0*l2*(dth1**2*l1*m2*np.sin(th1 - th2) + dth1**2*l1*m3*np.sin(th1 - th2) + dth1**2*l1*m4*np.sin(th1 - th2) - dth3**2*l3*m3*np.sin(th2 - th3) - dth3**2*l3*m4*np.sin(th2 - th3) - dth4**2*l4*m4*np.sin(th2 - th4) + g*m2*np.sin(th2) + g*m3*np.sin(th2) + g*m4*np.sin(th2) - lda1*np.cos(th2) - lda2*np.sin(th2)) + self.tau[1][0]], 
                          [1.0*l3*(dth1**2*l1*m3*np.sin(th1 - th3) + dth1**2*l1*m4*np.sin(th1 - th3) + dth2**2*l2*m3*np.sin(th2 - th3) + dth2**2*l2*m4*np.sin(th2 - th3) - dth4**2*l4*m4*np.sin(th3 - th4) + g*m3*np.sin(th3) + g*m4*np.sin(th3) - lda1*np.cos(th3) - lda2*np.sin(th3)) + self.tau[2][0]], 
                          [1.0*l4*(dth1**2*l1*m4*np.sin(th1 - th4) + dth2**2*l2*m4*np.sin(th2 - th4) + dth3**2*l3*m4*np.sin(th3 - th4) + g*m4*np.sin(th4) - lda1*np.cos(th4) - lda2*np.sin(th4)) + self.tau[3][0]]])

            if self.is_control:
                tau_ff = A @ tau_ff[:4, :] + self.feedforward_control(B)
                B += tau_ff

            self.ddth = np.matmul(np.linalg.inv(A), B) 
            self.dth += self.ddth * dt

            # for i in range(4):
            #     if abs(self.dth[i][0]) < 1e-2:
            #         self.dth[i][0] = 0.0

            self.th  += self.dth * dt 

        self.x1[0][0] = -l1 * np.sin(th1) 
        self.x1[1][0] = l1 * np.cos(th1) 

        self.x2[0][0] = -l1 * np.sin(th1) - l2 * np.sin(th2) 
        self.x2[1][0] = l1 * np.cos(th1) + l2 * np.cos(th2) 

        self.x3[0][0] = -l1 * np.sin(th1) - l2 * np.sin(th2) - l3 * np.sin(th3) 
        self.x3[1][0] = l1 * np.cos(th1) + l2 * np.cos(th2) + l3 * np.cos(th3)

        self.x4[0][0] = -l1 * np.sin(th1) - l2 * np.sin(th2) - l3 * np.sin(th3) - l4 * np.sin(th4)
        self.x4[1][0] = l1 * np.cos(th1) + l2 * np.cos(th2) + l3 * np.cos(th3) + l4 * np.cos(th4)

    def run(self, ax:Axes, dt:float):
        self.system(dt=dt)

        self.plot_robot(ax)
    
    def roll_out(self, t_eval:np.ndarray):
        t0 = t_eval[0]
        t_prev = t0

        pos1 = list()
        pos2 = list()
        pos3 = list()
        pos4 = list()

        for i in range(0, len(t_eval)):
            t = t_eval[i]
            dt = t - t_prev

            self.system(dt=dt)

            pos1.append( [self.x1[0][0], self.x1[1][0]] )
            pos2.append( [self.x2[0][0], self.x2[1][0]] )
            pos3.append( [self.x3[0][0], self.x3[1][0]] )
            pos4.append( [self.x4[0][0], self.x4[1][0]] )

            t_prev = t
        
        return pos1, pos2, pos3, pos4

    def set_desired_joint_poses(self, th_d:np.ndarray):
        self.th_d = th_d
    
    def set_control_gains(self, k1:float, k2:float):
        self.k1 = k1
        self.k2 = k2

    def plot_robot(self, ax:Axes, i:int, p1:list, p2:list, p3:list, p4:list):
        ax.plot(0.0, 0.0, 'o')

        ax.plot(p1[i][0], p1[i][1], 'o')
        ax.plot( (0.0, p1[i][0]), (0.0, p1[i][1]) )
        
        ax.plot(p2[i][0], p2[i][1], 'o')
        ax.plot( (p1[i][0], p2[i][0]), (p1[i][1], p2[i][1]) )

        ax.plot(p3[i][0], p3[i][1], 'o')
        ax.plot( (p2[i][0], p3[i][0]), (p2[i][1], p3[i][1]) )

        ax.plot(p4[i][0], p4[i][1], 'o')
        ax.plot( (p3[i][0], p4[i][0]), (p3[i][1], p4[i][1]) )

if __name__ == "__main__":
    field = Field(size = 5.0)
    field_axis = field.get_axis()

    pendulum = DoublePendulum()
    pendulum.is_control = True

    pendulum.set_desired_joint_poses( np.array([ [-np.pi/6], [-np.pi/6], [-5*np.pi/6 + 2*np.pi], [-5*np.pi/6 + 2*np.pi] ]) )
    pendulum.set_control_gains(k1=1.0, k2=3.0)

    pos1, pos2, pos3, pos4 = pendulum.roll_out(t_eval=field.t_eval)
    
    field.frame_callback = lambda i: pendulum.plot_robot(
        ax   = field_axis, 
        i    = i,
        p1 = pos1,
        p2 = pos2,
        p3 = pos3,
        p4 = pos4
    )

    field.start()
    field.spin()