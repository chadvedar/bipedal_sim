import numpy as np
import matplotlib.pyplot as plt
import time

from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from typing import List
from enum import auto, Enum

g = 9.81

class Event(Enum):
    N_CONTACT   = auto()
    LF_CONTACT  = auto()
    RF_CONTACT  = auto()
    LRF_CONTACT = auto()
    LRB_CONTACT = auto()
    LB_CONTACT  = auto()
    RB_CONTACT  = auto()

class DataPlot:
    def __init__(self, name:str):
        self.name = name
        self.x = list()
        self.y = list()

    def add_data(self, x:float, y:float):
        self.x.append(x)
        self.y.append(y)

class EventDataPlot:
    def __init__(self, name:str):
        self.name = name
        self.time_event : List[float] = list()
        self.event      : List[Event] = list()

    def add_data(self, t:float, event:Event):
        self.time_event.append(t)
        self.event.append(event)

class Field:
    def __init__(self, time_width:float=5.0, resol:int=5000, size:float=10.0):
        self.size   : float = size
        self.p_time : float = None

        self.t_span = (0, time_width)
        self.t_eval = np.linspace(*self.t_span, resol)

        self.init_field()
        self.init_static_plot(rows=4)
    
    def set_field_size(self):
        self.ax.set_aspect('equal')
        self.ax.set_xlim([-self.size, self.size])
        self.ax.set_ylim([-self.size, self.size])

    def set_scale_limit(self, y_bound:list=[-2.0, 3.0]):
        self.ax.set_ylim(y_bound)

    def update(self, frame):
        self.ax.clear()
        self.set_field_size()
        self.set_scale_limit()
        self.ax.axhline(y=0, color='r', linestyle='-')
        self.frame_callback(frame)

    def init_field(self):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        self.set_field_size()

    def init_static_plot(self, rows:int=1):
        self.fig_plot, self.ax_plot = plt.subplots(nrows=rows, ncols=1)

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

    def get_plot_axis(self) -> Axes:
        return self.ax_plot
    
    def spin(self):
        plt.show()

    def on_close(self, event):
        self.ani.event_source.stop()
        plt.close('all')

    def frame_callback(frame):
        pass

class MultiBody:
    def __init__(self, fsmax:float=1.0):
        self.m0 = 1.0
        self.m1 = 1.0
        self.m2 = 1.0
        self.m3 = 1.0
        self.m4 = 1.0

        self.l1 = 1.0
        self.l2 = 1.0
        self.l3 = 1.0
        self.l4 = 1.0

        self.x0   : np.ndarray = np.array([ [0.0], [2.0]] )

        self.q   : np.ndarray = np.array([ [self.x0[0][0]], [self.x0[1][0]], [-np.pi/10], [0.0], [0.0], [np.pi/4] ])
        self.dq  : np.ndarray = np.array([ [0.0], [0.0], [0.0], [0.0], [0.0], [0.0] ])
        self.ddq : np.ndarray = np.array([ [0.0], [0.0], [0.0], [0.0], [0.0], [0.0] ])

        self.cal_joint_pos()

        self.dx1  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddx1 : np.ndarray = np.array([ [0.0], [0.0] ])

        self.dx2  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddx2 : np.ndarray = np.array([ [0.0], [0.0] ])

        self.dx3  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddx3 : np.ndarray = np.array([ [0.0], [0.0] ])

        self.dx4  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddx4 : np.ndarray = np.array([ [0.0], [0.0] ])

        self.lda1 : np.ndarray = np.array([ [0.0], [0.0] ])
        self.lda2 : np.ndarray = np.array([ [0.0], [0.0] ])

        self.fsmax : float = fsmax

        self.tau  : np.ndarray = np.array([[0.0], [0.0], [0.0], [0.0]])

        self.is_left_foot_impact_activated = False
        self.is_right_foot_impact_activated = False
        self.is_left_right_foot_impact_activated = False
        self.is_left_right_foot_body_impact_activated = False
        self.is_left_foot_body_impact_activated = False
        self.is_right_foot_body_impact_activated = False

        self.sim_time = 0.0
        self.gnd_b_data_y = DataPlot(name="gnd_force_body_y")
        self.gnd_l_data_x = DataPlot(name="gnd_force_left_x")
        self.gnd_l_data_y = DataPlot(name="gnd_force_left_y")
        self.gnd_r_data_x = DataPlot(name="gnd_force_right_x")
        self.gnd_r_data_y = DataPlot(name="gnd_force_right_y")
        self.gnd_t_data_y = DataPlot(name="gnd_force_total_y")

        self.dq_data_1 = DataPlot(name="dq1")
        self.dq_data_2 = DataPlot(name="dq2")
        self.dq_data_3 = DataPlot(name="dq3")
        self.dq_data_4 = DataPlot(name="dq4")

        self.l_foot_vel_data_y = DataPlot(name="l_foot_vel_y")
        self.r_foot_vel_data_y = DataPlot(name="r_foot_vel_y")
        self.body_vel_data_y   = DataPlot(name="body_vel_y")

        self.event_timestamp = EventDataPlot(name="sim_event")

    def cal_joint_pos(self):
        th1, th2, th3, th4     = self.q[2][0], self.q[3][0], self.q[4][0], self.q[5][0]
        
        self.x1   : np.ndarray = np.array([ [self.x0[0][0] - self.l1*np.sin(th1)], 
                                            [self.x0[1][0] - self.l1*np.cos(th1)] ])
        
        self.x2   : np.ndarray = np.array([ [self.x1[0][0] - self.l2*np.sin(th2)], 
                                            [self.x1[1][0] - self.l2*np.cos(th2)] ])
        
        self.x3   : np.ndarray = np.array([ [self.x0[0][0] - self.l3*np.sin(th3)],
                                            [self.x0[1][0] - self.l3*np.cos(th3)] ])
        
        self.x4   : np.ndarray = np.array([ [self.x3[0][0] - self.l4*np.sin(th4)],
                                            [self.x3[1][0] - self.l4*np.cos(th4)] ])

    def get_jacobian(self) -> np.ndarray:
        th1 = self.q[2][0]
        th2 = self.q[3][0]
        th3 = self.q[4][0]
        th4 = self.q[5][0]
        l1  = self.l1
        l2  = self.l2
        l3  = self.l3
        l4  = self.l4

        J0 = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]
            ]
        )

        J1 = np.array(
            [
                [1, 0, -l1*np.cos(th1),  -l2*np.cos(th2), 0, 0],
                [0, 1,  l1*np.sin(th1),   l2*np.sin(th2), 0, 0],
            ]
        )

        J2 = np.array(
            [
                [1, 0, 0, 0, -l3*np.cos(th3),  -l4*np.cos(th4)],
                [0, 1, 0, 0,  l3*np.sin(th3),   l4*np.sin(th4)]
            ]
        )
        return J0, J1, J2

    def get_d_jacobian(self) -> np.ndarray:
        l1, l2, l3, l4         = self.l1, self.l2, self.l3, self.l4
        th1, th2, th3, th4     = self.q[2][0], self.q[3][0], self.q[4][0], self.q[5][0]
        dth1, dth2, dth3, dth4 = self.dq[2][0], self.dq[3][0], self.dq[4][0], self.dq[5][0]

        J0dot = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        J1dot = np.array([[0.0, 0.0, l1*np.sin(th1)*dth1,l2*np.sin(th2)*dth2, 0.0, 0.0],
                    [0.0, 0.0, l1*np.cos(th1)*dth1,l2*np.cos(th2)*dth2, 0.0, 0.0]])

        J2dot = np.array([[0.0, 0.0, 0.0, 0.0, l3*np.sin(th3)*dth3, l4*np.sin(th4)*dth4],
                    [0.0, 0.0, 0.0, 0.0, l3*np.cos(th3)*dth3, l4*np.cos(th4)*dth4]])

        return J0dot, J1dot, J2dot

    def remove_tiny_value(self, dq: np.ndarray, tol: float = 1e-2):
        _dq = dq.copy()
        _dq[np.abs(_dq) < tol] = 0.0
        return _dq

    def get_feet_velcoity(self):
        J0, J1, J2 = self.get_jacobian()
        x0_vel    = J0 @ self.dq
        foot_vel1 = J1 @ self.dq
        foot_vel2 = J2 @ self.dq
        return x0_vel, foot_vel1, foot_vel2

    def get_gnd_constraint(self, J):
        l1, l2, l3, l4         = self.l1, self.l2, self.l3, self.l4
        m0, m1, m2, m3, m4     = self.m0, self.m1, self.m2, self.m3, self.m4
        th1, th2, th3, th4     = self.q[2][0], self.q[3][0], self.q[4][0], self.q[5][0]

        M = np.array([[1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 0, 1.0*l1*(-m1 - m2)*np.cos(th1), -1.0*l2*m2*np.cos(th2), 1.0*l3*(-m3 - m4)*np.cos(th3), -1.0*l4*m4*np.cos(th4)], [0, 1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l2*m2*np.sin(th2), 1.0*l3*(m3 + m4)*np.sin(th3), 1.0*l4*m4*np.sin(th4)], [1.0*l1*(-m1 - m2)*np.cos(th1), 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l1**2*(m1 + m2), 1.0*l1*l2*m2*np.cos(th1 - th2), 0, 0], [-1.0*l2*m2*np.cos(th2), 1.0*l2*m2*np.sin(th2), 1.0*l1*l2*m2*np.cos(th1 - th2), 1.0*l2**2*m2, 0, 0], [1.0*l3*(-m3 - m4)*np.cos(th3), 1.0*l3*(m3 + m4)*np.sin(th3), 0, 0, 1.0*l3**2*(m3 + m4), 1.0*l3*l4*m4*np.cos(th3 - th4)], [-1.0*l4*m4*np.cos(th4), 1.0*l4*m4*np.sin(th4), 0, 0, 1.0*l3*l4*m4*np.cos(th3 - th4), 1.0*l4**2*m4]])
        M_inv = np.linalg.pinv(M)
    
        p = -np.linalg.pinv( (J @ M_inv @ J.T) ) @ J @ self.dq
        dq = self.dq + M_inv @ J.T @ p
        return dq

    def get_dq_gnd_contact_both_feet_and_body(self):
        J0, J1, J2 = self.get_jacobian()
        J = np.vstack((J0[1:2, :], J1, J2)) 
        dq = self.get_gnd_constraint(J)
        return dq
    
    def get_dq_gnd_contact_both_feet(self):
        J0, J1, J2 = self.get_jacobian()
        J = np.vstack((J1, J2)) 
        dq = self.get_gnd_constraint(J)
        return dq

    def get_dq_gnd_contact_left_feet(self):
        J0, J1, J2 = self.get_jacobian()
        J = np.vstack((J1)) 
        dq = self.get_gnd_constraint(J)
        return dq
    
    def get_dq_gnd_contact_right_feet(self):
        J0, J1, J2 = self.get_jacobian()
        J = np.vstack((J2)) 
        dq = self.get_gnd_constraint(J)
        return dq

    def get_dq_gnd_contact_left_foot_and_body(self):
        J0, J1, J2 = self.get_jacobian()
        J = np.vstack((J0[1:2, :], J1))
        dq = self.get_gnd_constraint(J)
        return dq

    def get_dq_gnd_contact_right_foot_and_body(self):
        J0, J1, J2 = self.get_jacobian()
        J = np.vstack((J0[1:2, :], J2))
        dq = self.get_gnd_constraint(J)
        return dq

    def is_left_foot_contact(self, foot_vel:float, tol=0.01):
            if self.x2[1][0] <= tol and foot_vel[1][0] < 0.0:
                return True
            return False
    
    def is_right_foot_contact(self, foot_vel:float, tol=0.01):
        if self.x4[1][0] <= tol and foot_vel[1][0] < 0.0:
            return True
        return False

    def is_body_contact(self, body_vel:float, tol=0.01):
        if self.x0[1][0] <= tol and body_vel[1][0] < 0.0:
            return True
        return False
    
    def is_left_foot_stay_contact(self, foot_vel:float, tol=0.01):
        if self.x2[1][0] <= tol :
            return True
        return False

    def is_right_foot_stay_contact(self, foot_vel:float, tol=0.01):
        if self.x4[1][0] <= tol :
            return True
        return False

    def is_body_stay_contact(self, body_vel:float, tol=0.01):
        if self.x0[1][0] <= tol :
            return True
        return False
    
    def free_falling_dynamic(self, tau1:float=0.0, tau2:float=0.0, tau3:float=0.0, tau4:float=0.0):
        l1, l2, l3, l4         = self.l1, self.l2, self.l3, self.l4
        m0, m1, m2, m3, m4     = self.m0, self.m1, self.m2, self.m3, self.m4
        th1, th2, th3, th4     = self.q[2][0], self.q[3][0], self.q[4][0], self.q[5][0]
        dth1, dth2, dth3, dth4 = self.dq[2][0], self.dq[3][0], self.dq[4][0], self.dq[5][0]
        
        A = np.array([[1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 0, 1.0*l1*(-m1 - m2)*np.cos(th1), -1.0*l2*m2*np.cos(th2), 1.0*l3*(-m3 - m4)*np.cos(th3), -1.0*l4*m4*np.cos(th4)], [0, 1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l2*m2*np.sin(th2), 1.0*l3*(m3 + m4)*np.sin(th3), 1.0*l4*m4*np.sin(th4)], [1.0*l1*(-m1 - m2)*np.cos(th1), 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l1**2*(m1 + m2), 1.0*l1*l2*m2*np.cos(th1 - th2), 0, 0], [-1.0*l2*m2*np.cos(th2), 1.0*l2*m2*np.sin(th2), 1.0*l1*l2*m2*np.cos(th1 - th2), 1.0*l2**2*m2, 0, 0], [1.0*l3*(-m3 - m4)*np.cos(th3), 1.0*l3*(m3 + m4)*np.sin(th3), 0, 0, 1.0*l3**2*(m3 + m4), 1.0*l3*l4*m4*np.cos(th3 - th4)], [-1.0*l4*m4*np.cos(th4), 1.0*l4*m4*np.sin(th4), 0, 0, 1.0*l3*l4*m4*np.cos(th3 - th4), 1.0*l4**2*m4]])
        B = np.array([[-1.0*dth1**2*l1*m1*np.sin(th1) - 1.0*dth3**2*l3*m3*np.sin(th3) - 1.0*m2*(dth1**2*l1*np.sin(th1) + dth2**2*l2*np.sin(th2)) - 1.0*m4*(dth3**2*l3*np.sin(th3) + dth4**2*l4*np.sin(th4))], [-1.0*dth1**2*l1*m1*np.cos(th1) - 1.0*dth1**2*l1*m2*np.cos(th1) - 1.0*dth2**2*l2*m2*np.cos(th2) - 1.0*dth3**2*l3*m3*np.cos(th3) - 1.0*dth3**2*l3*m4*np.cos(th3) - 1.0*dth4**2*l4*m4*np.cos(th4) - 1.0*g*m0 - 1.0*g*m1 - 1.0*g*m2 - 1.0*g*m3 - 1.0*g*m4], [-1.0*dth2**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l1*m1*np.sin(th1) - 1.0*g*l1*m2*np.sin(th1) + 1.0*tau1], [1.0*dth1**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l2*m2*np.sin(th2) + 1.0*tau2], [-1.0*dth4**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l3*m3*np.sin(th3) - 1.0*g*l3*m4*np.sin(th3) + 1.0*tau3], [1.0*dth3**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l4*m4*np.sin(th4) + 1.0*tau4]])

        ddq = np.matmul(np.linalg.pinv(A), B) 
        return ddq, np.zeros((5, 1))

    def left_foot_contact_dynamic(self, tau1:float=0.0, tau2:float=0.0, tau3:float=0.0, tau4:float=0.0):
        l1, l2, l3, l4         = self.l1, self.l2, self.l3, self.l4
        m0, m1, m2, m3, m4     = self.m0, self.m1, self.m2, self.m3, self.m4
        th1, th2, th3, th4     = self.q[2][0], self.q[3][0], self.q[4][0], self.q[5][0]
        dth1, dth2, dth3, dth4 = self.dq[2][0], self.dq[3][0], self.dq[4][0], self.dq[5][0]

        J0, J1, J2 = self.get_jacobian()
        J1dot = np.array([[0.0, 0.0, l1*np.sin(th1)*dth1,l2*np.sin(th2)*dth2, 0.0, 0.0],
                            [0.0, 0.0, l1*np.cos(th1)*dth1,l2*np.cos(th2)*dth2, 0.0, 0.0]])

        A = np.array([[1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 0, 1.0*l1*(-m1 - m2)*np.cos(th1), -1.0*l2*m2*np.cos(th2), 1.0*l3*(-m3 - m4)*np.cos(th3), -1.0*l4*m4*np.cos(th4)], [0, 1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l2*m2*np.sin(th2), 1.0*l3*(m3 + m4)*np.sin(th3), 1.0*l4*m4*np.sin(th4)], [1.0*l1*(-m1 - m2)*np.cos(th1), 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l1**2*(m1 + m2), 1.0*l1*l2*m2*np.cos(th1 - th2), 0, 0], [-1.0*l2*m2*np.cos(th2), 1.0*l2*m2*np.sin(th2), 1.0*l1*l2*m2*np.cos(th1 - th2), 1.0*l2**2*m2, 0, 0], [1.0*l3*(-m3 - m4)*np.cos(th3), 1.0*l3*(m3 + m4)*np.sin(th3), 0, 0, 1.0*l3**2*(m3 + m4), 1.0*l3*l4*m4*np.cos(th3 - th4)], [-1.0*l4*m4*np.cos(th4), 1.0*l4*m4*np.sin(th4), 0, 0, 1.0*l3*l4*m4*np.cos(th3 - th4), 1.0*l4**2*m4]])
        B = np.array([[-1.0*dth1**2*l1*m1*np.sin(th1) - 1.0*dth3**2*l3*m3*np.sin(th3) - 1.0*m2*(dth1**2*l1*np.sin(th1) + dth2**2*l2*np.sin(th2)) - 1.0*m4*(dth3**2*l3*np.sin(th3) + dth4**2*l4*np.sin(th4))], [-1.0*dth1**2*l1*m1*np.cos(th1) - 1.0*dth1**2*l1*m2*np.cos(th1) - 1.0*dth2**2*l2*m2*np.cos(th2) - 1.0*dth3**2*l3*m3*np.cos(th3) - 1.0*dth3**2*l3*m4*np.cos(th3) - 1.0*dth4**2*l4*m4*np.cos(th4) - 1.0*g*m0 - 1.0*g*m1 - 1.0*g*m2 - 1.0*g*m3 - 1.0*g*m4], [-1.0*dth2**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l1*m1*np.sin(th1) - 1.0*g*l1*m2*np.sin(th1) + 1.0*tau1], [1.0*dth1**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l2*m2*np.sin(th2) + 1.0*tau2], [-1.0*dth4**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l3*m3*np.sin(th3) - 1.0*g*l3*m4*np.sin(th3) + 1.0*tau3], [1.0*dth3**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l4*m4*np.sin(th4) + 1.0*tau4]])

        T = np.block([
            [A,              -J1.T],
            [J1, np.zeros((2, 2))]
        ])

        C = np.vstack([
            B,
            -J1dot @ self.dq
        ])

        sol = np.linalg.pinv(T) @ C
        
        ddq = sol[:6]
        lam = sol[6:]

        return ddq, np.vstack([ np.zeros((1,1)), lam, np.zeros((2,1)) ])
    
    def right_foot_contact_dynamic(self, tau1:float=0.0, tau2:float=0.0, tau3:float=0.0, tau4:float=0.0):
        l1, l2, l3, l4         = self.l1, self.l2, self.l3, self.l4
        m0, m1, m2, m3, m4     = self.m0, self.m1, self.m2, self.m3, self.m4
        th1, th2, th3, th4     = self.q[2][0], self.q[3][0], self.q[4][0], self.q[5][0]
        dth1, dth2, dth3, dth4 = self.dq[2][0], self.dq[3][0], self.dq[4][0], self.dq[5][0]

        J0, J1, J2 = self.get_jacobian()
        J2dot = np.array([[0.0, 0.0, 0.0, 0.0, l3*np.sin(th3)*dth3, l4*np.sin(th4)*dth4],
                          [0.0, 0.0, 0.0, 0.0, l3*np.cos(th3)*dth3, l4*np.cos(th4)*dth4]])

        A = np.array([[1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 0, 1.0*l1*(-m1 - m2)*np.cos(th1), -1.0*l2*m2*np.cos(th2), 1.0*l3*(-m3 - m4)*np.cos(th3), -1.0*l4*m4*np.cos(th4)], [0, 1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l2*m2*np.sin(th2), 1.0*l3*(m3 + m4)*np.sin(th3), 1.0*l4*m4*np.sin(th4)], [1.0*l1*(-m1 - m2)*np.cos(th1), 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l1**2*(m1 + m2), 1.0*l1*l2*m2*np.cos(th1 - th2), 0, 0], [-1.0*l2*m2*np.cos(th2), 1.0*l2*m2*np.sin(th2), 1.0*l1*l2*m2*np.cos(th1 - th2), 1.0*l2**2*m2, 0, 0], [1.0*l3*(-m3 - m4)*np.cos(th3), 1.0*l3*(m3 + m4)*np.sin(th3), 0, 0, 1.0*l3**2*(m3 + m4), 1.0*l3*l4*m4*np.cos(th3 - th4)], [-1.0*l4*m4*np.cos(th4), 1.0*l4*m4*np.sin(th4), 0, 0, 1.0*l3*l4*m4*np.cos(th3 - th4), 1.0*l4**2*m4]])
        B = np.array([[-1.0*dth1**2*l1*m1*np.sin(th1) - 1.0*dth3**2*l3*m3*np.sin(th3) - 1.0*m2*(dth1**2*l1*np.sin(th1) + dth2**2*l2*np.sin(th2)) - 1.0*m4*(dth3**2*l3*np.sin(th3) + dth4**2*l4*np.sin(th4))], [-1.0*dth1**2*l1*m1*np.cos(th1) - 1.0*dth1**2*l1*m2*np.cos(th1) - 1.0*dth2**2*l2*m2*np.cos(th2) - 1.0*dth3**2*l3*m3*np.cos(th3) - 1.0*dth3**2*l3*m4*np.cos(th3) - 1.0*dth4**2*l4*m4*np.cos(th4) - 1.0*g*m0 - 1.0*g*m1 - 1.0*g*m2 - 1.0*g*m3 - 1.0*g*m4], [-1.0*dth2**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l1*m1*np.sin(th1) - 1.0*g*l1*m2*np.sin(th1) + 1.0*tau1], [1.0*dth1**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l2*m2*np.sin(th2) + 1.0*tau2], [-1.0*dth4**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l3*m3*np.sin(th3) - 1.0*g*l3*m4*np.sin(th3) + 1.0*tau3], [1.0*dth3**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l4*m4*np.sin(th4) + 1.0*tau4]])

        T = np.block([
            [A,              -J2.T],
            [J2, np.zeros((2, 2))]
        ])

        C = np.vstack([
            B,
            -J2dot @ self.dq
        ])

        sol = np.linalg.pinv(T) @ C
        
        ddq = sol[:6]
        lam = sol[6:]

        return ddq, np.vstack([ np.zeros((1,1)), np.zeros((2,1)), lam ])

    def right_left_foot_contact_dynamic(self, tau1:float=0.0, tau2:float=0.0, tau3:float=0.0, tau4:float=0.0):
        l1, l2, l3, l4         = self.l1, self.l2, self.l3, self.l4
        m0, m1, m2, m3, m4     = self.m0, self.m1, self.m2, self.m3, self.m4
        th1, th2, th3, th4     = self.q[2][0], self.q[3][0], self.q[4][0], self.q[5][0]
        dth1, dth2, dth3, dth4 = self.dq[2][0], self.dq[3][0], self.dq[4][0], self.dq[5][0]

        J0, J1, J2 = self.get_jacobian()
        Jdot = np.array([
                    [0.0, 0.0, l1*np.sin(th1)*dth1,l2*np.sin(th2)*dth2, 0.0, 0.0],
                    [0.0, 0.0, l1*np.cos(th1)*dth1,l2*np.cos(th2)*dth2, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0,l3*np.sin(th3)*dth3, l4*np.sin(th4)*dth4],
                    [0.0, 0.0, 0.0, 0.0,l3*np.cos(th3)*dth3, l4*np.cos(th4)*dth4]
        ])

        A = np.array([[1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 0, 1.0*l1*(-m1 - m2)*np.cos(th1), -1.0*l2*m2*np.cos(th2), 1.0*l3*(-m3 - m4)*np.cos(th3), -1.0*l4*m4*np.cos(th4)], [0, 1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l2*m2*np.sin(th2), 1.0*l3*(m3 + m4)*np.sin(th3), 1.0*l4*m4*np.sin(th4)], [1.0*l1*(-m1 - m2)*np.cos(th1), 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l1**2*(m1 + m2), 1.0*l1*l2*m2*np.cos(th1 - th2), 0, 0], [-1.0*l2*m2*np.cos(th2), 1.0*l2*m2*np.sin(th2), 1.0*l1*l2*m2*np.cos(th1 - th2), 1.0*l2**2*m2, 0, 0], [1.0*l3*(-m3 - m4)*np.cos(th3), 1.0*l3*(m3 + m4)*np.sin(th3), 0, 0, 1.0*l3**2*(m3 + m4), 1.0*l3*l4*m4*np.cos(th3 - th4)], [-1.0*l4*m4*np.cos(th4), 1.0*l4*m4*np.sin(th4), 0, 0, 1.0*l3*l4*m4*np.cos(th3 - th4), 1.0*l4**2*m4]])
        B = np.array([[-1.0*dth1**2*l1*m1*np.sin(th1) - 1.0*dth3**2*l3*m3*np.sin(th3) - 1.0*m2*(dth1**2*l1*np.sin(th1) + dth2**2*l2*np.sin(th2)) - 1.0*m4*(dth3**2*l3*np.sin(th3) + dth4**2*l4*np.sin(th4))], [-1.0*dth1**2*l1*m1*np.cos(th1) - 1.0*dth1**2*l1*m2*np.cos(th1) - 1.0*dth2**2*l2*m2*np.cos(th2) - 1.0*dth3**2*l3*m3*np.cos(th3) - 1.0*dth3**2*l3*m4*np.cos(th3) - 1.0*dth4**2*l4*m4*np.cos(th4) - 1.0*g*m0 - 1.0*g*m1 - 1.0*g*m2 - 1.0*g*m3 - 1.0*g*m4], [-1.0*dth2**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l1*m1*np.sin(th1) - 1.0*g*l1*m2*np.sin(th1) + 1.0*tau1], [1.0*dth1**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l2*m2*np.sin(th2) + 1.0*tau2], [-1.0*dth4**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l3*m3*np.sin(th3) - 1.0*g*l3*m4*np.sin(th3) + 1.0*tau3], [1.0*dth3**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l4*m4*np.sin(th4) + 1.0*tau4]])

        J = np.vstack((J1, J2))

        T = np.block([
            [A,              -J.T],
            [J, np.zeros((4, 4))]
        ])

        C = np.vstack([
            B,
            -Jdot @ self.dq
        ])
        
        sol = np.linalg.pinv(T) @ C
        
        ddq = sol[:6]
        lam = sol[6:]
        
        return ddq, np.vstack((np.zeros((1,1)), lam))
    
    def right_left_foot_body_contact_dynamic(self, tau1:float=0.0, tau2:float=0.0, tau3:float=0.0, tau4:float=0.0):
        l1, l2, l3, l4         = self.l1, self.l2, self.l3, self.l4
        m0, m1, m2, m3, m4     = self.m0, self.m1, self.m2, self.m3, self.m4
        th1, th2, th3, th4     = self.q[2][0], self.q[3][0], self.q[4][0], self.q[5][0]
        dth1, dth2, dth3, dth4 = self.dq[2][0], self.dq[3][0], self.dq[4][0], self.dq[5][0]

        J0, J1, J2 = self.get_jacobian()
        J_body = J0[1:2, :]

        J = np.vstack((J_body, J1, J2))

        Jdot = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, l1*np.sin(th1)*dth1,
            l2*np.sin(th2)*dth2, 0.0, 0.0],
            [0.0, 0.0, l1*np.cos(th1)*dth1,
            l2*np.cos(th2)*dth2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0,
            l3*np.sin(th3)*dth3, l4*np.sin(th4)*dth4],
            [0.0, 0.0, 0.0, 0.0,
            l3*np.cos(th3)*dth3, l4*np.cos(th4)*dth4]
        ])

        A = np.array([[1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 0, 1.0*l1*(-m1 - m2)*np.cos(th1), -1.0*l2*m2*np.cos(th2), 1.0*l3*(-m3 - m4)*np.cos(th3), -1.0*l4*m4*np.cos(th4)], [0, 1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l2*m2*np.sin(th2), 1.0*l3*(m3 + m4)*np.sin(th3), 1.0*l4*m4*np.sin(th4)], [1.0*l1*(-m1 - m2)*np.cos(th1), 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l1**2*(m1 + m2), 1.0*l1*l2*m2*np.cos(th1 - th2), 0, 0], [-1.0*l2*m2*np.cos(th2), 1.0*l2*m2*np.sin(th2), 1.0*l1*l2*m2*np.cos(th1 - th2), 1.0*l2**2*m2, 0, 0], [1.0*l3*(-m3 - m4)*np.cos(th3), 1.0*l3*(m3 + m4)*np.sin(th3), 0, 0, 1.0*l3**2*(m3 + m4), 1.0*l3*l4*m4*np.cos(th3 - th4)], [-1.0*l4*m4*np.cos(th4), 1.0*l4*m4*np.sin(th4), 0, 0, 1.0*l3*l4*m4*np.cos(th3 - th4), 1.0*l4**2*m4]])
        B = np.array([[-1.0*dth1**2*l1*m1*np.sin(th1) - 1.0*dth3**2*l3*m3*np.sin(th3) - 1.0*m2*(dth1**2*l1*np.sin(th1) + dth2**2*l2*np.sin(th2)) - 1.0*m4*(dth3**2*l3*np.sin(th3) + dth4**2*l4*np.sin(th4))], [-1.0*dth1**2*l1*m1*np.cos(th1) - 1.0*dth1**2*l1*m2*np.cos(th1) - 1.0*dth2**2*l2*m2*np.cos(th2) - 1.0*dth3**2*l3*m3*np.cos(th3) - 1.0*dth3**2*l3*m4*np.cos(th3) - 1.0*dth4**2*l4*m4*np.cos(th4) - 1.0*g*m0 - 1.0*g*m1 - 1.0*g*m2 - 1.0*g*m3 - 1.0*g*m4], [-1.0*dth2**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l1*m1*np.sin(th1) - 1.0*g*l1*m2*np.sin(th1) + 1.0*tau1], [1.0*dth1**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l2*m2*np.sin(th2) + 1.0*tau2], [-1.0*dth4**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l3*m3*np.sin(th3) - 1.0*g*l3*m4*np.sin(th3) + 1.0*tau3], [1.0*dth3**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l4*m4*np.sin(th4) + 1.0*tau4]])

        T = np.block([
            [A, -J.T],
            [J, np.zeros((J.shape[0], J.shape[0]))]
        ])

        C = np.vstack([
            B,
            -Jdot @ self.dq
        ])
        
        sol = np.linalg.solve(T, C)

        ddq = sol[:6]
        lam = sol[6:]

        return ddq, lam

    def left_foot_body_contact_dynamic(self, tau1:float=0.0, tau2:float=0.0, tau3:float=0.0, tau4:float=0.0):
        l1, l2, l3, l4         = self.l1, self.l2, self.l3, self.l4
        m0, m1, m2, m3, m4     = self.m0, self.m1, self.m2, self.m3, self.m4
        th1, th2, th3, th4     = self.q[2][0], self.q[3][0], self.q[4][0], self.q[5][0]
        dth1, dth2, dth3, dth4 = self.dq[2][0], self.dq[3][0], self.dq[4][0], self.dq[5][0]

        J0, J1, J2 = self.get_jacobian()
        J_body = J0[1:2, :]

        J = np.vstack((J_body, J1))

        Jdot = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, l1*np.sin(th1)*dth1, l2*np.sin(th2)*dth2, 0.0, 0.0],
            [0.0, 0.0, l1*np.cos(th1)*dth1, l2*np.cos(th2)*dth2, 0.0, 0.0],
        ])

        A = np.array([[1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 0, 1.0*l1*(-m1 - m2)*np.cos(th1), -1.0*l2*m2*np.cos(th2), 1.0*l3*(-m3 - m4)*np.cos(th3), -1.0*l4*m4*np.cos(th4)], [0, 1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l2*m2*np.sin(th2), 1.0*l3*(m3 + m4)*np.sin(th3), 1.0*l4*m4*np.sin(th4)], [1.0*l1*(-m1 - m2)*np.cos(th1), 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l1**2*(m1 + m2), 1.0*l1*l2*m2*np.cos(th1 - th2), 0, 0], [-1.0*l2*m2*np.cos(th2), 1.0*l2*m2*np.sin(th2), 1.0*l1*l2*m2*np.cos(th1 - th2), 1.0*l2**2*m2, 0, 0], [1.0*l3*(-m3 - m4)*np.cos(th3), 1.0*l3*(m3 + m4)*np.sin(th3), 0, 0, 1.0*l3**2*(m3 + m4), 1.0*l3*l4*m4*np.cos(th3 - th4)], [-1.0*l4*m4*np.cos(th4), 1.0*l4*m4*np.sin(th4), 0, 0, 1.0*l3*l4*m4*np.cos(th3 - th4), 1.0*l4**2*m4]])
        B = np.array([[-1.0*dth1**2*l1*m1*np.sin(th1) - 1.0*dth3**2*l3*m3*np.sin(th3) - 1.0*m2*(dth1**2*l1*np.sin(th1) + dth2**2*l2*np.sin(th2)) - 1.0*m4*(dth3**2*l3*np.sin(th3) + dth4**2*l4*np.sin(th4))], [-1.0*dth1**2*l1*m1*np.cos(th1) - 1.0*dth1**2*l1*m2*np.cos(th1) - 1.0*dth2**2*l2*m2*np.cos(th2) - 1.0*dth3**2*l3*m3*np.cos(th3) - 1.0*dth3**2*l3*m4*np.cos(th3) - 1.0*dth4**2*l4*m4*np.cos(th4) - 1.0*g*m0 - 1.0*g*m1 - 1.0*g*m2 - 1.0*g*m3 - 1.0*g*m4], [-1.0*dth2**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l1*m1*np.sin(th1) - 1.0*g*l1*m2*np.sin(th1) + 1.0*tau1], [1.0*dth1**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l2*m2*np.sin(th2) + 1.0*tau2], [-1.0*dth4**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l3*m3*np.sin(th3) - 1.0*g*l3*m4*np.sin(th3) + 1.0*tau3], [1.0*dth3**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l4*m4*np.sin(th4) + 1.0*tau4]])

        T = np.block([
            [A, -J.T],
            [J, np.zeros((J.shape[0], J.shape[0]))]
        ])

        C = np.vstack([
            B,
            -Jdot @ self.dq
        ])
        
        sol = np.linalg.solve(T, C)

        ddq = sol[:6]
        lam = sol[6:]

        return ddq, np.vstack((lam, np.zeros((2,1))))

    def right_foot_body_contact_dynamic(self, tau1:float=0.0, tau2:float=0.0, tau3:float=0.0, tau4:float=0.0):
        l1, l2, l3, l4         = self.l1, self.l2, self.l3, self.l4
        m0, m1, m2, m3, m4     = self.m0, self.m1, self.m2, self.m3, self.m4
        th1, th2, th3, th4     = self.q[2][0], self.q[3][0], self.q[4][0], self.q[5][0]
        dth1, dth2, dth3, dth4 = self.dq[2][0], self.dq[3][0], self.dq[4][0], self.dq[5][0]

        J0, J1, J2 = self.get_jacobian()
        J_body = J0[1:2, :]

        J = np.vstack((J_body, J2))

        Jdot = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, l3*np.sin(th3)*dth3, l4*np.sin(th4)*dth4],
            [0.0, 0.0, 0.0, 0.0, l3*np.cos(th3)*dth3, l4*np.cos(th4)*dth4]
        ])

        A = np.array([[1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 0, 1.0*l1*(-m1 - m2)*np.cos(th1), -1.0*l2*m2*np.cos(th2), 1.0*l3*(-m3 - m4)*np.cos(th3), -1.0*l4*m4*np.cos(th4)], [0, 1.0*m0 + 1.0*m1 + 1.0*m2 + 1.0*m3 + 1.0*m4, 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l2*m2*np.sin(th2), 1.0*l3*(m3 + m4)*np.sin(th3), 1.0*l4*m4*np.sin(th4)], [1.0*l1*(-m1 - m2)*np.cos(th1), 1.0*l1*(m1 + m2)*np.sin(th1), 1.0*l1**2*(m1 + m2), 1.0*l1*l2*m2*np.cos(th1 - th2), 0, 0], [-1.0*l2*m2*np.cos(th2), 1.0*l2*m2*np.sin(th2), 1.0*l1*l2*m2*np.cos(th1 - th2), 1.0*l2**2*m2, 0, 0], [1.0*l3*(-m3 - m4)*np.cos(th3), 1.0*l3*(m3 + m4)*np.sin(th3), 0, 0, 1.0*l3**2*(m3 + m4), 1.0*l3*l4*m4*np.cos(th3 - th4)], [-1.0*l4*m4*np.cos(th4), 1.0*l4*m4*np.sin(th4), 0, 0, 1.0*l3*l4*m4*np.cos(th3 - th4), 1.0*l4**2*m4]])
        B = np.array([[-1.0*dth1**2*l1*m1*np.sin(th1) - 1.0*dth3**2*l3*m3*np.sin(th3) - 1.0*m2*(dth1**2*l1*np.sin(th1) + dth2**2*l2*np.sin(th2)) - 1.0*m4*(dth3**2*l3*np.sin(th3) + dth4**2*l4*np.sin(th4))], [-1.0*dth1**2*l1*m1*np.cos(th1) - 1.0*dth1**2*l1*m2*np.cos(th1) - 1.0*dth2**2*l2*m2*np.cos(th2) - 1.0*dth3**2*l3*m3*np.cos(th3) - 1.0*dth3**2*l3*m4*np.cos(th3) - 1.0*dth4**2*l4*m4*np.cos(th4) - 1.0*g*m0 - 1.0*g*m1 - 1.0*g*m2 - 1.0*g*m3 - 1.0*g*m4], [-1.0*dth2**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l1*m1*np.sin(th1) - 1.0*g*l1*m2*np.sin(th1) + 1.0*tau1], [1.0*dth1**2*l1*l2*m2*np.sin(th1 - th2) - 1.0*g*l2*m2*np.sin(th2) + 1.0*tau2], [-1.0*dth4**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l3*m3*np.sin(th3) - 1.0*g*l3*m4*np.sin(th3) + 1.0*tau3], [1.0*dth3**2*l3*l4*m4*np.sin(th3 - th4) - 1.0*g*l4*m4*np.sin(th4) + 1.0*tau4]])

        T = np.block([
            [A, -J.T],
            [J, np.zeros((J.shape[0], J.shape[0]))]
        ])

        C = np.vstack([
            B,
            -Jdot @ self.dq
        ])
        
        sol = np.linalg.solve(T, C)

        ddq = sol[:6]
        lam = sol[6:]
        
        return ddq, np.vstack((lam[0], np.zeros((2,1)), lam[1:]))
    
    def dynamic(self, J:list, dJ:list, tau1:float=0.0, tau2:float=0.0, tau3:float=0.0, tau4:float=0.0):
        J_indices = [i for i, j in enumerate(J) if not j is None ]

        lam = np.zeros((5, 1))

        if J_indices == [0, 1, 2]:
            self.ddq, lam = self.right_left_foot_body_contact_dynamic(tau1, tau2, tau3, tau4)

        elif J_indices == [1, 2]:
            self.ddq, lam = self.right_left_foot_contact_dynamic(tau1, tau2, tau3, tau4)

        elif J_indices == [0, 1]:
            self.ddq, lam = self.left_foot_body_contact_dynamic(tau1, tau2, tau3, tau4)

        elif J_indices == [0, 2]:
            self.ddq, lam = self.right_foot_body_contact_dynamic(tau1, tau2, tau3, tau4)

        elif J_indices == [2]:
            self.ddq, lam = self.right_foot_contact_dynamic(tau1, tau2, tau3, tau4)

        elif J_indices == [1]:
            self.ddq, lam = self.left_foot_contact_dynamic(tau1, tau2, tau3, tau4)

        else:
            self.ddq, lam = self.free_falling_dynamic(tau1, tau2, tau3, tau4)

        return self.ddq , lam

    def system(self, dt:float, tau1:float=0.0, tau2:float=0.0, tau3:float=0.0, tau4:float=0.0):
        self.sim_time += dt
        event = Event.N_CONTACT
        
        x_vel, foot_vel1, foot_vel2 = self.get_feet_velcoity()

        body_vel = x_vel
        left_foot_vel = foot_vel1
        right_foot_vel = foot_vel2

        if self.is_left_foot_contact(left_foot_vel) and self.is_right_foot_contact(right_foot_vel) and self.is_body_contact(body_vel):
            if not self.is_left_right_foot_body_impact_activated:
                self.dq = self.get_dq_gnd_contact_both_feet_and_body()

                self.is_left_foot_impact_activated = False
                self.is_right_foot_impact_activated = False
                self.is_left_right_foot_impact_activated = False
                self.is_left_right_foot_body_impact_activated = True
                self.is_left_foot_body_impact_activated = False
                self.is_right_foot_body_impact_activated = False
                event = Event.LRB_CONTACT

        elif self.is_left_foot_contact(left_foot_vel) and self.is_body_contact(body_vel):
            if not self.is_left_foot_body_impact_activated:
                self.dq = self.get_dq_gnd_contact_left_foot_and_body()
                
                self.is_left_foot_impact_activated = False
                self.is_right_foot_impact_activated = False
                self.is_left_right_foot_impact_activated = False
                self.is_left_right_foot_body_impact_activated = False
                self.is_left_foot_body_impact_activated = True
                self.is_right_foot_body_impact_activated = False
                event = Event.LB_CONTACT

        elif self.is_right_foot_contact(right_foot_vel) and self.is_body_contact(body_vel):
            if not self.is_right_foot_body_impact_activated:
                self.dq = self.get_dq_gnd_contact_right_foot_and_body()
            
                self.is_left_foot_impact_activated = False
                self.is_right_foot_impact_activated = False
                self.is_left_right_foot_impact_activated = False
                self.is_left_right_foot_body_impact_activated = False
                self.is_left_foot_body_impact_activated = False
                self.is_right_foot_body_impact_activated = True
                event = Event.RB_CONTACT


        elif self.is_left_foot_contact(left_foot_vel) and self.is_right_foot_contact(right_foot_vel):
            if not self.is_left_right_foot_impact_activated:
                self.dq = self.get_dq_gnd_contact_both_feet()

                self.is_left_foot_impact_activated = False
                self.is_right_foot_impact_activated = False
                self.is_left_right_foot_impact_activated = True
                self.is_left_right_foot_body_impact_activated = False
                self.is_left_foot_body_impact_activated = False
                self.is_right_foot_body_impact_activated = False
                event = Event.LRF_CONTACT
                

        elif self.is_left_foot_contact(left_foot_vel):
            if not self.is_left_foot_impact_activated:
                self.dq = self.get_dq_gnd_contact_left_feet()

                self.is_left_foot_impact_activated = True
                self.is_right_foot_impact_activated = False
                self.is_left_right_foot_impact_activated = False
                self.is_left_right_foot_body_impact_activated = False
                self.is_left_foot_body_impact_activated = False
                self.is_right_foot_body_impact_activated = False
                event = Event.LF_CONTACT
                

        elif self.is_right_foot_contact(right_foot_vel):
            if not self.is_right_foot_impact_activated:
                self.dq = self.get_dq_gnd_contact_right_feet()

                self.is_left_foot_impact_activated = False
                self.is_right_foot_impact_activated = True
                self.is_left_right_foot_impact_activated = False
                self.is_left_right_foot_body_impact_activated = False
                self.is_left_foot_body_impact_activated = False
                self.is_right_foot_body_impact_activated = False
                event = Event.RF_CONTACT

        J_general    = self.get_jacobian()
        dJ_general = self.get_d_jacobian()

        J, dJ, lam = [None, None, None], [None, None, None], -1 * np.ones((5,1))
        
        if self.is_body_stay_contact(body_vel):
            J[0] = J_general[0]
            dJ[0] = dJ_general[0] 

        if self.is_left_foot_stay_contact(left_foot_vel):
            J[1] = J_general[1] 
            dJ[1] = dJ_general[1] 

        if self.is_right_foot_stay_contact(right_foot_vel):
            J[2] = J_general[2] 
            dJ[2] = dJ_general[2]

        isControlled = True
        if isControlled:
            tau1, tau2, tau3, tau4 = self.stance_control_law()
            
        while True:

            self.ddq, lam = self.dynamic(J, dJ, tau1, tau2, tau3, tau4)
            lam_y = np.array([lam[0][0], lam[2][0], lam[4][0]]) 
            violation_index = np.where( lam_y < 0.0 )[0]            

            if violation_index.shape[0] == 0:
                break
            else:
                J, dJ = [None for _ in range(3)], [None for _ in range(3)]
                pass_index = np.where( lam_y > 0.0 )[0]

                for i in pass_index:
                    J[i] = J_general[i]
                    dJ[i] =  dJ_general[i] 
                    

        self.dq += self.ddq * dt
        self.q  += self.dq * dt

        self.x0 = np.array([ [self.q[0][0]], [self.q[1][0]] ])
        self.cal_joint_pos()

        self.save_dynamic_sim(
            t      = self.sim_time,
            gnd_by = lam[0][0],
            gnd_lx = lam[1][0],
            gnd_ly = lam[2][0],
            gnd_rx = lam[3][0],
            gnd_ry = lam[4][0],
            dq     = self.dq,
            l_foot_vel = foot_vel1,
            r_foot_vel = foot_vel2,
            body_vel   = body_vel,
            event  = event
        )

        return 
    
    def run(self, ax:Axes, dt:float):
        self.system(dt=dt)

        self.plot_robot(ax)
    
    def roll_out(self, t_eval:np.ndarray):
        t0 = t_eval[0]
        t_prev = t0

        pos0 = list()
        pos1 = list()
        pos2 = list()
        pos3 = list()
        pos4 = list()

        for i in range(0, len(t_eval)):
            t = t_eval[i]
            dt = t - t_prev

            self.system(dt=dt)

            pos0.append( [self.x0[0][0], self.x0[1][0]] )
            pos1.append( [self.x1[0][0], self.x1[1][0]] )
            pos2.append( [self.x2[0][0], self.x2[1][0]] )
            pos3.append( [self.x3[0][0], self.x3[1][0]] )
            pos4.append( [self.x4[0][0], self.x4[1][0]] )

            t_prev = t
        
        return pos0, pos1, pos2, pos3, pos4

    def stance_control_law(self):
        kp = 25.0
        kd = 1.0
        th1d, th2d, th3d, th4d = -np.pi/10, 0.0, 0.0, np.pi/10
        th1, th2, th3, th4     = self.q[2][0], self.q[3][0], self.q[4][0], self.q[5][0]
        dth1, dth2, dth3, dth4 = self.dq[2][0], self.dq[3][0], self.dq[4][0], self.dq[5][0]
        
        tau = -kp * (np.array([[th1], [th2], [th3], [th4]]) - np.array([[th1d], [th2d], [th3d], [th4d]])) - kd * np.array([ [dth1],[dth2],[dth3],[dth4] ])
        tau1 = tau[0][0]
        tau2 = tau[1][0]
        tau3 = tau[2][0]
        tau4 = tau[3][0]

        return tau1, tau2, tau3, tau4
    
    def plot_robot(self, ax:Axes, i:int, p0:list, p1:list, p2:list, p3:list, p4:list, factor:int=16):
        num_replay = len(p0)
        _i = i*factor
        _i %= num_replay

        ax.plot(p0[_i][0], p0[_i][1], 'o')

        ax.plot(p1[_i][0], p1[_i][1], 'o')
        ax.plot( (p0[_i][0], p1[_i][0]), (p0[_i][1], p1[_i][1]) )
        
        ax.plot(p2[_i][0], p2[_i][1], 'o')
        ax.plot( (p1[_i][0], p2[_i][0]), (p1[_i][1], p2[_i][1]) )

        ax.plot(p3[_i][0], p3[_i][1], 'o')
        ax.plot( (p0[_i][0], p3[_i][0]), (p0[_i][1], p3[_i][1]) )

        ax.plot(p4[_i][0], p4[_i][1], 'o')
        ax.plot( (p3[_i][0], p4[_i][0]), (p3[_i][1], p4[_i][1]) )

    def save_dynamic_sim(
            self, 
            t          : float, 
            gnd_by     : float, 
            gnd_lx     : float, 
            gnd_ly     : float, 
            gnd_rx     : float, 
            gnd_ry     : float, 
            dq         : np.ndarray, 
            l_foot_vel : np.ndarray,
            r_foot_vel : np.ndarray, 
            body_vel   : np.ndarray,
            event      : Event
        ):

        self.gnd_b_data_y.add_data(x=t, y=gnd_by)
        self.gnd_l_data_x.add_data(x=t, y=gnd_lx)
        self.gnd_l_data_y.add_data(x=t, y=gnd_ly)
        self.gnd_r_data_x.add_data(x=t, y=gnd_rx)
        self.gnd_r_data_y.add_data(x=t, y=gnd_ry)
        self.gnd_t_data_y.add_data(x=t, y=gnd_by + gnd_ly + gnd_ry)

        self.dq_data_1.add_data(x=t, y=dq[2][0])
        self.dq_data_2.add_data(x=t, y=dq[3][0])
        self.dq_data_3.add_data(x=t, y=dq[4][0])
        self.dq_data_4.add_data(x=t, y=dq[5][0])

        self.l_foot_vel_data_y.add_data(x=t, y=l_foot_vel[1][0])
        self.r_foot_vel_data_y.add_data(x=t, y=r_foot_vel[1][0])
        self.body_vel_data_y.add_data(x=t, y=body_vel[1][0])

        if not event == Event.N_CONTACT:
            self.event_timestamp.add_data(t, event)

def show_sim_plot(ax:List[Axes], data:List[DataPlot]):
    for _, d in enumerate(data[:2]):
        ax[0].plot(d.x, d.y, label=d.name)
        ax[0].legend()

    for _, d in enumerate(data[2:5]):
        ax[1].plot(d.x, d.y, label=d.name)
        ax[1].legend()

    for _, d in enumerate(data[5:9]):
        ax[2].plot(d.x, d.y, label=d.name)
        ax[2].legend()

    for _, d in enumerate(data[9:]):
        ax[3].plot(d.x, d.y, label=d.name)
        ax[3].legend()
        ax[3].grid(True)

    print(f"sum ground reaction y-force = time:[{data[-1].x[-1]} s.], value:[{data[-1].y[-1]} N.]")

def show_sim_event_plot(ax:List[Axes], eventData:EventDataPlot):
    for t in eventData.time_event:
        for i in range(len(ax)):
            ax[i].axvline(t, color='r', linestyle='--')

    print(f"Event List : [{eventData.event}]")
    print(f"Event timestamp : [{eventData.time_event}]")

if __name__ == "__main__":
    field = Field(size = 5.0)
    field_axis = field.get_axis()
    plot_axis  = field.get_plot_axis()

    multibody = MultiBody()

    pos0, pos1, pos2, pos3, pos4 = multibody.roll_out(t_eval=field.t_eval)
    
    field.frame_callback = lambda i: multibody.plot_robot(
        ax   = field_axis, 
        i    = i,
        p0 = pos0,
        p1 = pos1,
        p2 = pos2,
        p3 = pos3,
        p4 = pos4
    )

    show_sim_plot(plot_axis, [
            multibody.gnd_l_data_x,
            multibody.gnd_r_data_x,
            multibody.gnd_l_data_y,
            multibody.gnd_r_data_y,
            multibody.gnd_t_data_y,
            multibody.dq_data_1,
            multibody.dq_data_2,
            multibody.dq_data_3,
            multibody.dq_data_4,
            multibody.l_foot_vel_data_y,
            multibody.r_foot_vel_data_y,
            multibody.body_vel_data_y
        ]
    )

    show_sim_event_plot(plot_axis, multibody.event_timestamp)

    field.start()
    field.spin()