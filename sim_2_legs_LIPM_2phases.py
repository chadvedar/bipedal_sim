import numpy as np
import matplotlib.pyplot as plt
import time
import threading

from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from typing import List
from enum import Enum, auto
from abc import ABC, abstractmethod

G = 9.81

class Field:
    def __init__(self, size:float=10.0):
        self.size   : float = size
        self.p_time : float = None
        self.dt     : float = 1.0/60.0
        self.init_field()
    
    def set_field_size(self):
        self.ax.set_aspect('equal')
        self.ax.set_xlim([-5.0, 5.0])
        self.ax.set_ylim([-0.1, 2.0])

    def update(self, frame):
        self.ax.clear()
        self.set_field_size()
        self.frame_callback(self.dt)
    
    def init_field(self):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        self.set_field_size()

    def start(self):
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_release)
        self.ani = FuncAnimation(self.fig, self.update, interval=0.0001, cache_frame_data=False)

    def get_axis(self) -> Axes:
        return self.ax
    
    def spin(self):
        plt.show()

    def on_press(self, event):
        pass

    def on_release(self, event):
        pass

    def on_close(self, event):
        self.ani.event_source.stop()
        plt.close('all')

    def frame_callback(dt):
        pass

class BasedDynamic(ABC):
    def __init__(self, m0:float, m1:float, m2:float, l1:float, l2:float):
        self.m0 = m0
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2

        self.x0 : np.ndarray = np.zeros((2,1))
        self.x1 : np.ndarray = np.zeros((2,1))
        self.x2 : np.ndarray = np.zeros((2,1))
        self.th : np.ndarray = np.zeros((2,1))

        self.tau : np.ndarray = np.zeros((2,1))
        
    @abstractmethod
    def system(self):
        pass

class LeftBaseBipedal(BasedDynamic):
    def __init__(self, m0:float, m1:float, m2:float, l1:float, l2:float):
        super().__init__(m0, m1, m2, l1, l2)

        self.th   : np.ndarray = np.array([ [0.0], [-np.pi/4] ])
        self.dth  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddth : np.ndarray = np.array([ [0.0], [0.0] ])

        self.x1   : np.ndarray = np.array([ [-self.l1*np.sin(self.th[0][0])], [self.l1*np.cos(self.th[0][0])] ])
        self.dx1  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddx1 : np.ndarray = np.array([ [0.0], [0.0] ])

        self.x2   : np.ndarray = np.array([ [self.x1[0][0] - self.l2*np.sin(self.th[1][0])], 
                                            [self.x1[1][0] + self.l2*np.cos(self.th[1][0])] ])
        self.dx2  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddx2 : np.ndarray = np.array([ [0.0], [0.0] ])

        self.lda1 : float = 0.0
        self.lda2 : float = 0.0

    def get_jacobian(self) -> np.ndarray:
        J = np.array(
            [
                [-self.l1*np.cos(self.th[0][0]), -self.l2*np.cos(self.th[1][0])],
                [-self.l1*np.sin(self.th[0][0]), -self.l2*np.sin(self.th[1][0])]
            ]
        )
        return J
    
    def get_foot_velcoity(self):
        J = self.get_jacobian()
        foot_vel = J @ self.dth
        return foot_vel
    
    def get_dth_gnd_contact(self):
        J = self.get_jacobian()
        M = np.array([
            [2*self.m1*self.l1**2, self.l1*self.l2*self.m1*np.cos(self.th[0][0]-self.th[1][0])],
            [self.l1*self.l2*self.m1*np.cos(self.th[0][0]-self.th[1][0]), self.l2**2*self.m1]
        ])
        M_inv = np.linalg.pinv(M)

        p = -np.linalg.pinv( (J @ M_inv @ J.T) ) @ J @ self.dth
        dth = self.dth + M_inv @ J.T @ p

        return dth
    
    def system(self, dt:float):
        foot_vel = self.get_foot_velcoity()
        
        if self.x2[1][0] <= 0 and foot_vel[1][0] < 0:
            self.dth = self.get_dth_gnd_contact()

            if abs(self.dth[0][0]) < 1e-5:
                self.dth[0][0] = 0.0

            if abs(self.dth[1][0]) < 1e-5:
                self.dth[1][0] = 0.0

        if self.x2[1][0] <= 0 and (self.tau[1][0] <= 0.0 and self.tau[0][0] <= 0.0):
        
            A = np.array([
                [2*self.m1*self.l1**2, self.l1*self.l2*self.m1*np.cos(self.th[0][0]-self.th[1][0]), self.l1*np.cos(self.th[0][0]), self.l1*np.sin(self.th[0][0])],
                [self.l1*self.l2*self.m1*np.cos(self.th[0][0]-self.th[1][0]), self.l2**2*self.m1, self.l1*np.cos(self.th[1][0]), self.l1*np.sin(self.th[1][0])],
                [-self.l1*np.cos(self.th[0][0]), -self.l2*np.cos(self.th[1][0]), 0.0, 0.0],
                [-self.l1*np.sin(self.th[0][0]), -self.l2*np.sin(self.th[1][0]), 0.0, 0.0]
            ])
            B = np.array([
                [self.l1*(G*self.m1*np.sin(self.th[0][0]) + G*self.m2*np.sin(self.th[0][0]) - self.l2*self.m1*np.sin(self.th[0][0]-self.th[1][0])*self.dth[1][0]**2)],
                [self.l2*(G*self.m2*np.sin(self.th[1][0]) + self.l1*self.m1*np.sin(self.th[0][0]-self.th[1][0])*self.dth[0][0]**2)],
                [-self.l1*np.sin(self.th[0][0])*self.dth[0][0]**2 - self.l2*np.sin(self.th[1][0])*self.dth[1][0]**2],
                [-self.l1*np.cos(self.th[0][0])*self.dth[0][0]**2 - self.l2*np.cos(self.th[1][0])*self.dth[1][0]**2]
            ])

            sol = np.matmul(np.linalg.inv(A), B) 
            self.ddth = sol[:2,]
    
            self.dth += self.ddth * dt
            self.th  += self.dth * dt 

        else:

            A = np.array([
                [2*self.m1*self.l1**2, self.l1*self.l2*self.m1*np.cos(self.th[0][0]-self.th[1][0])],
                [self.l1*self.l2*self.m1*np.cos(self.th[0][0]-self.th[1][0]), self.l2**2*self.m1]
            ])
            B = np.array([
                [self.l1*(G*self.m1*np.sin(self.th[0][0]) + G*self.m2*np.sin(self.th[0][0]) - self.l2*self.m1*np.sin(self.th[0][0]-self.th[1][0])*self.dth[1][0]**2)] + self.tau[0][0],
                [self.l2*(G*self.m2*np.sin(self.th[1][0]) + self.l1*self.m1*np.sin(self.th[0][0]-self.th[1][0])*self.dth[0][0]**2) + self.tau[1][0] ],
            ])
            self.ddth = np.matmul(np.linalg.inv(A), B) 
    
            self.dth += self.ddth * dt

        self.th  += self.dth * dt 
    
        self.x1[0][0] = self.x0[0][0] -self.l1 * np.sin(self.th[0][0])
        self.x1[1][0] = self.l1 * np.cos(self.th[0][0])

        self.x2[0][0] = self.x0[0][0] -self.l1*np.sin(self.th[0][0]) - self.l2*np.sin(self.th[1][0])
        self.x2[1][0] = self.l1 * np.cos(self.th[0][0]) + self.l2*np.cos(self.th[1][0])

        return self.x0, self.x1, self.x2, self.th

class RightBaseBipedal(BasedDynamic):
    def __init__(self, m0:float, m1:float, m2:float, l1:float, l2:float):
        super().__init__(m0, m1, m2, l1, l2)

        self.th   : np.ndarray = np.array([ [0.0], [0.0] ])
        self.dth  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddth : np.ndarray = np.array([ [0.0], [0.0] ])

        self.x1   : np.ndarray = np.array([ [self.x2[0][0] - self.l1*np.sin(self.th[0][0])], [self.l1*np.cos(self.th[0][0])] ])
        self.dx1  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddx1 : np.ndarray = np.array([ [0.0], [0.0] ])

        self.x0   : np.ndarray = np.array([ [self.x1[0][0] - self.l2*np.sin(self.th[1][0])], 
                                            [self.x1[1][0] + self.l2*np.cos(self.th[1][0])] ])
        self.dx0  : np.ndarray = np.array([ [0.0], [0.0] ])
        self.ddx0 : np.ndarray = np.array([ [0.0], [0.0] ])

    def get_jacobian(self) -> np.ndarray:
        J = np.array(
            [
                [-self.l1*np.cos(self.th[0][0]), -self.l2*np.cos(self.th[1][0])],
                [-self.l1*np.sin(self.th[0][0]), -self.l2*np.sin(self.th[1][0])]
            ]
        )
        return J
    
    def get_foot_velcoity(self):
        J = self.get_jacobian()
        foot_vel = J @ self.dth
        return foot_vel
    
    def get_dth_gnd_contact(self):
        J = self.get_jacobian()
        M = np.array([
            [self.m0*self.l1**2, self.l1*self.l2*self.m0*np.cos(self.th[0][0] - self.th[1][0])],
            [self.l1*self.l2*self.m0*np.cos(self.th[0][0] - self.th[1][0]), (self.m1 + self.m0)*self.l2**2]
        ])
        M_inv = np.linalg.pinv(M)

        p = -np.linalg.pinv( (J @ M_inv @ J.T) ) @ J @ self.dth
        dth = self.dth + M_inv @ J.T @ p

        return dth
    
    def system(self, dt:float):
        foot_vel = self.get_foot_velcoity()
        
        if self.x0[1][0] <= 0 and foot_vel[1][0] < 0:
            self.dth = self.get_dth_gnd_contact()

            if abs(self.dth[0][0]) < 1e-5:
                self.dth[0][0] = 0.0

            if abs(self.dth[1][0]) < 1e-5:
                self.dth[1][0] = 0.0

        if self.x0[1][0] <= 0 and (self.tau[1][0] <= 0.0 and self.tau[0][0] <= 0.0):
            A = np.array([
                [self.m0*self.l1**2, self.l1*self.l2*self.m0*np.cos(self.th[0][0] - self.th[1][0]), self.l1*np.cos(self.th[0][0]), self.l1*np.sin(self.th[0][0])],
                [self.l1*self.l2*self.m0*np.cos(self.th[0][0] - self.th[1][0]), (self.m1 + self.m0)*self.l2**2, self.l2*np.cos(self.th[1][0]), self.l2*np.sin(self.th[1][0])],
                [-self.l1*np.cos(self.th[0][0]), -self.l2*np.cos(self.th[1][0]), 0.0, 0.0],
                [-self.l1*np.sin(self.th[0][0]), -self.l2*np.sin(self.th[1][0]), 0.0, 0.0]
            ])

            B = np.array([
                [self.l1*self.m0*(G*np.sin(self.th[0][0]) - self.l2*np.sin(self.th[0][0]-self.th[1][0])*self.dth[1][0]**2)],
                [self.l2*(G*self.m0*np.sin(self.th[1][0]) + G*self.m1*np.sin(self.th[1][0]) + self.l1*self.m0*np.sin(self.th[0][0]-self.th[1][0])*self.dth[0][0]**2)],
                [-self.l1*np.sin(self.dth[0][0])*self.dth[0][0]**2 - self.l2*np.sin(self.th[1][0])*self.dth[1][0]**2],
                [-self.l1*np.cos(self.dth[0][0])*self.dth[0][0]**2 - self.l2*np.cos(self.th[1][0])*self.dth[1][0]**2]
            ])

            sol = np.matmul(np.linalg.inv(A), B) 
            self.ddth = sol[:2,]
       
            self.dth += self.ddth * dt
            self.th  += self.dth * dt 

        else:
            A = np.array([
                [self.m0*self.l1**2, self.l1*self.l2*self.m0*np.cos(self.th[0][0] - self.th[1][0])],
                [self.l1*self.l2*self.m0*np.cos(self.th[0][0] - self.th[1][0]), (self.m1 + self.m0)*self.l2**2]
            ])
            B = np.array([
                [self.l1*self.m0*(G*np.sin(self.th[0][0]) - self.l2*np.sin(self.th[0][0]-self.th[1][0])*self.dth[1][0]**2) - self.tau[0][0] ],
                [self.l2*(G*self.m0*np.sin(self.th[1][0]) + G*self.m1*np.sin(self.th[1][0]) + self.l1*self.m0*np.sin(self.th[0][0]-self.th[1][0])*self.dth[0][0]**2) - self.tau[1][0] ]
            ])
            self.ddth = np.matmul(np.linalg.inv(A), B) 

            self.dth += self.ddth * dt
            self.th  += self.dth * dt 

        self.x1[0][0] = self.x2[0][0] - self.l2 * np.sin(self.th[1][0])
        self.x1[1][0] = self.l2 * np.cos(self.th[1][0])

        self.x0[0][0] = self.x2[0][0] -self.l2*np.sin(self.th[1][0]) - self.l1*np.sin(self.th[0][0])
        self.x0[1][0] = self.l2 * np.cos(self.th[1][0]) + self.l1*np.cos(self.th[0][0])

        return self.x0, self.x1, self.x2, self.th
        
class WalkingPhase(Enum):
    RIGHT_FOOT_STANCE = auto()
    LEFT_FOOT_STANCE = auto()

class DoublePendulum(BasedDynamic):
    def __init__(self):
        super().__init__(m0=1.0, m1=1.0, m2=1.0, l1=1.0, l2=1.0)

        self.walking_phase      : WalkingPhase     = WalkingPhase.LEFT_FOOT_STANCE
        self.right_base_dynamic : RightBaseBipedal = RightBaseBipedal(self.m0, self.m1, self.m2, self.l1, self.l2)
        self.left_base_dynamic  : LeftBaseBipedal  = LeftBaseBipedal(self.m0, self.m1, self.m2, self.l1, self.l2)
        self.current_dynamic    : BasedDynamic     = self.left_base_dynamic

        self.phase_lock         : threading.Lock   = threading.Lock()

    def switch_left_stance_phase(self):
        self.phase_lock.acquire()
        if self.walking_phase != WalkingPhase.LEFT_FOOT_STANCE:
            if self.current_dynamic.x0[1][0] <= 0.0:
                print("switch to left foot stance")
                self.walking_phase = WalkingPhase.LEFT_FOOT_STANCE
                self.current_dynamic = self.left_base_dynamic
                
                self.current_dynamic.x0 = self.x0
                self.current_dynamic.x1 = self.x1
                self.current_dynamic.x2 = self.x2

                self.current_dynamic.th[0][0] = self.th[0][0] - np.pi
                self.current_dynamic.th[1][0] = self.th[1][0] - np.pi
        self.phase_lock.release()
    
    def switch_right_stance_phase(self):
        self.phase_lock.acquire()
        if self.walking_phase != WalkingPhase.RIGHT_FOOT_STANCE:
            if self.current_dynamic.x2[1][0] <= 0.0:
                print("switch to right foot stance")
                self.walking_phase = WalkingPhase.RIGHT_FOOT_STANCE
                self.current_dynamic = self.right_base_dynamic
                
                self.current_dynamic.x0 = self.x0
                self.current_dynamic.x1 = self.x1
                self.current_dynamic.x2 = self.x2

                self.current_dynamic.th[0][0] = self.th[0][0] - np.pi
                self.current_dynamic.th[1][0] = self.th[1][0] - np.pi
        self.phase_lock.release()

    def system(self, dt:float):
        self.phase_lock.acquire()
        
        if self.walking_phase == WalkingPhase.RIGHT_FOOT_STANCE:
            self.x0, self.x1, self.x2, self.th = self.current_dynamic.system(dt)
            # self.switch_left_stance_phase()

        elif self.walking_phase == WalkingPhase.LEFT_FOOT_STANCE:
            self.x0, self.x1, self.x2, self.th = self.current_dynamic.system(dt)
            # self.switch_right_stance_phase()

        self.phase_lock.release()

    def run(self, ax:Axes, dt:float):
        self.system(dt=dt)
        self.plot_robot(ax)
    
    def plot_robot(self, ax:Axes):
        ax.plot(self.x0[0][0], self.x0[1][0], 'o')

        ax.plot(self.x1[0][0], self.x1[1][0], 'o')
        ax.plot( (self.x0[0][0], self.x1[0][0]), 
                 (self.x0[1][0], self.x1[1][0]) )
        
        ax.plot(self.x2[0][0], self.x2[1][0], 'o')
        ax.plot( (self.x1[0][0], self.x2[0][0]), 
                 (self.x1[1][0], self.x2[1][0]) )

def on_press(event, bipedal:DoublePendulum):
    if event.key == 'z':
        bipedal.switch_right_stance_phase()
    elif event.key == 'x':
        bipedal.switch_left_stance_phase()
    elif event.key == '1':
        bipedal.current_dynamic.tau[0][0] = 30.0
    elif event.key == '2':
        bipedal.current_dynamic.tau[1][0] = 30.0

def on_relase(event, bipedal:DoublePendulum):  
    if event.key == '1':
        bipedal.current_dynamic.tau[0][0] = 0.0
    elif event.key == '2':
        bipedal.current_dynamic.tau[1][0] = 0.0


if __name__ == "__main__":
    field = Field(size = 2.0)
    field_axis = field.get_axis()

    bipedal = DoublePendulum()

    field.frame_callback = lambda dt: bipedal.run(ax=field_axis, dt=dt)
    field.on_press = lambda event : on_press(event, bipedal)
    field.on_release = lambda event : on_relase(event, bipedal)

    field.start()
    field.spin()