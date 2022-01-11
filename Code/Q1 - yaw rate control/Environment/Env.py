import numpy as np
from   gym import Env
from   gym.utils import seeding
from gym.envs.classic_control import rendering
import car, reward
import torch



class load(Env): # CarRL-v0
     metadata = {'render.modes': ['human', 'rgb_array'],
                 'video.frames_per_second': 2 } 
     
     def __init__(self,dt,Tpd = np.deg2rad(2)):
         ### grid size ###
         self.dt             = dt
         self.Tpd            = Tpd
         self.grid_size      = 600
         #### initial conditions ####
         self.done,self.viewer   = False, None           # see here
         self.st_x, self.st_y    = 0,0
         # self.actions_set        = {'0':-2,'1': 0,'2': 2} 
         self.actions_set        = {'0':-3,'1': -1,'2': 0, '3':1, '4':3} 
         
     def reset(self):
         #######################
         ## H = [last quadrant, last waypoint,previous heading error, previous CTE,goal]
         ########################
         self.done                             = False
         self.current_state                    = torch.tensor([0.0,0.0,0.0,0.0])
         return self.current_state 
     
     def step(self,A):
         self.ip                              = self.current_state.clone().detach()
         self.op                              = car.activate(self.ip,np.deg2rad(self.actions_set[str(A)])) 
         self.reward_a                        = reward.get_based_psi_dot(self.op,self.Tpd )
         # self.reward_a                        = reward.get_based_on_y(ip)(self.op) 
         #############################################################
         self.current_state                   = torch.tensor(self.op)
         self.error                           = abs(self.Tpd  - self.current_state[3])
         if abs(self.op[0]) > 1:
             self.done = True 
         self._  = "for yaw rate control"
         return self.current_state, [self.reward_a,self.error], self.done,self._ 
     
     def action_space_sample(self):
         n = np.random.randint(0,3)
         return n 
     
     def render(self,mode='human'):
         pass 
 
     def close(self):
         pass
             
             


