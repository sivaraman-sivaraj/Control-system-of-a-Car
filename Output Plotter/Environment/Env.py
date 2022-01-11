import numpy as np
from   gym import Env
from   gym.utils import seeding
from gym.envs.classic_control import rendering
import car, reward, LOS
import torch 


class load(Env): # CarRL-v0
     metadata = {'render.modes': ['human', 'rgb_array'],
                 'video.frames_per_second': 2 } 
     
     def __init__(self,wp,dt,Tpd = np.deg2rad(2)):
         self.wp             = wp
         ### grid size ###
         self.dt             = dt
         self.Tpd            = Tpd
         self.grid_size      = 600
         #### initial conditions ####
         self.done,self.viewer   = False, None           # see here
         self.st_x, self.st_y    = 0,0
         # self.actions_set        = {'0':-2,'1': 0,'2': 2} 
         self.actions_set        = {'0':-20,'1': -5,'2': -1, '3':0, '4':1,'5':5, '6':20} 
         
     def reset(self):
         #######################
         ## H = [last quadrant, last waypoint,previous heading error, previous CTE,goal]
         ########################
         self.done                             = False
         self.current_state                    = torch.tensor([0.0,0.0,0.0,0.0])
         return self.current_state 
     
     def step(self,A):
         self.ip                              = self.current_state.clone().detach()
         self.op,self.S                       = car.activate(self.ip,A[1],np.deg2rad(self.actions_set[str(A[0])])) 
         self.psi_d,self.y_e                  = LOS.activate(self.S, self.wp)
         self.reward_a                        = reward.get(self.op[2],self.psi_d, self.y_e)
         #############################################################
         self.current_state                   = torch.tensor(self.op)
         
         self._  = "for yaw rate control"
         return self.current_state, [self.reward_a,self.S], self.done,self._ 
     
     def action_space_sample(self):
         n = np.random.randint(0,7)
         return n 
     
     def render(self,mode='human'):
         pass
 
     def close(self):
         pass
             
             
################################
########## To Check ############
################################
# Ship boundary points ##
# A = car_points1()
# B = car_points()
# print(A)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(6,6))
# plt.xlim(-5,5)
# plt.ylim(-5,5)
# for i in range(len(A)):
#     plt.scatter(A[i][0],A[i][1],color="g")
# for j in range(len(B)):
#     plt.scatter(B[j][0],B[j][1],color="r")
################################
############# End ##############
################################

