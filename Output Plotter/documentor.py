import sys, os,pathlib,os
Environmet_Folder = pathlib.Path("Environment")
sys.path.insert(1,os.path.join(os.getcwd(),Environmet_Folder)) 
import matplotlib.pyplot as plt 
from matplotlib import animation 
import Q_network, track_points
import gym,numpy as np,random,time,math,os,sys
from gym.envs.registration import register 
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
###############################################
############### GIF Function ##################
###############################################
def Create_GIF(frames,filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save( os.path.join(os.getcwd(),filename), writer='imagemagick', fps=60) 
###############################################
###############################################
############################################### 
wp,Xp,Yp,L = track_points.R_curve(40)
dt      = 0.01
Tpd     = np.deg2rad(2) 
steps   = 650
##################################################
##### Registering the Environment in Gym #########
##################################################
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
      if 'CarRL-v0' in env:
          del gym.envs.registration.registry.env_specs[env]

register(id ='CarRL-v0', entry_point='Env:load',kwargs={'wp':wp,'dt' : dt, 'Tpd':Tpd}) 

##########################################
########### action selection #############
##########################################

def select_action(state):
    
   with torch.no_grad():
       clone = torch.clone(state)
       temp  = clone.detach().tolist() 
       op    = policy_net(torch.tensor(temp))
   return np.argmax(op.detach().numpy())
   
#############################################
################## Result ###################
#############################################
env                     = gym.make('CarRL-v0')
policy_net              = Q_network.FFNN() 
policy_net.load_state_dict(torch.load("W.pt")) 
# actions_set        = {'0':-2,'1': 0,'2': 2} # 3 actions
# actions_set        = {'0':-1,'1': -0.5,'2': 0, '3':0.5, '4':10} 
actions_set        = {'0':-20,'1': -5,'2': -1, '3':0, '4':1,'5':5, '6':20} 
# actions_set        = {'0':-3,'1': -1,'2': 0, '3':1, '4':3} 

car_current_state       = env.reset() 
States                  = [car_current_state] 
Y,Y_dot,Psi,Psi_dot     = [],[],[],[] 
Delta                   = []
frames = []
Ss                       = [[0.0,0.0]]
for i in range(steps):
    # frames.append(env.render(mode="rgb_array"))
    
    state_temp                            = States[-1]
    action                                = select_action(state_temp)
    observation, [reward,sp], done, _  = env.step([5,Ss[-1]]) # Select and perform an action 
    #############################################################
    States.append(observation) 
    Y.append(observation[0]) 
    Y_dot.append(observation[1]) 
    Psi.append(np.rad2deg(observation[2])) 
    Psi_dot.append(np.rad2deg(observation[3])+0.1)
    Delta.append(actions_set[str(action)])
    Ss.append(sp)
    
# env.close()
# Create_GIF(frames,filename='gym_animation.gif') 
dot = '\u0307'
def Plot1(Y,Y_dot,Delta):
    plt.figure(figsize=(9,6))
    plt.subplot(3,1,1) 
    plt.plot(Y,color="crimson",label="sway(y)") 
    plt.title("5 Action State : Control Effort($\delta$) and it's States(y,$\dot{y}$)") 
    plt.ylabel("meters(m)")
    plt.grid(color='b', linestyle='--', alpha=0.3) 
    plt.legend(loc="best")
    
    plt.subplot(3,1,2) 
    plt.plot(Y_dot,color="teal",label="sway rate($\dot{y}$)") 
    plt.ylabel("m/s")
    plt.grid(color='b', linestyle='--', alpha=0.3) 
    plt.legend(loc="best") 
    
    plt.subplot(3,1,3) 
    plt.plot(Delta,color="Orangered",label="streering command") 
    plt.ylabel("$\delta$ (in degree)")
    plt.grid(color='b', linestyle='--', alpha=0.3) 
    plt.xlabel("Time steps(in 0.01 s interval)") 
    # plt.ylim(0,1.1)
    plt.legend(loc="best")
    # plt.savefig("Sway_plot.jpg",dpi=420)
    plt.show()
    
    
Plot1(Y,Y_dot,Delta)


def Plot2(Psi,Psi_dot,Delta):
    plt.figure(figsize=(9,6))
    plt.subplot(3,1,1) 
    plt.plot(Psi,color="m",label="yaw Angle($\psi$)") 
    plt.title("5 Action State : Control Effort($\delta$) and it's States($\psi$,$\dot{\psi}$)") 
    plt.ylabel("deg")
    plt.grid(color='b', linestyle='--', alpha=0.3) 
    plt.legend(loc="best")
    
    plt.subplot(3,1,2) 
    plt.plot(Psi_dot,color="teal",label="yaw rate($\dot{\psi}$)") 
    plt.ylabel("deg/s")
    plt.grid(color='b', linestyle='--', alpha=0.3) 
    plt.legend(loc="best") 
    
    plt.subplot(3,1,3) 
    plt.plot(Delta,color="red",label="streering command") 
    plt.ylabel("$\delta$ (in degree)")
    plt.grid(color='b', linestyle='--', alpha=0.3) 
    plt.xlabel("Time steps(in 0.01 s interval)")
    # plt.ylim(0,1.1)
    plt.legend(loc="best")
    plt.savefig("dqn_rudder.jpg",dpi=420)
    plt.show()
    

Plot2(Psi,Psi_dot,Delta)


##############################################################
##############################################################
x_path, y_path = [],[]
for i in range(len(Ss)):
    x_path.append(Ss[i][0])
    y_path.append(Ss[i][1]) 
    
plt.figure(figsize=(6,6))
plt.plot(x_path,y_path,'r',label = "Trained Path")
plt.axhline(y=40, color="grey", alpha = 0.2)
plt.axvline(x=40, color="grey", alpha= 0.2)
plt.title("DQN Method of Trajectory Tracking")
plt.plot(Xp,Yp,'g',label="Target Path")
plt.grid(linestyle="--")
plt.xlabel("sway(y)")
plt.ylabel("Surge(x)") 
plt.legend(loc="best")
plt.savefig("DQN_PT.jpg",dpi=420)








