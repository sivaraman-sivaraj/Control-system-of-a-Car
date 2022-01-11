###############################################
###############################################
###############################################
import sys, os,pathlib
Environmet_Folder = pathlib.Path("Environment")
sys.path.insert(1,os.path.join(os.getcwd(),Environmet_Folder))
import torch,torch.nn as nn, torch.optim as optim
import gym,numpy as np,random,time,math,os,sys,pathlib
from gym.envs.registration import register
from collections import namedtuple, deque
from itertools import count
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
import matplotlib.pyplot as plt 
from matplotlib import animation
import Q_network
###############################################
import graph
graph.create_op_folder()
weight_path = os.getcwd()+"\\" +"Results"+"\\"+"0. Weights"
reward_path = os.getcwd()+"\\" +"Results"+"\\"+"1. Reward_Plots"
error_path  = os.getcwd()+"\\" +"Results"+"\\"+"2. Error_Plots" 
others_path = os.getcwd()+"\\" +"Results"+"\\"+"3. Others" 
###############################################
###############################################
dt      = 0.1
Tpd     = np.deg2rad(2) 
steps   = 300
No_Episodes = 2151
##################################################
##### Registering the Environment in Gym #########
##################################################
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
      if 'CarRL-v0' in env:
          del gym.envs.registration.registry.env_specs[env]

register(id ='CarRL-v0', entry_point='Env:load',kwargs={'dt' : dt, 'Tpd':Tpd})
###################################################
################## DQN Training ###################
###################################################

######################################
############## Memory ################
######################################
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    def all_sample(self):
        return self.memory
    
memory = ReplayMemory(3000) # we can alter it for some different
########################################
########## eps calculation #############
########################################
def eps_calculation(i_episode):
    """
    decaying epision value is an exploration parameter
    
    """
    
    start = 0.999
    end   = 0.1
    eps = end + (start- end) * math.exp(-0.5 * i_episode / 100)
    return eps 

##########################################
########### action selection #############
##########################################

def select_action(state,eps):
    
    sample = random.random()
    
    if sample > eps:
        with torch.no_grad():
            clone = torch.clone(state)
            temp  = clone.detach().tolist() 
            # IP = []
            # for i in range(len(temp)):
            #     t1 = round(temp[i],4)
            #     IP.append(t1)
            # print(temp)
            op    = policy_net(torch.tensor(temp))
            return np.argmax(op.detach().numpy())
    else:
        return env.action_space_sample() 
#########################################
############# optimizer #################
#########################################

def optimize_model():
    if len(memory) < batch_size:
        return 0
    
    transitions = memory.sample(batch_size)
    batch       = Transition(*zip(*transitions))
    # print(batch)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)),
                                  device=device, dtype=torch.bool)
    
    non_final_next_states = torch.tensor(batch.next_state)
    state_batch  = torch.tensor(batch.state)
    action_batch = torch.tensor(batch.action)
    reward_batch = torch.tensor(batch.reward)
    action_batch = action_batch.unsqueeze(1)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values   = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    expected_state_action_values = expected_state_action_values.unsqueeze(1)
    # Compute F1 smoothLoss or MSE loss()
    criterion = nn.MSELoss()#SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss

#########################################################
###### Initializing the Environment in Gym & DQN ########
#########################################################
env                     = gym.make('CarRL-v0')
device                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size              = 64
gamma                   = 0.99
target_update           = 10
policy_net              = Q_network.FFNN()
target_net              = Q_network.FFNN()
policy_net              = policy_net.to(device=device)
target_net              = target_net.to(device=device)
optimizer               = optim.Adam(policy_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)#optim.RMSprop()
 
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

############################################
############ Main function #################
############################################   

def Train_DQN(N):
    """
    Parameters
    ----------
    N           : Number of Episodes

    Returns
    -------
    NoEpi                : Episode Duration for N episodes
    CUmulative_reward    : Cumulative reward for N episodes
    HEs                  : Average Heading Error for N episodes
    MSEs                 : Mean Square Error for N episodes
    """
    NoEpi             = [] # episode duration
    Cumulative_reward = [] # cumulative reward
    HEs               = [] # Average Heading error for an episode
    MSEs              = [] # Mean Square error for N episode
    for i_episode in range(N):
        total_reward = 0
        total_he     = 0 
        total_mse    = 0
        eps   = eps_calculation(i_episode)
        if i_episode % 200 == 0:
            print("Episode: ",i_episode,"Running....")
        ##############################################
        #### Initialize the environment and state ####
        ##############################################
        car_current_state    = env.reset()
        state                = car_current_state
        LoE                  = 1                     # length of episode
        for it in count():
            
            # env.render()
            action                                = select_action(state,eps)
            observation, [reward,error], done, _  = env.step(action) # Select and perform an action
            LoE                                   += 1
            
            if it >= steps:
                done = True
            
            if done == True:
                NoEpi.append(it+1)
                break
            
            next_state = observation                   # Observe new state
            
            #######################################
            #### Store the transition in memory ###
            #######################################
            st_m   = state.tolist()
            n_st_m = next_state.tolist()
            r_m    = reward.item()
            
            memory.push(st_m, action, n_st_m, r_m)
            #################################
            ###### Move to the next state ###
            #################################
            state = observation.clone().detach()
            #################################
            ######### optimization ##########
            #################################
            loss = optimize_model()
            total_reward += reward.item()
            total_he     += np.rad2deg(error)
            total_mse    += float(loss)
        # env.close()
        HEs.append(total_he/LoE) # theta is global declaration
        Cumulative_reward.append(total_reward/LoE)
        MSEs.append(total_mse/LoE)
        ##############################################################################
        ####### Update the target network, copying all weights and biases in DQN #####
        ###############################################################################
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        ################################################################
        ####### Periodical Weights Savings and result checking #########
        ################################################################
        if i_episode % 100 == 0 and i_episode != 0:
            torch.save(policy_net.state_dict(),os.path.join(weight_path, "W"+str(i_episode)+".pt"))
        if i_episode % 100 == 0 and i_episode != 0:
            np.save(os.path.join(others_path, "NoEpi.npy"),NoEpi)
            np.save(os.path.join(others_path, "Cumulative_reward.npy"),Cumulative_reward)
            np.save(os.path.join(others_path, "HEs.npy"),HEs)
            np.save(os.path.join(others_path, "MSEs.npy"),MSEs )
            graph.plot_result1(NoEpi, Cumulative_reward,reward_path,i_episode)
            graph.plot_result2(HEs,MSEs,error_path,i_episode)
    
    ###########################################
    ######## Final Weights Savings ############
    ###########################################
    torch.save(policy_net.state_dict(),os.path.join(weight_path, "W.pt"))
    graph.plot_result1(NoEpi, Cumulative_reward,reward_path)
    graph.plot_result2(HEs,MSEs,error_path)
    np.save(os.path.join(others_path, "NoEpi.npy"),NoEpi)
    np.save(os.path.join(others_path, "Cumulative_reward.npy"),Cumulative_reward)
    np.save(os.path.join(others_path, "HEs.npy"),HEs)
    np.save(os.path.join(others_path, "MSEs.npy"),MSEs )
    
        

# start = time.time()
# Train_DQN(No_Episodes)

# print("complete...!")
# end = time.time()
# print("Total time has taken for Training the process : ",(round((end-start)/60,1)), " minutes" )
# with open(os.path.join(others_path, "time.txt"), "w") as text_file:
#     text_file.write("Total time has taken for Training the process : {0} minutes".format(round((end-start)/60,1)))
############################################
################## End #####################
############################################



