import numpy as np 
import matplotlib.pyplot as plt

def get_based_psi_dot(ip,TPd):
    """
    Parameters
    ----------
    ip               : [y,y_dot,psi,psi_dot]
    TPd              :Thresold psi dot

    Returns
    -------
    reward

    """
    Error = TPd - ip[3]
    return -abs(Error)*1000

def get_based_on_psi_dot2(ip,S,Tpd):
    Vx = 10
    r  = np.square(S[0] - 40) + np.square(S[1])
    R  = np.sqrt(r)
    
    psi_rate = Vx/R
    
    Error = Tpd - psi_rate
    return (1/(Error+1))*100
    


###############################################
################# To Evaluate #################
###############################################
# Y = []
# for i in range(-4,4):
#     temp = get_based_psi_dot([0,0,0,np.deg2rad(i)],np.deg2rad(2))
#     # temp = get_based_on_y([abs((i/100)),0,0,0])
#     Y.append(temp)

# plt.plot(Y)
# plt.axhline(y = 0.03490)
# plt.axhline(y = 0)
###############################################
###############################################



