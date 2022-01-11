import numpy as np 
import matplotlib.pyplot as plt

def get(psi,psi_d,ye):
    """
    Parameters
    ----------
    psi   : actual heading
    psi_d : desired heading

    Returns
    -------
    reward

    """
    Error0  = abs(psi_d - psi)
    Error1  = abs(ye)
    Error   = -(Error0*Error1)
    return Error


###############################################
################# To Evaluate #################
###############################################
# Y = []
# for i in range(-400,400):
#     temp = get([0,0,0,np.deg2rad(i/10)],np.deg2rad(20))
    
#     Y.append(temp)

# plt.plot(Y)
# plt.axhline(y = 0.03490)
# plt.axhline(y = 0)
###############################################
###############################################



