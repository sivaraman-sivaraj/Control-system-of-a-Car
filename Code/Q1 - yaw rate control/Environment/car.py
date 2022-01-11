import numpy as np
import matplotlib.pyplot as plt

def activate(ip,delta,dt=0.01):
    """
    The Governing Equation :
        
        ẋ(t) = A*x + B*u
    
    Parameters
    ----------
    ip       : [y,y_dot,psi,psi_dot]
    delta    : change of angle in radian
        
    Returns
    -------
    next state

    """
    
    m           = 2325     # mass of the car
    Iz          = 4132     # Momnet of inetia about CG
    lf,lr       = 2,2      # length of front and rear axle from CG 
    Vx          = 10       # Surge Velocity
    C_alpha_f   = 100000   # physical constant for front wheel
    C_alpha_r   = 126000   # physical constant for rear wheel 
    
    #############################################################
    ####################### State Matrix ########################
    #############################################################
    A22         = -(2*(C_alpha_f+C_alpha_r))/(m*Vx)
    A24         = -Vx-(2*(C_alpha_f*lf - C_alpha_r*lr)/(m*Vx)) 
    A42         = -2*(C_alpha_f*lf - C_alpha_r*lr)/(Iz*Vx)
    A44         = -2*((C_alpha_f*lf**2)+(C_alpha_r*lr**2))/(Iz*Vx)
    
    A  = np.array([[0,1,0,0],
                   [0,A22,0,A24],
                   [0,0,0,1],
                   [0,A42,0,A44]])
    #############################################################
    ######################### Input Matrix ######################
    #############################################################
    B  = np.array([[0],
                   [2*C_alpha_f/m],
                   [0],
                   [2*lf*C_alpha_f/Iz]])
    ############################################################
    ############################################################
    X      = np.reshape(np.array(ip),(4,1))
    X_dot  = A.dot(X) + B*delta
    X_dot  *= dt
    X_dot     = np.reshape(X_dot,(4)) 
    ###########################################################
    OP    = [ip[0]+X_dot[0],ip[1]+X_dot[1],ip[2]+X_dot[2],ip[3]+X_dot[3]]
    
    return OP

# ss  = activate([0,0,0,0], np.deg2rad(0.5),1)
# print(ss)


###############################################
################# To Evaluate #################
###############################################
# Y,Y1,Psi,Psi1 = [],[],[],[]
# temp            = [0,0,0,0]
# Data          = [temp]
# N             = 300
# dt            = 0.01
# for i in range(N):
#     if i%2 == 0 :
#         op = activate(Data[-1], np.deg2rad(-4), dt)
#     elif i%2 == 1:
#         op = activate(Data[-1], np.deg2rad(4), dt)
#     Data.append(op)
#     Y.append(op[0])
#     Y1.append(op[1])
#     Psi.append(np.rad2deg(round(op[2],3)))
#     Psi1.append(op[3])

    

# plt.figure(figsize=(9,8))
# plt.subplot(4,1,1)
# plt.plot(Y,color='g', label = "Sway")
# plt.ylabel("meters")
# plt.legend(loc="best")
# plt.title("Model Evalution for 3 seconds : (-4$^\circ$ & 4$^\circ$) alternate steering command  ")
# plt.grid()

# plt.subplot(4,1,2)
# plt.plot(Y1,color='m', label = "Sway Velocity")
# plt.ylabel("m/s")
# plt.legend(loc="best")
# plt.grid()

# plt.subplot(4,1,3)
# plt.plot(Psi,color='teal', label = "Yaw Angle")
# plt.ylabel("deg")
# plt.legend(loc="best")
# plt.grid()

# plt.subplot(4,1,4)
# plt.plot(Psi1,color='crimson', label = "Yaw Angle rate")
# plt.ylabel("deg/s^2")
# plt.legend(loc="best")
# plt.grid()
# plt.savefig("PN.jpg",dpi=420)
###############################################
###############################################
###############################################
    
    
    
    
    