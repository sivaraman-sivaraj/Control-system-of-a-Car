import numpy as np
import matplotlib.pyplot as plt

def activate(ip,S,delta,dt=0.01):
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
    ########################################################### 
    a,b = S[0],S[1]
    a  += Vx * np.sin(OP[2].item())*dt
    b  += Vx * np.cos(OP[2].item())*dt
    return OP,[a,b]


activate([0,0,0,0],[0,0],0.2,0.01)
###############################################
################# To Evaluate #################
###############################################
# import track_points
# wp,Xp,Yp,L = track_points.R_curve(40)
# Spatial_position = [[0,0]]
# x_path0, y_path0 = [0],[0]
# Y,Y1,Psi,Psi1 = [],[],[],[]
# temp            = [0,0,0,0]
# Data          = [temp]
# N             = 630
# dt            = 0.01
# for i in range(N):
#     op,C = activate(Data[-1], Spatial_position[-1], np.deg2rad(5.901875), dt)
#     Data.append(op)
#     Y.append(op[0])
#     Y1.append(op[1])
#     Psi.append(np.rad2deg(round(op[2],3)))
#     Psi1.append(op[3]) 
#     Spatial_position.append(C)
#     #########################
#     x_path0.append(C[0])  
#     y_path0.append(C[1])  

# RC0 = round(max(x_path0)/2,2)


# # N                 = 740
# # Spatial_position1 = [[0,0]]
# # x_path1, y_path1 = [0],[0] 
# # temp            = [0,0,0,0]
# # Data          = [temp]
# # for i in range(N):
# #     if i%2 == 0 :
# #         op,C = activate(Data[-1], Spatial_position1[-1], np.deg2rad(20), dt)
# #     elif i%2 == 1:
# #         op,C = activate(Data[-1], Spatial_position1[-1], np.deg2rad(20), dt)
# #     Data.append(op)
# #     Spatial_position1.append(C)
# #     #########################
# #     x_path1.append(C[0])  
# #     y_path1.append(C[1])  

# # RC1 = round(max(x_path1)/2,2)

# # N                 = 1480
# # Spatial_position1 = [[0,0]]
# # x_path2, y_path2 = [0],[0] 
# # temp            = [0,0,0,0]
# # Data          = [temp]
# # for i in range(N):
# #     if i%2 == 0 :
# #         op,C = activate(Data[-1], Spatial_position1[-1], np.deg2rad(10), dt)
# #     elif i%2 == 1:
# #         op,C = activate(Data[-1], Spatial_position1[-1], np.deg2rad(10), dt)
# #     Data.append(op)
# #     Spatial_position1.append(C)
# #     #########################
# #     x_path2.append(C[0])  
# #     y_path2.append(C[1])  

# # RC2 = round(max(x_path2)/2,2)

# # N                 = 1480*2
# # Spatial_position1 = [[0,0]]
# # x_path3, y_path3 = [0],[0] 
# # temp            = [0,0,0,0]
# # Data          = [temp]
# # for i in range(N):
# #     if i%2 == 0 :
# #         op,C = activate(Data[-1], Spatial_position1[-1], np.deg2rad(5), dt)
# #     elif i%2 == 1:
# #         op,C = activate(Data[-1], Spatial_position1[-1], np.deg2rad(5), dt)
# #     Data.append(op)
# #     Spatial_position1.append(C)
# #     #########################
# #     x_path3.append(C[0])  
# #     y_path3.append(C[1])  

# # RC3 = round(max(x_path3)/2,2)

# print("The diameter of the circular motion is: ",max(x_path0))
# plt.figure(figsize=(6,6))
# plt.plot(x_path0, y_path0,label="Trajectory Path for $\delta$ = 5.9$^\circ$ command, RC = 39.986" ,color="crimson")
# # plt.plot(x_path1, y_path1,label="For $\delta$ = 20$^\circ$, RC = "+str(RC1) ,color="m")
# # plt.plot(x_path2, y_path2,label="For $\delta$ = 10$^\circ$, RC = "+str(RC2) ,color="b")
# # plt.plot(x_path3, y_path3,label="For $\delta$ = 5$^\circ$, RC = "+str(RC3) ,color="teal")
# plt.axvline(x=0,color="black",)
# plt.axhline(y = 0, color="black")
# plt.axvline(x= 40,color="g", linestyle="--")
# plt.scatter(40,0,marker="8",label="Radius Center")
# plt.xlabel("sway(y)")
# plt.ylabel("Surge(x)")
# plt.grid(linestyle="--")
# plt.plot(Xp,Yp,'g',label="Target Path")
# plt.title("Trajectory Matching")
# plt.legend(loc="best")
# plt.savefig("Tmatch.jpg",dpi=420)
###############################################
###############################################
###############################################
    
    
    
    
    