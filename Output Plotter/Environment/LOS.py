import numpy as np


def nearest_point(x,y,SP):
    """
    Parameters
    ----------
    x,y : spatial position of the agent
    SP  : separated points in the prior quadrant

    Returns
    -------
    nearest waypoints index

    """
    D              = dict()
    error_distance = list()                         # calculating the euclidian distance of all Separated Points
    for i in range(len(SP)):
       er_temp         = np.sqrt(((SP[i][0]-x)**2)+((SP[i][1]-y)**2))
       error_distance.append(er_temp)
       D[str(er_temp)] = i

    sorted_distance = sorted(error_distance)    # arranging the points in ascending order
    k               = D[str(sorted_distance[0])] 
    return k                                    # point index

def get_y_e_HE(ip,wp_k,wp_k_1):
    """
    Parameters
    ----------
    wp_k          :     (x_k,y_k)              - K_th way point  
    wp_k_1        :     (x_k+1,y_k+1)          - K+1_th way point 
    
    Returns
    -------
    cross track error

    """
    ###############################################
    ## Horizontal path tangential angle/ gamma  ###
    ###############################################
    del_x = wp_k_1[0]-wp_k[0]
    del_y = wp_k_1[1]-wp_k[1]
    g_p = np.arctan2(del_y, del_x)
    #########################################
    ###cross track error calculation (CTE) ##
    #########################################
    y_e     = -(ip[0]-wp_k[0])*np.sin(g_p) + (ip[1]-wp_k[1])*np.cos(g_p)  # Equation 24
    #############################
    ## finding the del_h value ##
    #############################
    delta_h        = 4              # look ahead distance
    ##########################################
    ## Calculation of desired heading angle ##
    ##########################################
    psi_d          = g_p + np.arctan2(-y_e,delta_h)  # Desired Heading angle # equation 29
    return psi_d,y_e



def activate(ip,S_prp):
    
    #############################################
    ######## Choosing the best way points #######
    #############################################
    SP        = S_prp
    wp_near   = nearest_point(ip[0],ip[1], SP) # nearest waypoint index
    if (wp_near+1) != len(S_prp):
        wp_k, wp_k_1 =  S_prp[wp_near],S_prp[wp_near+1]
    else:
        wp_k, wp_k_1 =  S_prp[wp_near-1],S_prp[wp_near]
    #############################################
    ###### Calculating the CTE ##################
    #############################################
    psi_d,y_e         =  get_y_e_HE(ip, wp_k, wp_k_1)
    
    return psi_d, y_e


#########################################
############## To Check #################
#########################################
# import matplotlib.pyplot as plt
# import track_points
# import reward

# wp,x,y,L   = track_points.R_curve(40)
# R = []
# for i in range(len(wp)):
#     print(i)
#     ip   = [0,0,0,wp[i][0],wp[i][1],0,0,0]
#     op   = [0,0,0,wp[i][0],wp[i][1],0,0,0]
#     y_e  = activate(ip,wp)
#     ss   = reward.get_based_on_y(y_e)
#     R.append(ss)
    
# plt.plot(R)
########################################
######### End ##########################
########################################











