import numpy as np
import matplotlib.pyplot as plt 

def Distance(A,B):
    D = np.sqrt(np.square(A[0]-B[0]) + np.square(A[1]-B[1]))
    return D

def interpretval(P1,P2,intv):
    a,b    = np.array(P1), np.array(P2)
    DV_    = b-a
    DV     = DV_/np.linalg.norm(DV_)
    l,m    = DV[0],DV[1]
    newpt  = [intv*l+P1[0],intv*m+P1[1]]
    return newpt

def Eq_Distance_maker(wp,lbp):
    WP_es = [wp[0]]
    
    current_pnt = wp[0]
    index_first = 1
    
    k           = 0
    
    Tot_dis     = 0
    for ii in range(len(wp)-1):
        temp     = Distance(wp[ii],wp[ii+1])
        Tot_dis += temp
        
    Int_Ds      = 2*lbp 
    NP          = int(Tot_dis / Int_Ds)
    
    for k in range(NP):
        New_presence   = []
        dis_sum   = 0
        pt_now    = current_pnt
        kk        = 0
        pt_target = wp[index_first]
        remainder = Int_Ds
        while len(New_presence) == 0:
            dis_temp = Distance(pt_now,pt_target)
            dis_sum  += dis_temp
            if dis_sum >= Int_Ds:
                newpt  = interpretval(pt_now,pt_target,remainder)
                New_presence.append(newpt)
            else:
                remainder -= dis_temp
                pt_now     = pt_target
                kk        += 1
                if index_first + kk > len(wp):
                    newpt  = wp[-1]
                    New_presence.append(newpt)
                else:
                    pt_target = wp[index_first+kk]
        WP_es.append(newpt)
        current_pnt   = newpt
        index_first   += kk
    return WP_es



def R_curve(r):
    P,X,Y = list(), list(), list() 
    
    for i in range(r):
        temp = np.sqrt( (r**2) - (i -r)**2 )
        X.append(i) 
        Y.append(temp)
        P.append([i,temp]) 
    
    L  = np.pi * r/2 
    SP = Eq_Distance_maker(P,1)
    return SP,X,Y,L 


def Arc_spline():
    wp        = []
    X,Y       = [],[]
    L         = 0 
        
    def f(x):
        num = 250
        den = 1 + np.exp(-0.03*x)
        return num/den
    
    for i in range(-200,200):
        temp  = f(i)
        wp.append([(i+200)/6,temp]) 
        X.append((i+200)/6)
        Y.append(temp/6)
    SP = Eq_Distance_maker(wp,8)
    return SP,X,Y,L


##########################################
##########################################
# P,X,Y,L = R_curve(40)
# # P,X,Y,L = Arc_spline()
# plt.figure(figsize=(9,6))
# plt.plot(X[::8],Y[::8],marker = "*")
# plt.title("Radius of Curvature of 40m")
# plt.grid()
# print(L)
##########################################
##########################################

