############################################################################
############## in version 2 of HTS - Semi transport conductance.  #####################
############## Introducing third terms from left side and right side
##############################################################################
import numpy as nm
import cmath as cm
import pdb
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy.sparse as spr
import sys
### global constants
m_0 = 9.10E-31 # in Kg
m_eff = 0.07*m_0
h = 4.135E-15  # planks in eV
#h = 6.626E-34
h_bar = h/(2*nm.pi)
h_bar_sq = h_bar**2 
e = 1.6E-19 
Kb = 1.38E-23  #8.617E-5 eVK-1 # 1.38E-23 in    JK-1 
#E = 0.5 

count =1 
##########Simulation constants ##############
Tc = 90  # 9 for Nb and 90 for YBCO
E_fermi =2  #E in eV # in J for InGaas
K_fermi = nm.sqrt(2*m_eff*E_fermi/(e*h_bar_sq))  ### accounting for the units. Energy is in eV, so there should be a 'e' there
################ simulation constant calculations




#########Simulation parameters############
del_0 = 1.764*Kb*Tc
#T_range = nm.concatenate([nm.arange(0.01,1,0.01),nm.arange(1,90,1)])
T_range = nm.arange(1,10,5)
V_range = nm.arange(0,2E-3,1E-4)  
E_range = nm.arange(-4*del_0,4*del_0,del_0/100)  ## assume symmetric about E=0 
th_step = nm.pi/10000 
th_range =nm.arange(-nm.pi/2,nm.pi/2,th_step) 
z=0
fileName = "data Z = "+str(z)+" sWave.txt" #### File Name to save
E_fermi_L =2  #E in eV # in J for InGaas
E_fermi_R =2  #E in eV # in J for InGaas
K_fermi_L = nm.sqrt(2*m_eff*E_fermi_L/(e*h_bar_sq)) 
K_fermi_R = nm.sqrt(2*m_eff*E_fermi_R/(e*h_bar_sq)) 
lam = K_fermi_R/K_fermi_L 
#sig_N = 4*lam/((1 + lam)^2 + 4*z^2) 
sig_N = 1  ### for un normalized currents
c1 = 1

def getDel(th,delT,k):
    delTemp = delT
    #delTemp = delT*nm.cos(2*th)
    return cm.polar(delTemp)

def getSigT(th,E):
    M_matrix = spr.dokmatrix((N_value*8,N_value*8),dtype=float32)
    for n in N_range:
        ################ Get the values for the left side of the nth segment
        K_L_P = nm.sqrt(K_fermi_L**2 + 2*m_eff*E/(e*h_bar_sq)) ### not changing these for now. They will eventually change in each segment
        K_L_N = nm.sqrt(K_fermi_L**2 - 2*m_eff*E/(e*h_bar_sq))### not changing these for now. They will eventually change in each segment
        [del_p_l,psi_p_l] = getDel(th,delT(n),K_L_P) 
        [del_n_l,psi_n_l] = getDel(th,delT(n),-K_L_N)
        u_p_l =cm.sqrt((1 + cm.sqrt(1 - (del_p_l/E)**2))/2) 
        u_n_l =cm.sqrt((1 + cm.sqrt(1 - (del_n_l/E)**2))/2) 
        v_p_l =cm.sqrt((1 - cm.sqrt(1 - (del_p_l/E)**2))/2) 
        v_n_l =cm.sqrt((1 - cm.sqrt(1 - (del_n_l/E)**2))/2)
        ############### Get the values for the right side of the nth segment
        K_R_P = nm.sqrt(K_fermi_R**2 + 2*m_eff*E/(e*h_bar_sq)) ### not changing these for now. They will eventually change in each segment
        K_R_N = nm.sqrt(K_fermi_R**2 - 2*m_eff*E/(e*h_bar_sq)) ### not changing these for now. They will eventually change in each segment
        [del_p_r,psi_p_r] = getDel(th,delT(n+1),K_R_P) 
        [del_n_r,psi_n_r] = getDel(th,delT(n+1),-K_R_N)
        u_p_r =cm.sqrt((1 + cm.sqrt(1 - (del_p_r/E)**2))/2) 
        u_n_r =cm.sqrt((1 + cm.sqrt(1 - (del_n_r/E)**2))/2) 
        v_p_r =cm.sqrt((1 - cm.sqrt(1 - (del_p_r/E)**2))/2) 
        v_n_r =cm.sqrt((1 - cm.sqrt(1 - (del_n_r/E)**2))/2)
        
        
        
        
    Y = nm.array([1,0,1,0])
    M = nm.zeros((4,4),dtype=complex)
    M[0] = nm.array([0,-1,0,-1]) 
    M[1] = nm.array([-1,0,1,0]) 
    M[2] = nm.array([u_p,nm.exp(-1j*psi_p)*v_p,(lam + z*2j)*u_p,(lam + z*2j)*nm.exp(-1j*psi_p)*v_p]) 
    M[3] = nm.array([nm.exp(1j*psi_n)*v_n,u_n,-(lam - z*2j)*nm.exp(1j*psi_n)*v_n,-(lam - z*2j)*u_n])    
    M = nm.transpose(M)
    X = nm.linalg.solve(M,Y)  
    A_M = X[0]*nm.conj(X[0]) 
    B_M = X[1]*nm.conj(X[1])         
    return (1 + A_M - B_M)*nm.cos(th)/2
#getSigR(-7.4344E-22,-1.5708)

def getFermi(E,V):
    return (1/(1 + nm.exp((E - e*V)/KbT)))
    

def currentIntegrand(E,V):
    return (getFermi(E,V) - getFermi(E,0))*integrate.quad(getSigT,-nm.pi/2,nm.pi/2,args=E)[0]


c2 = 0
cond = nm.zeros((E_range.size,T_range.size))
for T in T_range:
    KbT = Kb*T 
    delT = del_0*1.74*nm.sqrt(1 - T/Tc) if T < Tc else 0
    c1 = 0
    for E in E_range:
        cond[c1,c2] = integrate.quad(getSigT,-nm.pi/2,nm.pi/2,args=E)[0]
        c1 = c1 + 1
    plt.plot(E_range/del_0,cond[:,c2])
    c2 = c2 +1

####################### Write to File #######################

writeArray = nm.zeros((c1+1,c2+1))
writeArray[1:,0] = E_range
writeArray[0,1:] = T_range
writeArray[1:,1:] = cond
nm.savetxt(fileName,writeArray)
###############################################################
plt.show()
