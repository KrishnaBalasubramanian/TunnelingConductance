############################################################################
############## in version 1 of HTS - Semi transport conductance. Works Well Do not modify #####################
##############################################################################
import numpy as nm
import cmath as cm
import pdb
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys
### global constants
m_0 = 9.10E-31 # in Kg
m_eff = 0.07*m_0
h = 4.135E-15  # planks in eV
#h = 6.626E-34
h_bar = h/(2*nm.pi)
h_bar_sq = h_bar**2 
e = 1.6E-19 
Kb = 8.617E-5  #8.617E-5 eVK-1 # 1.38E-23 in    JK-1 
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
T_range = nm.arange(0,10,20)
V_range = nm.arange(0,2E-3,1E-4)  
E_range = nm.arange(-4*del_0,4*del_0,del_0/100)  ## assume symmetric about E=0 
th_step = nm.pi/10000 
th_range =nm.arange(-nm.pi/2,nm.pi/2,th_step) 
z_0=5
fileName = "data Z = "+str(z_0)+" sWave.txt" #### File Name to save
E_fermi_L =2  #E in eV # in J for InGaas
E_fermi_R =2  #E in eV # in J for InGaas
K_fermi_L = nm.sqrt(2*m_eff*E_fermi_L/(e*h_bar_sq)) 
K_fermi_R = nm.sqrt(2*m_eff*E_fermi_R/(e*h_bar_sq)) 
lam = K_fermi_R/K_fermi_L 
#sig_N = 4*lam/((1 + lam)^2 + 4*z^2) 
sig_N = 1  ### for un normalized currents
c1 = 1
phi_d = 0.5*nm.pi
alpha = 0
def getDel(th,delT,k):
    #delTemp = delT
    #delTemp = delT*nm.exp(1j*(-nm.sign(k) +1)*phi_d/2)
    delTemp = delT*nm.cos(2*(th*nm.sign(k) - alpha))
    return cm.polar(delTemp)

def getSigT(th,E):
    K_L_P = cm.sqrt(K_fermi_L**2 + 2*m_eff*E/(e*h_bar_sq)) 
    K_L_N = cm.sqrt(K_fermi_L**2 - 2*m_eff*E/(e*h_bar_sq))
    [del_p,psi_p] = getDel(th,delT,K_L_P) 
    [del_n,psi_n] = getDel(th,delT,-K_L_N)
    Omeg_P = cm.sqrt(E**2 - del_p**2)
    Omeg_N = cm.sqrt(E**2 - del_n**2)
    K_R_P = cm.sqrt(K_fermi_L**2 + 2*m_eff*Omeg_P/(e*h_bar_sq)) 
    K_R_N = cm.sqrt(K_fermi_L**2 - 2*m_eff*Omeg_N/(e*h_bar_sq))
    lam_0 = K_fermi_R/K_fermi_L
    z = z_0/nm.cos(th)
    u_p =cm.sqrt((1 + cm.sqrt(1 - (del_p/E)**2))/2) 
    u_n =cm.sqrt((1 + cm.sqrt(1 - (del_n/E)**2))/2) 
    v_p =cm.sqrt((1 - cm.sqrt(1 - (del_p/E)**2))/2) 
    v_n =cm.sqrt((1 - cm.sqrt(1 - (del_n/E)**2))/2)
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
    return (1 + A_M - B_M)*nm.cos(th)
#getSigR(-7.4344E-22,-1.5708)
def getSigN(Z):
    return 2/(1 + Z**2)    
def getFermi(E,V):
    return (1/(1 + nm.exp((E - e*V)/KbT)))
    

def currentIntegrand(E,V):
    return (getFermi(E,V) - getFermi(E,0))*integrate.quad(getSigT,-nm.pi/2,nm.pi/2,args=E)[0]


c2 = 0
condT = nm.zeros((E_range.size,T_range.size))
condR = nm.zeros((E_range.size,T_range.size))
for T in T_range:
    KbT = Kb*T 
    delT = del_0*nm.sqrt(1 - T/Tc) if T < Tc else 0
    c1 = 0
    for E in E_range:
        condT[c1,c2] = integrate.quad(getSigT,-nm.pi/2,nm.pi/2,args=E)[0]/getSigN(z_0)
        condR[c1,c2] = getSigT(0,E)*2/getSigN(z_0)
        c1 = c1 + 1
    plt.plot(E_range/del_0,condT[:,c2]) ## Chose between condR and condT
    c2 = c2 +1

####################### Write to File #######################

writeArray = nm.zeros((c1+1,c2+1))
writeArray[1:,0] = E_range
writeArray[0,1:] = T_range
writeArray[1:,1:] = condT
nm.savetxt(fileName,writeArray)
###############################################################
plt.show()
