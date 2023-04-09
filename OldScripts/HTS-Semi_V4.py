############################################################################
############## in version 4 of HTS - Semi transport conductance.  #####################
############## extending it to multiple segments  ########## Works Well - atleast upto 4 segments. ####
################# Do Not modify############################
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
Kb = 8.617E-5   #8.617E-5 eVK-1 # 1.38E-23 in    JK-1 
#E = 0.5 

count =1 
##########Simulation constants ##############
Tc = 90  # 9 for Nb and 90 for YBCO
E_fermi =2  #E in eV # in J for InGaas
K_fermi = nm.sqrt(2*m_eff*E_fermi/(e*h_bar_sq))  ### accounting for the units. Energy is in eV, so there should be a 'e' there
################ simulation constant calculations




#########Simulation parameters############
del_0 = 1.764*Kb*Tc
N_value = 3 ### Three interfaces and four segments
#T_range = nm.concatenate([nm.arange(0.01,1,0.01),nm.arange(1,90,1)])
T_range = nm.arange(1,10,5)
V_range = nm.arange(0,2E-3,1E-4)  
E_range = nm.arange(-4*del_0,4*del_0,del_0/100)  ## assume symmetric about E=0 
th_step = nm.pi/10000 
th_range =nm.arange(-nm.pi/2,nm.pi/2,th_step) 
z_value=1
fileName = "data Z = "+str(z_value)+" sWave.txt" #### File Name to save
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
    Y_matrix = nm.zeros(N_value*4 + 4,dtype = float)
    #M_matrix = spr.dok_matrix((N_value*4 + 4,N_value*4 + 4),dtype=float) # define a sparse matrix
    M_matrix = nm.zeros((N_value*4 + 4,N_value*4 + 4),dtype=complex)
    r= 0
    M_matrix[r,0] = 1
    Y_matrix[r] = 1
    r = r+1
    M_matrix[r,1] = 1
    r=  r + 1
    c2=0
    for n in range(N_value):# number of segments in the calculation
        ################ Each segment deals with 8 variables, 4 equations. So 8 cols, 4 rows ###
        ################ Get the values for the left side of the nth segment
        K_L_P = nm.sqrt(K_fermi_L**2 + 2*m_eff*E/(e*h_bar_sq)) ### not changing these for now. They will eventually change in each segment
        K_L_N = nm.sqrt(K_fermi_L**2 - 2*m_eff*E/(e*h_bar_sq))### not changing these for now. They will eventually change in each segment
        lam_l_p = K_L_P/K_fermi
        lam_l_n = K_L_N/K_fermi

        [del_p_l,psi_l_p] = getDel(th,segDel[n],K_L_P) 
        [del_n_l,psi_l_n] = getDel(th,segDel[n],-K_L_N)
        u_l_p =cm.sqrt((1 + cm.sqrt(1 - (del_p_l/E)**2))/2) 
        u_l_n =cm.sqrt((1 + cm.sqrt(1 - (del_n_l/E)**2))/2) 
        v_l_p =cm.sqrt((1 - cm.sqrt(1 - (del_p_l/E)**2))/2)*nm.exp(-1j*psi_l_p) 
        v_l_n =cm.sqrt((1 - cm.sqrt(1 - (del_n_l/E)**2))/2)*nm.exp(1j*psi_l_n)
        ############### Get the values for the right side of the nth segment
        K_R_P = nm.sqrt(K_fermi_R**2 + 2*m_eff*E/(e*h_bar_sq)) ### not changing these for now. They will eventually change in each segment
        K_R_N = nm.sqrt(K_fermi_R**2 - 2*m_eff*E/(e*h_bar_sq)) ### not changing these for now. They will eventually change in each segment
        lam_r_p = K_R_P/K_fermi
        lam_r_n = K_R_N/K_fermi
        [del_p_r,psi_r_p] = getDel(th,segDel[n+1],K_R_P) 
        [del_n_r,psi_r_n] = getDel(th,segDel[n+1],-K_R_N)
        u_r_p =cm.sqrt((1 + cm.sqrt(1 - (del_p_r/E)**2))/2) 
        u_r_n =cm.sqrt((1 + cm.sqrt(1 - (del_n_r/E)**2))/2) 
        v_r_p =cm.sqrt((1 - cm.sqrt(1 - (del_p_r/E)**2))/2)*nm.exp(-1j*psi_r_n) 
        v_r_n =cm.sqrt((1 - cm.sqrt(1 - (del_n_r/E)**2))/2)*nm.exp(1j*psi_r_p)
        M_matrix[r,c2:c2+8] =nm.array([u_l_p,v_l_n,v_l_p,u_l_n,-u_r_p,-v_r_n,-v_r_p,-u_r_n])         
        r = r+ 1
        M_matrix[r,c2:c2+8] =nm.array([v_l_p,u_l_n,u_l_p,v_l_n,-v_r_p,-u_r_n,-u_r_p,-v_r_n])         
        r = r+ 1
        M_matrix[r,c2:c2+8] =nm.array([-lam_l_p*u_l_p,lam_l_n*v_l_n,-lam_l_n*v_l_p,lam_l_p*u_l_n,(lam_r_p +2j*z_inter[n])*u_r_p, -(lam_r_n - 2j *z_inter[n])*v_r_n,(lam_r_p + 2j*z_inter[n])*v_r_p, -(lam_r_n - 2j*z_inter[n])*u_r_n])         
        r = r+ 1
        M_matrix[r,c2:c2+8] =nm.array([-lam_l_p*v_l_p,lam_l_n*u_l_n,-lam_l_n*u_l_p,lam_l_p*v_l_n,(lam_r_p +2j*z_inter[n])*v_r_p, -(lam_r_n - 2j *z_inter[n])*u_r_n,(lam_r_p + 2j*z_inter[n])*u_r_p, -(lam_r_n - 2j*z_inter[n])*v_r_n])         
        r= r+1
        c2 = c2 + 4 
        ################# end of eight equations per segment
    M_matrix[r,c2+2] = 1
    r = r+1
    M_matrix[r,c2+3] = 1
    
    X = nm.linalg.solve(M_matrix,Y_matrix)    
    A_M = float(X[c2-2]*nm.conj(X[c2-2]))# use A of the penultimate segment
    B_M = float(X[0]*nm.conj(X[0])) # use B of the first segment
    
    return (1 + A_M - B_M)*nm.cos(th)/2
#getSigR(-7.4344E-22,-1.5708)

def getFermi(E,V):
    return (1/(1 + nm.exp((E - e*V)/KbT)))
    

def currentIntegrand(E,V):
    return (getFermi(E,V) - getFermi(E,0))*integrate.quad(getSigT,-nm.pi/2,nm.pi/2,args=E)[0]


c2 = 0
cond = nm.zeros((E_range.size,T_range.size))
segDel=nm.zeros(N_value+1)
z_inter=nm.zeros(N_value)
for T in T_range:
    KbT = Kb*T 
    delT = del_0*1.74*nm.sqrt(1 - T/Tc) if T < Tc else 0 ##temperature dependent delta
    segDel[0] = 0
    segDel[1] = 0
    segDel[2] = delT
    segDel[3] = delT

    z_inter[0] = 0
    z_inter[1] = z_value
    z_inter[2] = 0

    c1 = 0
    for E in E_range:
        cond[c1,c2] = integrate.quad(getSigT,-nm.pi/2,nm.pi/2,args=E)[0]
        c1 = c1 + 1
    plt.plot(E_range/del_0,cond[:,c2])
    
    c2 = c2 +1

####################### Write to File #######################
nm.savetxt('test.csv',cond,delimiter=',',fmt='%.2f')
writeArray = nm.zeros((c1+1,c2+1))
writeArray[1:,0] = E_range
writeArray[0,1:] = T_range
writeArray[1:,1:] = cond
nm.savetxt(fileName,writeArray)
###############################################################
plt.show()
