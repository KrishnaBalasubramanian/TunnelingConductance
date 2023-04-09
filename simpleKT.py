import numpy as nm
#########Simulation parameters############
Tc = 90  # 9 for Nb and 90 for YBCO
N_value = 1 ### NUmber of interfaces - N_value + 1 is the number of segments
l_seg = 10E-9
#T_range = nm.concatenate([nm.arange(0.01,1,0.01),nm.arange(1,90,1)])
T_range = nm.arange(1,10,5)
#z_value=1
#fileName = "data Z = "+str(z_value)+" sWave.txt" #### File Name to save
E_fermi =2  #E in eV # in J for InGaas
E_range = nm.arange(-0.05,0.05,0.001)  ## assume symmetric about E=0 
#E_fermi_L =2  #E in eV # in J for InGaas
#E_fermi_R =2  #E in eV # in J for InGaas
#K_fermi_L = nm.sqrt(2*m_eff*E_fermi_L/(e*h_bar_sq)) 
#K_fermi_R = nm.sqrt(2*m_eff*E_fermi_R/(e*h_bar_sq)) 
#sig_N = 4*lam/((1 + lam)^2 + 4*z^2) 
sig_N = 1  ### for un normalized currents
c1 = 1
segCount = N_value +1 ## number of segments = number of interfaces + 1
dirName = "KTResults" #### directory to open
#####################################################################
############### define the delta of various segments ######
############## Should have N_value + 1 segments [0 - N_value] ##########
#############################################################
inDel = nm.zeros(segCount)# initialize as normal components
inDel[1] = 1 # change only those segments
############## delta Symmetry on each segment - defaulting to S- Wave
angleDependent=True
segSym = ["S-Wave"]*segCount
segSym[1] = "D-Wave" 
alpha = nm.pi/4
############### m_eff at each segment - defaulting to 1 on each segment
m_eff = 0.07
m_eff_seg = [m_eff]*segCount
############### Barrier potential at each segment - defaulting to 0
V_seg = [0]*segCount
#V_seg[1] = 1 # in eV  - change only non zero components
#####################################################################
################ define the interface impedance###############
############### SHould have N_value interfaces [0 - N_value - 1] #####
#################################################
z_inter = nm.zeros(N_value) # initialize as 0 impedance interfaces
z_inter[0] = 1 # change only what is necessary

