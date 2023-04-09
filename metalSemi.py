import numpy as nm
#########Simulation parameters############
Tc = 90  # 9 for Nb and 90 for YBCO
N_value = 2 ### NUmber of interfaces - N_value + 1 is the number of segments
l_seg = 10E-9 ### Length of each segment
segCount = N_value +1 ## number of segments = number of interfaces + 1
#T_range = nm.concatenate([nm.arange(0.01,1,0.01),nm.arange(1,90,1)])
T_range = nm.arange(1,2,5)

E_fermi =2  #E in eV # in J for InGaas
E_range = nm.arange(0,10,0.1)  ## assume symmetric about E=0 
dirName = "MetalSemiResults" #### directory to open
#####################################################################
############### define the delta of various segments ######
############## Should have N_value + 1 segments [0 - N_value] ##########
#############################################################
inDel = nm.zeros(segCount)# initialize as normal components
#inDel[1] = 1 # change only those segments
############## delta Symmetry on each segment - defaulting to S- Wave
angleDependent=False
segSym = ["S-Wave"]*segCount
alpha = 0
############### m_eff at each segment - defaulting to 1 on each segment
m_eff = 0.07
m_eff_seg = [m_eff]*segCount
############### Barrier potential at each segment - defaulting to 0
V_seg = [0]*segCount
V_seg[0] = 0 # in eV  - change only non zero components
V_seg[1] = 4 # in eV  - change only non zero components
V_seg[2] = 0 # in eV  - change only non zero components

#####################################################################
################ define the interface impedance###############
############### SHould have N_value interfaces [0 - N_value - 1] #####
#################################################
z_inter = nm.zeros(N_value) # initialize as 0 impedance interfaces
z_inter[0] = 0 # change only what is necessary
V_range = nm.arange(0, 4,0.5)
IVCalculate=True
