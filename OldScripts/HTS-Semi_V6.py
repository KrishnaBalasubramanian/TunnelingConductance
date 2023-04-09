############################################################################
############## in version 6 of HTS - Semi transport conductance.  #####################
##############  Allow variations in effective mass, potential and hence K- values in each segment ####
################# Do Not modify############################
## If there are N segments, the conductance is proportional to 1 + A(penultimate seg) - B(first segment)##
##############################################################################
import numpy as nm
import cmath as cm
import pdb
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy.sparse as spr
import sys
import os
### global constants
m_0 = 9.10E-31 # in Kg
m_eff = 0.07*m_0
h = 4.135E-15  # 4.135E-15 in eV and 6.626E-34 in J
h_bar = h/(2*nm.pi)
h_bar_sq = h_bar**2 
e = 1.6E-19 
Kb = 8.617E-5  #8.617E-5 eVK-1 # 1.38E-23 in    JK-1 



#########Simulation parameters############
Tc = 90  # 9 for Nb and 90 for YBCO
del_0 = 1.764*Kb*Tc # Del at 0K
N_value = 1 ### NUmber of interfaces - N_value + 1 is the number of segments
#T_range = nm.concatenate([nm.arange(0.01,1,0.01),nm.arange(1,90,1)])
T_range = nm.arange(1,10,5)
V_range = nm.arange(0,2E-3,1E-4)  
E_range = nm.arange(-4*del_0,4*del_0,del_0/50)  ## assume symmetric about E=0 
th_step = nm.pi/50
th_range =nm.arange(-nm.pi/2,nm.pi/2,th_step) 
#z_value=1
#fileName = "data Z = "+str(z_value)+" sWave.txt" #### File Name to save
E_fermi =2  #E in eV # in J for InGaas
K_fermi = nm.sqrt(2*m_eff*E_fermi/(e*h_bar_sq))  ### accounting for the units. Energy is in eV, so there should be a 'e' there
#E_fermi_L =2  #E in eV # in J for InGaas
#E_fermi_R =2  #E in eV # in J for InGaas
#K_fermi_L = nm.sqrt(2*m_eff*E_fermi_L/(e*h_bar_sq)) 
#K_fermi_R = nm.sqrt(2*m_eff*E_fermi_R/(e*h_bar_sq)) 
#sig_N = 4*lam/((1 + lam)^2 + 4*z^2) 
sig_N = 1  ### for un normalized currents
c1 = 1
segCount = N_value +1 ## number of segments = number of interfaces + 1
dirName = "calcResults" #### directory to open
alpha = nm.pi/4
#####################################################################
############### define the delta of various segments ######
############## Should have N_value + 1 segments [0 - N_value] ##########
#############################################################
inDel = nm.zeros(segCount)# initialize as normal components
inDel[1] = 1 # change only those segments
############## delta Symmetry on each segment - defaulting to S- Wave
segSym = ["S-Wave"]*segCount
segSym[1] = "D-Wave" 
############### m_eff at each segment - defaulting to 1 on each segment
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


if not os.path.exists(dirName):
    os.mkdir(dirName) ### Do it very early to see if you have write permission. Otherwise why calculate

def getDel(th,delT,k,sym):
    if sym == 'S-Wave':
        delTemp = delT
    elif sym == 'D-Wave':
        delTemp = delT*nm.cos(2*(nm.sign(k)*th - alpha))
    elif sym == 'coin':
        delTemp = delT*nm.sign(k)
    else:
        delTemp = delT
    return cm.polar(delTemp)


def getSigT(th,E):
    Y_matrix = nm.zeros(segCount*4,dtype = float)# you have 4 variables per segment
    #M_matrix = spr.dok_matrix((N_value*4 + 4,N_value*4 + 4),dtype=float) # define a sparse matrix
    M_matrix = nm.zeros((segCount*4,segCount*4),dtype=complex)
    r= 0
    M_matrix[r,0] = 1
    Y_matrix[r] = 1
    r = r+1
    M_matrix[r,1] = 1
    r=  r + 1
    c2=0
    for n in range(N_value):# number of Interfaces in the calculation
        ################ Each segment deals with 8 variables, 4 equations. So 8 cols, 4 rows ###
        ################ Get the values for the left side of the nth segment
        K_L_P = nm.sqrt(2*m_eff_seg[n]*(E_fermi + E - V_seg[n])/(e*h_bar_sq)) ### not changing these for now. They will eventually change in each segment
        K_L_N = nm.sqrt(2*m_eff_seg[n]*(E_fermi - E - V_seg[n])/(e*h_bar_sq))### not changing these for now. They will eventually change in each segment
        lam_l_p = K_L_P/K_fermi
        lam_l_n = K_L_N/K_fermi

        [del_p_l,psi_l_p] = getDel(th,segDel[n],K_L_P,segSym[n]) 
        [del_n_l,psi_l_n] = getDel(th,segDel[n],-K_L_N,segSym[n])
        u_l_p =cm.sqrt((1 + cm.sqrt(1 - (del_p_l/E)**2))/2) 
        u_l_n =cm.sqrt((1 + cm.sqrt(1 - (del_n_l/E)**2))/2) 
        v_l_p =cm.sqrt((1 - cm.sqrt(1 - (del_p_l/E)**2))/2)*nm.exp(-1j*psi_l_p) 
        v_l_n =cm.sqrt((1 - cm.sqrt(1 - (del_n_l/E)**2))/2)*nm.exp(1j*psi_l_n)
        ############### Get the values for the right side of the nth segment
        K_R_P = nm.sqrt(2*m_eff_seg[n+1]*(E_fermi + E - V_seg[n+1])/(e*h_bar_sq)) ### not changing these for now. They will eventually change in each segment
        K_R_N = nm.sqrt(2*m_eff_seg[n+1]*(E_fermi - E - V_seg[n+1])/(e*h_bar_sq))### including segment related meff and V
        lam_r_p = K_R_P/K_fermi
        lam_r_n = K_R_N/K_fermi
        [del_p_r,psi_r_p] = getDel(th,segDel[n+1],K_R_P,segSym[n+1]) 
        [del_n_r,psi_r_n] = getDel(th,segDel[n+1],-K_R_N,segSym[n+1])
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
    nm.savetxt('test.csv',M_matrix,delimiter=',',fmt='%.2f')
    X = nm.linalg.solve(M_matrix,Y_matrix)
    X = X * nm.conj(X) ## get absolute values of the solution
    ##convert solutions to probability currents
    sols = nm.zeros((segCount,4),dtype=float)
    sols[:,0] = X[::4]*(abs(u_r_p)**2 - abs(v_r_p)**2) # C's
    sols[:,1] = X[1::4]*(abs(u_r_p)**2 - abs(v_r_p)**2) # D's
    sols[:,2] = X[2::4] # A's
    sols[:,3] = X[3::4] # B's
    
    return sols
    #return (1 + A_M - B_M)*nm.cos(th)/2
#getSigR(-7.4344E-22,-1.5708)

def getFermi(E,V):
    return (1/(1 + nm.exp((E - e*V)/KbT)))
    

def currentIntegrand(E,V):
    return (getFermi(E,V) - getFermi(E,0))*integrate.quad(getSigT,-nm.pi/2,nm.pi/2,args=E)[0]


c2 = 0
fullSol = nm.zeros((E_range.size,T_range.size,segCount,4))
cond = nm.zeros((E_range.size,T_range.size),dtype=float)
segDel=nm.zeros(segCount)
for T in T_range:
    KbT = Kb*T 
    delT = del_0*1.74*nm.sqrt(1 - T/Tc) if T < Tc else 0 ##temperature dependent delta
    segDel = inDel * delT
    c1 = 0
    for E in E_range:
        resMat = nm.zeros((segCount,4,th_range.size),dtype=float)
        soltn = nm.zeros(segCount*4,dtype=float)
        d1=0
        for th in th_range: ## calculate the scattering parameters for each angle
            resMat[:,:,d1] = getSigT(th,E)*nm.cos(th) # account for the angular incidence
            d1 = d1+ 1
        for d1 in range(segCount): # for each variable conduct the angle integration
            fullSol[c1,c2,d1,0] = integrate.trapz(resMat[d1,0,:],th_range)
            fullSol[c1,c2,d1,1] = integrate.trapz(resMat[d1,1,:],th_range)
            fullSol[c1,c2,d1,2] = integrate.trapz(resMat[d1,2,:],th_range)
            fullSol[c1,c2,d1,3] = integrate.trapz(resMat[d1,3,:],th_range)
            cond[c1,c2] = (2 + fullSol[c1,c2,N_value-1,2] - fullSol[c1,c2,0,3])/2 
        #pdb.set_trace()
        c1 = c1 + 1
    # plt.plot(E_range/del_0,cond[:,c2])
    
    c2 = c2 +1

####################### Write to File #######################

#writeArray = nm.zeros((c1+1,c2+1))
#writeArray[1:,0] = E_range
#writeArray[0,1:] = T_range
#writeArray[1:,1:] = cond
nm.save(dirName + "/fullSol",fullSol)
nm.save(dirName + "/cond",fullSol)
###############################################################

#################################################
##### Plot the segment wise transmission parameters
######################################

fig0 = plt.figure(0)
ax0 =fig0.add_subplot(2,2,1)
ax0.imshow(fullSol[:,0,:,0],aspect='auto')
ax0.set_title('C vs Segment')
ax1 =fig0.add_subplot(2,2,2)
ax1.imshow(fullSol[:,0,:,1],aspect='auto')
ax1.set_title('D vs Segment')
ax2 =fig0.add_subplot(2,2,3)
ax2.imshow(fullSol[:,0,:,2],aspect='auto')
ax2.set_title('A vs Segment')
ax3 =fig0.add_subplot(2,2,4)
ax3.imshow(fullSol[:,0,:,3],aspect='auto')
ax3.set_title('B vs Segment')
fig0.savefig(dirName+"/parameters.png")
####################################################
###### Plot eventual normalized conductance vs energy #######################
######################################################
fig1 = plt.figure(1)
ax4= fig1.add_subplot(1,1,1)
c1 = 0
for T in T_range: 
    ax4.plot(E_range/del_0,cond[:,c1],label=str(T)+'K')
ax4.set_ylabel('Normalized Conductance')
ax4.set_xlabel('Energy (eV/del(0)')
fig1.savefig(dirName+"/normConductance.png")

fig3 = plt.figure(2)
fig3.subplots_adjust(hspace=1)

ax5 = fig3.add_subplot(3,1,1)
ax5.bar(range(segCount),inDel,label="delta",width=1,align='edge')
ax5.set_ylabel('Delta')

ax6 = fig3.add_subplot(3,1,2,sharex=ax5)
ax6.bar(range(1,segCount),z_inter,width=0.1)
ax6.set_ylabel('Impedance')
                   
ax7 = fig3.add_subplot(3,1,3,sharex=ax5)
ax7.bar(range(segCount),V_seg,width=1,align='edge')
ax7.set_ylabel('Potential')
fig3.savefig(dirName+"/conditions.png")


plt.show()

