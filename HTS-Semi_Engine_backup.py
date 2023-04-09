
############################################################################
############## in version 6 of HTS - Semi transport conductance.  #####################
##############  Allow variations in effective mass, potential and hence K- values in each segment ####
################# Do Not modify############################
## If there are N segments, the conductance is proportional to 1 + A(penultimate seg) - B(first segment)##
##############################################################################
################# Last modified July 17, 2020. ##################
######## List of changes - moved th_range to input file - check simpleBTK.py for example
######### Added condNormalize = True/False - again check simpleBTK.py for example.
######## The results are exported as a dict to results.npy. To read it read as nm.load('results.npy',allow_pickle=True).item()
import numpy as nm
import cmath as cm
import pdb
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from math import floor
import sys
import os
import importlib
########################## import the input file ###############
if len(sys.argv) <2 :
    print("Run with the input file as arguement")
    sys.exit()
exec('from %s import *'%sys.argv[1])
#############################################################################
### global constants
m_0 = 9.10E-31 # in Kg
h = 4.135E-15  # 4.135E-15 in eV and 6.626E-34 in J
h_bar = h/(2*nm.pi)
h_bar_sq = h_bar**2 
e = 1.6E-19 
Kb = 8.617E-5  #8.617E-5 eVK-1 # 1.38E-23 in    JK-1 

########################## Derived variables ####################
del_0 = 1.764*Kb*Tc # Del at 0K
K_fermi = nm.sqrt(2*m_eff*m_0*E_fermi/(e*h_bar_sq))  ### accounting for the units. Energy is in eV, so there should be a 'e' there
#E_range = nm.arange(0.1,1,0.05)
#E_range = nm.arange(-4*del_0,4*del_0,del_0/50)  ## assume symmetric about E=0 
m_eff_seg = nm.array(m_eff_seg)*m_0 ## converting effective mass to right units 
th_seg = nm.zeros(segCount,dtype=float)
if not os.path.exists(dirName):
    os.mkdir(dirName) ### Do it very early to see if you have write permission. Otherwise why calculate

############## Define my own exponential function ## to avoid inf###
def myExp(inTerm):
    if (inTerm.real >1):
        #inTerm = -inTerm.real + 1j*inTerm.imag ## invert the real part if its greater than 1
        inTerm = 0 # just make the value capped at 1
    return nm.exp(inTerm)

def getDel(th,delT,k,sym):
    if sym == 'S-Wave':
        delTemp = delT
    elif sym == 'D-Wave':
        delTemp = delT*nm.cos(2*(nm.sign(k)*th - alpha)) # alpha is  the angle from the normal to interface
    elif sym == 'coin':
        delTemp = delT*myExp(1j*(-nm.sign(k) +1)*phi_d/2) # phi_d is difference between phi_p and phi_n
    else:
        delTemp = delT
    return cm.polar(delTemp)


def getSigT(th,E):
    Y_matrix = nm.zeros(segCount*4,dtype = float)# you have 4 variables per segment
    X= nm.zeros(segCount*4,dtype = complex)# you have 4 variables per segment
    Xsol = nm.zeros(segCount*4,dtype=complex)
    #M_matrix = spr.dok_matrix((N_value*4 + 4,N_value*4 + 4),dtype=float) # define a sparse matrix
    M_matrix = nm.zeros((segCount*4,segCount*4),dtype=complex)
    r= 0
    M_matrix[r,0] = 1
    Y_matrix[r] = 1
    r = r+1
    M_matrix[r,1] = 1
    r=  r + 1
    c2=0
    th_seg[c2] = th    # angular incidence of first interface is the input angle
    th_l = th
    for n in range(N_value):# number of Interfaces in the calculation
        ################ Each segment deals with 8 variables, 4 equations. So 8 cols, 4 rows ###
        ################ Get the values for the left side of the nth segment
        [del_p_l,psi_l_p] = getDel(th,segDel[n],1,segSym[n]) # using k=1 as ony the sign is used for now.  
        [del_n_l,psi_l_n] = getDel(th,segDel[n],-1,segSym[n])# using k= -1 as only sign is used for now. If delta also depends on the actual value of k, then there is a circular definition as k in-turn depends on the value of delta. So they must be consistent
        #K_L_P = cm.sqrt(2*m_eff_seg[n]*(E_fermi + cm.sqrt(E**2 - abs(del_p_l)**2) - V_seg[n])/(e*h_bar_sq)) 
        #K_L_N = cm.sqrt(2*m_eff_seg[n]*(E_fermi - cm.sqrt(E**2 - abs(del_p_l)**2) - V_seg[n])/(e*h_bar_sq))
        K_L_P = cm.sqrt(2*m_eff_seg[n]*(E_fermi + E + del_p_l - V_seg[n])/(e*h_bar_sq)) 
        K_L_N = cm.sqrt(2*m_eff_seg[n]*(E_fermi - E -del_p_l - V_seg[n])/(e*h_bar_sq))
        lam_l_p = K_L_P/K_fermi # 
        lam_l_n = K_L_N/K_fermi

        if (E == 0): ### to avoid divide by zero errors
            u_l_p =1
            u_l_n =1 
            v_l_p =0 
            v_l_n =0
        else:
            u_l_p =cm.sqrt((1 + cm.sqrt(1 - (del_p_l/E)**2))/2) 
            u_l_n =cm.sqrt((1 + cm.sqrt(1 - (del_n_l/E)**2))/2) 
            v_l_p =cm.sqrt((1 - cm.sqrt(1 - (del_p_l/E)**2))/2)*myExp(-1j*psi_l_p) 
            v_l_n =cm.sqrt((1 - cm.sqrt(1 - (del_n_l/E)**2))/2)*myExp(1j*psi_l_n)
       
        ############### Get the values for the right side of the nth segment
        [del_p_r,psi_r_p] = getDel(th,segDel[n+1],1,segSym[n+1]) 
        [del_n_r,psi_r_n] = getDel(th,segDel[n+1],-1,segSym[n+1])
        #K_R_P = cm.sqrt(2*m_eff_seg[n+1]*(E_fermi + cm.sqrt(E**2 - abs(del_p_r)**2) - V_seg[n+1])/(e*h_bar_sq)) 
        #K_R_N = cm.sqrt(2*m_eff_seg[n+1]*(E_fermi - cm.sqrt(E**2 - abs(del_p_r)**2) - V_seg[n+1])/(e*h_bar_sq))
        K_R_P = cm.sqrt(2*m_eff_seg[n+1]*(E_fermi + E +del_p_r - V_seg[n+1])/(e*h_bar_sq)) 
        K_R_N = cm.sqrt(2*m_eff_seg[n+1]*(E_fermi - E - del_n_r - V_seg[n+1])/(e*h_bar_sq))
        
        lam_r_p = K_R_P/K_fermi
        lam_r_n = K_R_N/K_fermi
        #### Correct for momentym conservation in the Y Axis. ####
        #
        if (E ==0) : ### to avoid divide by zero errors
            u_r_p =1
            u_r_n =1 
            v_r_p =0 
            v_r_n =0
        else:
            u_r_p =cm.sqrt((1 + cm.sqrt(1 - (del_p_r/E)**2))/2)
            u_r_n =cm.sqrt((1 + cm.sqrt(1 - (del_n_r/E)**2))/2)
            v_r_p =cm.sqrt((1 - cm.sqrt(1 - (del_p_r/E)**2))/2)*myExp(-1j*psi_r_n) 
            v_r_n =cm.sqrt((1 - cm.sqrt(1 - (del_n_r/E)**2))/2)*myExp(1j*psi_r_p)
        
        try:
            th_r = nm.arcsin(K_L_P * nm.sin(th)/K_R_P)
            zn = z_inter[n]/nm.cos(th)
        except: # zero division error when E = V. K = 0
            th_r = th 
            zn = z_inter[n]
           
        ########### Row 1 ###########
        M_matrix[r,c2] = u_l_p*myExp(1j*K_L_P*n*l_seg)
        M_matrix[r,c2+1] = v_l_n*myExp(-1j*K_L_N*n*l_seg)
        M_matrix[r,c2+2] = v_l_p*myExp(1j*K_L_N*n*l_seg)
        M_matrix[r,c2+3] = u_l_n*myExp(-1j*K_L_P*n*l_seg)
        M_matrix[r,c2+4] = -u_r_p*myExp(1j*K_R_P*n*l_seg)
        M_matrix[r,c2+5] = -v_r_n*myExp(-1j*K_R_N*n*l_seg)
        M_matrix[r,c2+6] = -v_r_p*myExp(1j*K_R_N*n*l_seg)
        M_matrix[r,c2+7] = -u_r_n*myExp(-1j*K_R_P*n*l_seg)
        r = r+ 1
                ########### Row 2 ###########
        M_matrix[r,c2] = v_l_p*myExp(1j*K_L_P*n*l_seg)
        M_matrix[r,c2+1] = u_l_n*myExp(-1j*K_L_N*n*l_seg)
        M_matrix[r,c2+2] = u_l_p*myExp(1j*K_L_N*n*l_seg)
        M_matrix[r,c2+3] = v_l_n*myExp(-1j*K_L_P*n*l_seg)
        M_matrix[r,c2+4] = -v_r_p*myExp(1j*K_R_P*n*l_seg)
        M_matrix[r,c2+5] = -u_r_n*myExp(-1j*K_R_N*n*l_seg)
        M_matrix[r,c2+6] = -u_r_p*myExp(1j*K_R_N*n*l_seg)
        M_matrix[r,c2+7] = -v_r_n*myExp(-1j*K_R_P*n*l_seg)
        r = r+1
                ########### Row 3 ###########
        M_matrix[r,c2] = -lam_l_p*u_l_p*myExp(1j*K_L_P*n*l_seg)
        M_matrix[r,c2+1] = lam_l_n*v_l_n*myExp(-1j*K_L_N*n*l_seg)
        M_matrix[r,c2+2] = -lam_l_n*v_l_p*myExp(1j*K_L_N*n*l_seg)
        M_matrix[r,c2+3] = lam_l_p*u_l_n*myExp(-1j*K_L_P*n*l_seg)
        M_matrix[r,c2+4] = (lam_r_p +2j*zn)*u_r_p*myExp(1j*K_R_P*n*l_seg)
        M_matrix[r,c2+5] = -(lam_r_n - 2j *zn)*v_r_n*myExp(-1j*K_R_N*n*l_seg)
        M_matrix[r,c2+6] = (lam_r_n + 2j*zn)*v_r_p*myExp(1j*K_R_N*n*l_seg)
        M_matrix[r,c2+7] = -(lam_r_p - 2j*zn)*u_r_n*myExp(-1j*K_R_P*n*l_seg)
        r = r + 1
                ########### Row 4 ###########
        M_matrix[r,c2] = -lam_l_p*v_l_p*myExp(1j*K_L_P*n*l_seg)
        M_matrix[r,c2+1] = lam_l_n*u_l_n*myExp(-1j*K_L_N*n*l_seg)
        M_matrix[r,c2+2] = -lam_l_n*u_l_p*myExp(1j*K_L_N*n*l_seg)
        M_matrix[r,c2+3] = lam_l_p*v_l_n*myExp(-1j*K_L_P*n*l_seg)
        M_matrix[r,c2+4] = (lam_r_p +2j*zn)*v_r_p*myExp(1j*K_R_P*n*l_seg)
        M_matrix[r,c2+5] = -(lam_r_n - 2j *zn)*u_r_n*myExp(-1j*K_R_N*n*l_seg)
        M_matrix[r,c2+6] = (lam_r_n + 2j*zn)*u_r_p*myExp(1j*K_R_N*n*l_seg)
        M_matrix[r,c2+7] = -(lam_r_p - 2j*zn)*v_r_n*myExp(-1j*K_R_P*n*l_seg)
        r = r+1
        c2 = c2 + 4 
        ################# end of eight equations per segment
    
    #pdb.set_trace()
    M_matrix[r,c2+2] = 1
    r = r+1
    M_matrix[r,c2+3] = 1
    
    try:
        X = nm.linalg.solve(M_matrix,Y_matrix)
    except nm.linalg.LinAlgError as err:
        print('Singular Matrix Error at Energy: ',E)
        print('See Singular Error.csv file')
        nm.savetxt(dirName + '/singularError.csv',M_matrix,delimiter=',',fmt='%.2f')
        #pdb.set_trace()
        X = nm.zeros(segCount*4)
        X[3] = 1 ## assume that the electron is fully scattered back. 
    XSol = X * nm.conj(X) ## get absolute values of the solution
    
    ##convert solutions to probability currents
    sols = nm.zeros((segCount,4),dtype=float)
    sols[:,0] = XSol[::4]*(abs(u_r_p)**2 - abs(v_r_p)**2) # C's
    sols[:,1] = XSol[1::4]*(abs(u_r_p)**2 - abs(v_r_p)**2) # D's
    sols[:,2] = XSol[2::4] # A's
    sols[:,3] = XSol[3::4] # B's
    #pdb.set_trace()
    return sols
   

def getSigN():
##    if (nm.cos(th) == 0):
##        return 0
##    else:
##        [del_p_l,psi_l_p] = getDel(th,segDel[n],1,segSym[n]) # using k=1 as ony the sign is used for now.  
##        [del_n_l,psi_l_n] = getDel(th,segDel[n],-1,segSym[n])# using k= -1 as only sign is used for now. If delta also depends on the actual value of k, then there is a circular definition as k in-turn depends on the value of delta. So they must be consistent
##        K_L_P = abs(nm.sqrt(2*m_eff_seg[n]*(E_fermi + cm.sqrt(E**2 - abs(del_p_l)**2) - V_seg[n])/(e*h_bar_sq)) )
##        ############### Get the values for the right side of the nth segment
##        [del_p_r,psi_r_p] = getDel(th,segDel[n+1],1,segSym[n+1]) 
##        [del_n_r,psi_r_n] = getDel(th,segDel[n+1],-1,segSym[n+1])
##        K_R_P = abs(nm.sqrt(2*m_eff_seg[n+1]*(E_fermi + cm.sqrt(E**2 - abs(del_p_r)**2) - V_seg[n+1])/(e*h_bar_sq)) )
##        lam_0 = K_R_P/K_L_P
##        th_next = nm.arcsin(K_L_P * nm.sin(th)/K_R_P)
##        lam = lam_0*nm.cos(th_next)/nm.cos(th)
##        pdb.set_trace()
    retVal = 1
    def integrand(th,z):
        return nm.cos(th)/(1+ z**2)
    for z in z_inter:
        retVal = retVal*integrate.quad(integrand,-nm.pi/2,nm.pi/2,args=z)[0]
    #pdb.set_trace()
    return retVal
    
    


c2 = 0
fullSol = nm.zeros((E_range.size,T_range.size,segCount,4))
cond = nm.zeros((E_range.size,T_range.size),dtype=float)
segDel=nm.zeros(segCount)
fig0 = plt.figure(0)
ax0= fig0.add_subplot(1,1,1)
for T in T_range:
    KbT = Kb*T 
    delT = del_0*nm.sqrt(1 - T/Tc) if T < Tc else 0 ##temperature dependent delta
    segDel = inDel * delT
    c1 = 0
    for E in E_range:
        
        d1=0
        if angleDependent:
            resMat = nm.zeros((segCount,4,th_range.size),dtype=float)
            for th in th_range: ## calculate the scattering parameters for each angle
                resMat[:,:,d1] = getSigT(th,E)*nm.cos(th) # First Col of resMat is segments, Second Column is for 4  variables ABCD, third for angle
                d1 = d1+ 1
        
            for d1 in range(segCount): # for each variable conduct the angle integration
                fullSol[c1,c2,d1,0] = integrate.trapz(resMat[d1,0,:],th_range)
                fullSol[c1,c2,d1,1] = integrate.trapz(resMat[d1,1,:],th_range)
                fullSol[c1,c2,d1,2] = integrate.trapz(resMat[d1,2,:],th_range)
                fullSol[c1,c2,d1,3] = integrate.trapz(resMat[d1,3,:],th_range)
        else: ### simple isotropic case. when delta is angle independent
            fullSol[c1,c2,:,:] = getSigT(0,E)*2 # 2 comes due to constant integration over -pi/2 to pi/2
            
        #if (fullSol[c1,c2,:,:] > 10).any():
        #    pdb.set_trace()
        if condNormalize:
            cond[c1,c2] = (2 + fullSol[c1,c2,N_value-1,2] - fullSol[c1,c2,0,3])/getSigN()
        else: 
            cond[c1,c2] = (2 + fullSol[c1,c2,N_value-1,2] - fullSol[c1,c2,0,3])
        
        c1 = c1 + 1
    ####################################################
    ###### Plot eventual normalized conductance vs energy #######################
    ######################################################
    ax0.plot(E_fermi + E_range,cond[:,c2],label=str(T) + "K")
    c2 = c2 +1
ax0.set_ylabel('Normalized Conductance')
ax0.set_xlabel('Energy (eV)')
fig0.savefig(dirName+"/normConductance.png")
####################### Write to File #######################

#writeArray = nm.zeros((c1+1,c2+1))
#writeArray[1:,0] = E_range
#writeArray[0,1:] = T_range
#writeArray[1:,1:] = cond
#nm.savetxt('cresults.csv',fullSol[:,0,:,0]/2)
results = {'fullSol':fullSol,'cond':cond,'E_range':E_range,'T_range':T_range}
nm.save(dirName + "/results",results)
###############################################################

#################################################
##### Plot the segment wise transmission parameters
######################################
fullSol = nm.clip(fullSol,a_min = 0, a_max=2)
ticPos = range(0,E_range.size,floor(E_range.size/5))
tics = []
for en in ticPos:
    tics.append(round(E_fermi + E_range[en],3))
fig1 = plt.figure(1)
ax1 =fig1.add_subplot(2,2,1)
ax1.set_title('Electron transmission')
im0= ax1.imshow(fullSol[:,0,:,0]/2,aspect='auto')
ax1.set_yticks(ticPos)
ax1.set_yticklabels(tics)
ax1.set_ylabel('Energy')
ax1.set_xlabel('Segment')
cbar1= fig1.colorbar(im0,ax=ax1,extend='both',shrink=0.8) 

ax2 =fig1.add_subplot(2,2,2)
im1= ax2.imshow(fullSol[:,0,:,1]/2,aspect='auto')
ax2.set_title('Transmit to Hole-Like')
ax2.set_yticks(ticPos)
ax2.set_yticklabels(tics)
ax2.set_ylabel('Energy')
ax2.set_xlabel('Segment')
cbar1 = fig1.colorbar(im1,ax=ax2,extend='both',shrink=0.8) 

ax3 =fig1.add_subplot(2,2,3)
im2 = ax3.imshow(fullSol[:,0,:,2]/2,aspect='auto')
ax3.set_title('Andreev Reflection')
ax3.set_yticks(ticPos)
ax3.set_yticklabels(tics)
ax3.set_ylabel('Energy')
ax3.set_xlabel('Segment')
cbar2 = fig1.colorbar(im2,ax=ax3,extend='both',shrink=0.8) 

ax4 =fig1.add_subplot(2,2,4)
im3 = ax4.imshow(fullSol[:,0,:,3]/2,aspect='auto')
ax4.set_title('Electron Reflection')
ax4.set_yticks(ticPos)
ax4.set_yticklabels(tics)
ax4.set_ylabel('Energy')
ax4.set_xlabel('Segment')
cbar3 = fig1.colorbar(im3,ax=ax4,extend='both',shrink=0.8) 

fig1.tight_layout()
fig1.savefig(dirName+"/parameters.png")
####################################################
###### Plot Simulation set up #######################
######################################################

fig2 = plt.figure(2)
fig2.subplots_adjust(hspace=1)

ax5 = fig2.add_subplot(3,1,1)
ax5.bar(range(segCount),inDel,label="delta",width=1,align='edge')
ax5.set_ylabel('Delta')

ax6 = fig2.add_subplot(3,1,2,sharex=ax5)
ax6.bar(range(1,segCount),z_inter,width=0.1)
ax6.set_ylabel('Impedance')
                   
ax7 = fig2.add_subplot(3,1,3,sharex=ax5)
ax7.bar(range(segCount),V_seg,width=1,align='edge')
ax7.set_ylabel('Potential')
fig2.savefig(dirName+"/conditions.png")

fig3 = plt.figure(3)
ax8 = fig3.add_subplot(1,1,1)
ax8.plot(E_fermi+E_range,fullSol[:,0,N_value,0]/2,label='C',color='b')
ax8.plot(E_fermi+E_range,fullSol[:,0,N_value,1]/2,label='D',color='y')
ax8.plot(E_fermi+E_range,fullSol[:,0,N_value-1,2]/2,label='A',color='g')
ax8.plot(E_fermi+E_range,fullSol[:,0,0,3]/2,label='B',color = 'r')
ax8.set_ylim(0,1.5)
ax8.legend()
ax8.set_xlabel('Energy (eV)')
ax8.set_ylabel('Scattering Amplitude')
fig3.savefig(dirName+"/2DResults.png")

plt.show()

