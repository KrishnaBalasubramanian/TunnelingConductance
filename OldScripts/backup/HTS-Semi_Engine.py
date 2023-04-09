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
th_step = nm.pi/50
th_range =nm.arange(-nm.pi/2,nm.pi/2,th_step) 
m_eff_seg = nm.array(m_eff_seg)*m_0 ## converting effective mass to right units 
th_seg = nm.zeros(segCount,dtype=float)
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
    th_seg[c2] = th # angular incidence of first interface is the input angle
    for n in range(N_value):# number of Interfaces in the calculation
        ################ Each segment deals with 8 variables, 4 equations. So 8 cols, 4 rows ###
        ################ Get the values for the left side of the nth segment
        [del_p_l,psi_l_p] = getDel(th,segDel[n],1,segSym[n]) # using k=1 as ony the sign is used for now.  
        [del_n_l,psi_l_n] = getDel(th,segDel[n],-1,segSym[n])# using k= -1 as only sign is used for now. If delta also depends on the actual value of k, then there is a circular definition as k in-turn depends on the value of delta. So they must be consistent
        K_L_P = nm.sqrt(2*m_eff_seg[n]*(E_fermi + cm.sqrt(E**2 - abs(del_p_l)**2) - V_seg[n])/(e*h_bar_sq)) 
        K_L_N = nm.sqrt(2*m_eff_seg[n]*(E_fermi - cm.sqrt(E**2 - abs(del_p_l)**2) - V_seg[n])/(e*h_bar_sq))
        lam_l_p = K_L_P/K_fermi # 
        lam_l_n = K_L_N/K_fermi

        u_l_p =cm.sqrt((1 + cm.sqrt(1 - (del_p_l/E)**2))/2) 
        u_l_n =cm.sqrt((1 + cm.sqrt(1 - (del_n_l/E)**2))/2) 
        v_l_p =cm.sqrt((1 - cm.sqrt(1 - (del_p_l/E)**2))/2)*nm.exp(-1j*psi_l_p) *nm.exp(1j*K_L_P*n*l_seg)
        v_l_n =cm.sqrt((1 - cm.sqrt(1 - (del_n_l/E)**2))/2)*nm.exp(1j*psi_l_n)*nm.exp(-1j*K_L_N*n*l_seg)
        ############### Get the values for the right side of the nth segment
        [del_p_r,psi_r_p] = getDel(th,segDel[n+1],1,segSym[n+1]) 
        [del_n_r,psi_r_n] = getDel(th,segDel[n+1],-1,segSym[n+1])
        K_R_P = nm.sqrt(2*m_eff_seg[n+1]*(E_fermi + cm.sqrt(E**2 - abs(del_p_r)**2) - V_seg[n+1])/(e*h_bar_sq)) 
        K_R_N = nm.sqrt(2*m_eff_seg[n+1]*(E_fermi - cm.sqrt(E**2 - abs(del_p_r)**2) - V_seg[n+1])/(e*h_bar_sq))
        lam_r_p = K_R_P/K_fermi
        lam_r_n = K_R_N/K_fermi
        #### Correct for momentym conservation in the Y Axis. ####
        #th = nm.arcsin(K_L_P * nm.sin(th)/K_R_P)

        u_r_p =cm.sqrt((1 + cm.sqrt(1 - (del_p_r/E)**2))/2)
        u_r_n =cm.sqrt((1 + cm.sqrt(1 - (del_n_r/E)**2))/2)
        v_r_p =cm.sqrt((1 - cm.sqrt(1 - (del_p_r/E)**2))/2)*nm.exp(-1j*psi_r_n) 
        v_r_n =cm.sqrt((1 - cm.sqrt(1 - (del_n_r/E)**2))/2)*nm.exp(1j*psi_r_p)
        ########### Row 1 ###########
        M_matrix[r,c2] = u_l_p*nm.exp(1j*K_L_P*n*l_seg)
        M_matrix[r,c2+1] = v_l_n*nm.exp(-1j*K_L_N*n*l_seg)
        M_matrix[r,c2+2] = v_l_p*nm.exp(1j*K_L_N*n*l_seg)
        M_matrix[r,c2+3] = u_l_n*nm.exp(-1j*K_L_P*n*l_seg)
        M_matrix[r,c2+4] = -u_r_p*nm.exp(1j*K_R_P*n*l_seg)
        M_matrix[r,c2+5] = -v_r_n*nm.exp(-1j*K_R_N*n*l_seg)
        M_matrix[r,c2+6] = -v_r_p*nm.exp(1j*K_R_N*n*l_seg)
        M_matrix[r,c2+7] = -u_r_n*nm.exp(-1j*K_R_P*n*l_seg)
        r = r+ 1
        M_matrix[r,c2] = v_l_p*nm.exp(1j*K_L_P*n*l_seg)
        M_matrix[r,c2+1] = u_l_n*nm.exp(-1j*K_L_N*n*l_seg)
        M_matrix[r,c2+2] = u_l_p*nm.exp(1j*K_L_N*n*l_seg)
        M_matrix[r,c2+3] = v_l_n*nm.exp(-1j*K_L_P*n*l_seg)
        M_matrix[r,c2+4] = -v_r_p*nm.exp(1j*K_R_P*n*l_seg)
        M_matrix[r,c2+5] = -u_r_n*nm.exp(-1j*K_R_N*n*l_seg)
        M_matrix[r,c2+6] = -u_r_p*nm.exp(1j*K_R_N*n*l_seg)
        M_matrix[r,c2+7] = -v_r_n*nm.exp(-1j*K_R_P*n*l_seg)
        r = r+1
        M_matrix[r,c2] = -lam_l_p*u_l_p*nm.exp(1j*K_L_P*n*l_seg)
        M_matrix[r,c2+1] = lam_l_n*v_l_n*nm.exp(-1j*K_L_N*n*l_seg)
        M_matrix[r,c2+2] = -lam_l_n*v_l_p*nm.exp(1j*K_L_N*n*l_seg)
        M_matrix[r,c2+3] = lam_l_p*u_l_n*nm.exp(-1j*K_L_P*n*l_seg)
        M_matrix[r,c2+4] = (lam_r_p +2j*z_inter[n])*u_r_p*nm.exp(1j*K_R_P*n*l_seg)
        M_matrix[r,c2+5] = -(lam_r_n - 2j *z_inter[n])*v_r_n*nm.exp(-1j*K_R_N*n*l_seg)
        M_matrix[r,c2+6] = (lam_r_n + 2j*z_inter[n])*v_r_p*nm.exp(1j*K_R_N*n*l_seg)
        M_matrix[r,c2+7] = -(lam_r_p - 2j*z_inter[n])*u_r_n*nm.exp(-1j*K_R_P*n*l_seg)
        
        r = r + 1
        M_matrix[r,c2] = -lam_l_p*v_l_p*nm.exp(1j*K_L_P*n*l_seg)
        M_matrix[r,c2+1] = lam_l_n*u_l_n*nm.exp(-1j*K_L_N*n*l_seg)
        M_matrix[r,c2+2] = -lam_l_n*u_l_p*nm.exp(1j*K_L_N*n*l_seg)
        M_matrix[r,c2+3] = lam_l_p*v_l_n*nm.exp(-1j*K_L_P*n*l_seg)
        M_matrix[r,c2+4] = (lam_r_p +2j*z_inter[n])*v_r_p*nm.exp(1j*K_R_P*n*l_seg)
        M_matrix[r,c2+5] = -(lam_r_n - 2j *z_inter[n])*u_r_n*nm.exp(-1j*K_R_N*n*l_seg)
        M_matrix[r,c2+6] = (lam_r_n + 2j*z_inter[n])*u_r_p*nm.exp(1j*K_R_N*n*l_seg)
        M_matrix[r,c2+7] = -(lam_r_p - 2j*z_inter[n])*v_r_n*nm.exp(-1j*K_R_P*n*l_seg)
        r = r+1
        c2 = c2 + 4 
        ################# end of eight equations per segment
    
    #pdb.set_trace()
    M_matrix[r,c2+2] = 1
    r = r+1
    M_matrix[r,c2+3] = 1
    nm.savetxt('test.csv',M_matrix,delimiter=',',fmt='%.2f')
    try:
        X = nm.linalg.solve(M_matrix,Y_matrix)
    except nm.linalg.LinAlgError as err:
        print('Singular Matrix Error: ',err)
        print('See test.csv file')
        nm.savetxt('singleError.csv',M_matrix,delimiter=',',fmt='%.2f')
        X = nm.zeros(segCount*4)

    X = X * nm.conj(X) ## get absolute values of the solution
    ##convert solutions to probability currents
    sols = nm.zeros((segCount,4),dtype=float)
    sols[:,0] = X[::4]*(abs(u_r_p)**2 - abs(v_r_p)**2) # C's
    sols[:,1] = X[1::4]*(abs(u_r_p)**2 - abs(v_r_p)**2) # D's
    sols[:,2] = X[2::4] # A's
    sols[:,3] = X[3::4] # B's
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
         
        cond[c1,c2] = (2 + fullSol[c1,c2,N_value-1,2] - fullSol[c1,c2,0,3])/getSigN()
        
        c1 = c1 + 1
    ####################################################
    ###### Plot eventual normalized conductance vs energy #######################
    ######################################################
    ax0.plot(E_range/del_0,cond[:,c2],label=str(T) + "K")
    c2 = c2 +1
ax0.set_ylabel('Normalized Conductance')
ax0.set_xlabel('Energy (eV/del(0)')
fig0.savefig(dirName+"/normConductance.png")
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

fig1 = plt.figure(1)
ax1 =fig1.add_subplot(2,2,1)
ax1.set_title('C vs Segment')
im0= ax1.imshow(fullSol[:,0,:,0],aspect='auto')
cbar1= fig1.colorbar(im0,ax=ax1,extend='both',shrink=0.8) 

ax2 =fig1.add_subplot(2,2,2)
im1= ax2.imshow(fullSol[:,0,:,1],aspect='auto')
ax2.set_title('D vs Segment')
cbar1 = fig1.colorbar(im1,ax=ax2,extend='both',shrink=0.8) 

ax3 =fig1.add_subplot(2,2,3)
im2 = ax3.imshow(fullSol[:,0,:,2],aspect='auto')
ax3.set_title('A vs Segment')
cbar2 = fig1.colorbar(im2,ax=ax3,extend='both',shrink=0.8) 

ax4 =fig1.add_subplot(2,2,4)
im3 = ax4.imshow(fullSol[:,0,:,3],aspect='auto')
ax4.set_title('B vs Segment')
cbar3 = fig1.colorbar(im3,ax=ax4,extend='both',shrink=0.8) 

fig1.tight_layout()
fig1.savefig(dirName+"/parameters.png")
####################################################
###### Plot eventual normalized conductance vs energy #######################
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


plt.show()

