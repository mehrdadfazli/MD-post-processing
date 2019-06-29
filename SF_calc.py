# -*- coding: utf-8 -*-
"""
Structure Factor (SF) and Density of States (DOS) calculator based on trajectories
Created on Mon Nov 12 16:19:22 2018

@author: Mehrdad
"""
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import time
import re
from scipy.fftpack import fft
import num

########################################################################
#--------------------------- Global Variables -------------------------#
########################################################################

id_axis = 0
time_axis = 1
xyz_axis = 2

n_Ens = 1 #Number of ensembles
n_Snp = 5001 #Number of snapshots or time points
n_Atoms = 8000
n_KP = 10 #Number of K points
n_Skip = 9 #Number of the header lines to skip

H_type = 2 
Si_type = 1
m_H = 1
m_Si = 28.09

def keyFunc(afilename):
    nondigits = re.compile("\D")
    return int(nondigits.sub("", afilename))

def readdata(file_list, col_names):
    """This function reads in the data given the file_list and the number of
       the file to read
       
       Input:
           file_list = a list of all snapshots' pathes
           idx = index of the file to read
       
       Output:
           system = a dataframe of the specified snapshot
           
    """
    assert len(file_list) == n_Snp
    
    #------------------- Initializing the outputs -----------------#
    
    vel = np.zeros([n_Atoms,n_Snp,3])
    trj = np.zeros([n_Atoms,n_Snp,3])
    Str = np.zeros([n_Atoms,n_Snp,6])
    TotE = np.zeros([n_Atoms,n_Snp])
    J_i = np.zeros([n_Atoms,n_Snp,3]) #per atom heat flux

    
    for j in range(n_Snp):
        myfile = pd.read_table(file_list[j],skiprows=9,delimiter=' ',header=None)
        
        assert len(myfile) == n_Atoms
        
        myfile.columns = col_names
        myfile = myfile.drop(columns='space')
        mySystem = myfile.sort_values(by=['atom_id']) #Sorting the system based on the atom id
        
        if j==0:
            mySystem['m'] = np.where(mySystem['type']==Si_type, m_Si, m_H)
            mass = np.array(mySystem['m'])
        
#        Hydrogens = mySystem[mySystem["type"] == H_type]
#        Silicons = mySystem[mySystem["type"] == Si_type]
        
        vel[:,j,:] = pd.concat([mySystem['vx'],mySystem['vy'],mySystem['vz']],axis=1)
        trj[:,j,:] = pd.concat([mySystem['x'],mySystem['y'],mySystem['z']],axis=1)
        Str[:,j,:] = pd.concat([mySystem['s1'],mySystem['s2'],\
                                   mySystem['s3'],mySystem['s4'],\
                                   mySystem['s5'],mySystem['s6']],axis=1)
        TotE[:,j] = mySystem['ke'] + mySystem['pe']

        J_i[:,j,0] = TotE[:,j]*vel[:,j,0] + Str[:,j,0]*vel[:,j,0] +\
                    Str[:,j,3]*vel[:,j,1] + Str[:,j,4]*vel[:,j,2]
            
        J_i[:,j,1] = TotE[:,j]*vel[:,j,0] + Str[:,j,3]*vel[:,j,0] +\
                    Str[:,j,1]*vel[:,j,1] + Str[:,j,5]*vel[:,j,2]
            
        J_i[:,j,0] = TotE[:,j]*vel[:,j,0] + Str[:,j,4]*vel[:,j,0] +\
                    Str[:,j,5]*vel[:,j,1] + Str[:,j,2]*vel[:,j,2]


    return(trj, vel, Str, TotE, J_i, mass)

def PDOS(vel, m=None, dt=1):
    """This function gets the velocities and mass and the recording frequency 
       then computes the phonon density of states
       
       Input:
           vel: a 3D numpy array [#atoms, #snapshots, 3]
           dt : period of the recording of the velocities (s)
           m  : mass array. A 1D array with masses of atoms [#atoms]
       
        Output:
            f   : the frequency array
            pdos: phonons vibrational density of states
       
    """
    fft_vel = np.abs(fft(vel, axis = time_axis))**2.0 #Fourier transforming the velocities
    full_faxis = np.fft.fftfreq(vel.shape[time_axis], dt)
    split_id = vel.shape[time_axis] // 2
    
    if m is not None:
        assert len(m) == vel.shape[id_axis]
        m = m[:,None,None]
        fft_vel *= m
    
    #----------------------------------------------#
    N = vel.shape[id_axis] #Number of particles in the system
    k_B = 1.38064852e-23 #Boltzmann Constant
    T = 300 #Temperature in Kelvin
    
    const = 3*N*k_B*T
    #----------------------------------------------#
    #Computing the PDOS(w)=sum(m_i*|v_i(w)|^2,i=1:N)/3NkT
    
    pdos = num.sum(fft_vel, axis = time_axis, keepdims=True)/const
    pdos = pdos[:split_id]
    f = full_faxis[:split_id]
   
    return(f, pdos)

def St_Fct_nd(trj, k, dt=1):
    """This function computes the number density (being the summation of 
    exp(ik.r) over all atoms) at a given time"""
    
    rho = np.zeros([n_KP,n_Snp])*1j
    
    for p in range(n_KP):
        k_rep = np.tile(k[p,:],(n_Atoms,1))
        for j in range(n_Snp):
            exponent = np.sum(np.multiply(k_rep,trj[:,j,:]),axis=1)*1j
            
            rho[p,j] = np.sum(np.exp(exponent))
    
    St_Fct = np.abs(fft(rho, axis = time_axis))**2.0
    full_faxis = np.fft.fftfreq(rho.shape[time_axis], dt)
    split_id = rho.shape[time_axis] // 2
    
    sf_rho = St_Fct[:,:split_id]
    f = full_faxis[:split_id]

    return(f,sf_rho)

def St_Fct_vel(trj, vel, k, m=None, dt=1):
    """This function gets the trajectories and k points and the recording 
       frequency, then computes the dynamical structure factor
       
       Input:
           trj: a 3D numpy array [#atoms, #snapshots, 3]
           dt : period of the recording of the velocities (s)
           k  : k points matrix. A 2D array [#k points, 3]
       
        Output:
            f : the frequency array
            sf: dynamical structure factor as a function of frequency and k
       
       """
       
    if m is None:
        m = np.ones([n_Atoms])
        
    assert trj.shape[xyz_axis] == k.shape[1] == vel.shape[xyz_axis]
    
    V_kt = np.zeros([n_KP, n_Snp, 3])*1j
    for p in range(n_KP):
        k_rep = np.tile(k[p,:],(n_Atoms,1))
        for j in range(n_Snp):
            exponent = np.exp(np.sum(np.multiply(k_rep,trj[:,0,:]),axis=1)*1j)
            
            V_kt[p,j,0] = np.sum(np.multiply(np.multiply(np.sqrt(m[:]),vel[:,j,0]),exponent))
            V_kt[p,j,1] = np.sum(np.multiply(np.multiply(np.sqrt(m[:]),vel[:,j,1]),exponent))
            V_kt[p,j,2] = np.sum(np.multiply(np.multiply(np.sqrt(m[:]),vel[:,j,2]),exponent))
            
    fft_vel = np.abs(fft(V_kt, axis = time_axis))**2.0
    full_faxis = np.fft.fftfreq(V_kt.shape[time_axis], dt)
    split_id = V_kt.shape[time_axis] // 2
    
    St_Fc_vel = fft_vel
    
    sf_vel = St_Fc_vel[:,:split_id,:]
    sf_vel = np.sum(sf_vel,axis=xyz_axis)
    f = full_faxis[:split_id]
    
    return(f,sf_vel)

def HF_func(trj, vel, k, St, E, J_i, dt=1):

    J = np.zeros([n_KP,n_Snp,3])*1j #total heat flux
    HF = np.zeros([n_KP,n_Snp,3])*1j #Total Heat Flux after FFT
    
    for p in range(n_KP):
        k_rep = np.tile(k[p,:],(n_Atoms,1))
        for j in range(n_Snp):
            exponent = np.sum(np.multiply(k_rep,trj[:,0,:]),axis=1)*1j
            
            J[p,j,0] = np.sum(np.multiply(J_i[:,j,0],np.exp(exponent)),axis=0)
            J[p,j,1] = np.sum(np.multiply(J_i[:,j,1],np.exp(exponent)),axis=0)
            J[p,j,2] = np.sum(np.multiply(J_i[:,j,2],np.exp(exponent)),axis=0)
    
    fft_J = np.abs(fft(J, axis = time_axis))**2.0
    full_faxis = np.fft.fftfreq(J.shape[time_axis], dt)
    split_id = J.shape[time_axis] // 2
    
    HF = fft_J[:,:split_id,:]
    HF = np.sum(HF,axis = xyz_axis)
    f = full_faxis[:split_id]
    
    return(f,HF)   


################################################################
##########------------------ Main -------------------###########
################################################################
        
if __name__ == '__main__':
    
    start = time.time()
   
    t_step = 5e-16
    smp_freq = 40
    total_time = 1e-10
    L_simbox = 55e-10
    L_lattice = 5e-10
    k_min = 2*np.pi/L_simbox
    k_max = 2*np.pi/L_lattice
    qVec = np.tile([1,0,0],(n_KP,1))
    aux = np.transpose(np.tile(np.linspace(0,(n_KP-1)*k_min,n_KP),(3,1)))
    k = np.multiply(qVec,aux) #matrix of K points. Each row is one k point
    columns_names = ['atom_id', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz', \
                      'ke','pe', 's1', 's2', 's3', 's4', 's5', 's6','space']
    
#    columns_names = ['atom_id', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz', \
#                      'ke','pe','space']
    
    
    ##########################################################
    #--------------- Energy Units Conversion ----------------#
    ##########################################################
    Hz_to_THz = 1e-12
    Hz_to_mev = 4.13567e-12
    Hz_to_cm = 33.35641e-12
    unit_conv = Hz_to_mev

    #####################################################################
    #------------- declaring the location of the files -----------------#
    #####################################################################
    
    trj = np.zeros([n_Ens, n_Atoms, n_Snp, 3])
    vel = np.zeros([n_Ens, n_Atoms, n_Snp, 3])
    Str = np.zeros([n_Ens, n_Atoms, n_Snp, 6])
    TotE = np.zeros([n_Ens, n_Atoms, n_Snp])
    J_i = np.zeros([n_Ens, n_Atoms, n_Snp, 3])
    
    SF_vel = np.zeros([n_Ens, n_KP, n_Snp//2])
    SF_rho = np.zeros([n_Ens, n_KP, n_Snp//2])
    PDos = np.zeros([n_Ens, n_Snp//2])
    HF_fft = np.zeros([n_Ens, n_KP, n_Snp//2])
    f = np.zeros([n_Snp//2])
    m = np.zeros([n_Atoms])
    
    for e in range(n_Ens):
        print('..Reading ensemble ',e+1,' at time ',time.ctime())
        datadir = "C:\\Users\\mf4yc\\Desktop\\Silicon\\a-Si\\{}.ens".format(e+1)
        file_list = glob.glob(os.path.join(os.getcwd(), datadir, "*.dump"))
        file_list = sorted(file_list,key=keyFunc)
    
        trj[e,:,:,:], vel[e,:,:,:], Str[e,:,:,:], TotE[e,:,:], J_i[e,:,:,:], m[:] \
        = readdata(file_list,columns_names)
    
    print('All files have been read. Starting the calculation of the properties...')
    print('Reading all files took :', (time.time() - start),' s')
    
    for e in range(n_Ens):
        f,PDos[e,:] = PDOS(vel=vel[e,:,:,:], m=m, dt=smp_freq*t_step)
        f,SF_vel[e,:,:] = St_Fct_vel(trj=trj[e,:,:,:], vel=vel[e,:,:,:],\
                m=m, dt= smp_freq*t_step, k=k)
        f,SF_rho[e,:,:] = St_Fct_nd(trj=trj[e,:,:,:], k=k, dt= smp_freq*t_step)
        f,HF_fft[e,:,:] = HF_func(trj=trj[e,:,:,:], vel=vel[e,:,:,:], k=k,\
              St= Str[e,:,:,:], E=TotE[e,:,:], J_i=J_i[e,:,:,:], dt= smp_freq*t_step)
    
    PDos = np.mean(PDos,axis=0)
    SF_vel = np.mean(SF_vel,axis=0)
    SF_rho = np.mean(SF_rho,axis=0)
    HF = np.mean(HF_fft,axis=0)
    f_Hz = f
    f_cm = f*Hz_to_cm
    f_mev = f*Hz_to_mev
    
    #------------------------ Saving Files ---------------------------#
    np.savetxt("PDOS.csv", PDos, delimiter=",")
    np.savetxt("SF_vel.csv", SF_vel, delimiter=",")
    np.savetxt("SF_rho.csv", SF_rho, delimiter=",")
    np.savetxt("HF.csv", SF_vel, delimiter=",")
    np.savetxt("f_Hz.csv", f_Hz, delimiter=",")
    np.savetxt("f_cm.csv", f_cm, delimiter=",")
    np.savetxt("f_mev.csv", f_mev, delimiter=",")
    
    end = time.time()
    print("total run time(s): ",end - start)
