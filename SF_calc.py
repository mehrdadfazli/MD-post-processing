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

def keyFunc(afilename):
    nondigits = re.compile("\D")
    return int(nondigits.sub("", afilename))

def readdata(file_list,num):
    """This function reads in the data given the file_list and the number of
    the file to read"""
#    column_names = ["atom_id", "type", "x", "y", "z", "vx", "vy", "vz", \
#                      "ke","pe", "space"]
    myfile = pd.read_table(file_list[num],skiprows=9,delimiter=' ',header=None)
    myfile.columns = ["atom_id", "type", "x", "y", "z", "vx", "vy", "vz", \
                      "ke","pe", "space"]
    myfile = myfile.drop(columns='space')
    system = myfile.sort_values(by=['atom_id']) #Sorting the system based on the atom id
    return(system)

def num_den(system,k):
    """This function computes the number density (being the summation of 
    exp(ik.r) over all atoms) at a given time"""
    rho = 0*1j
    for p in range(0,len(system)):
        rho += np.exp((k[i,0]*system["x"][p]+ \
                       k[i,1]*system["y"][p]+ \
                       k[i,2]*system["z"][p])*1j) #k is the matrix of k points
    return(rho)

def PDOS(vel):
    """This function gets the velocities and computes the phonon density of states"""
    vel_w = np.fft.fft(vel,axis = time_axis) #Fourier transforming the velocities
    vel_w_abs = np.absolute(vel_w) #Getting the absolute value of the velocities
    
    #----------------------------------------------#
    N = len(vel) #Number of particles in the system
    k_B = 1.38064852e-23 #Boltzmann Constant
    T = 300 #Temperature in Kelvin
    
    const = 3*N*k_B*T
    #----------------------------------------------#
    #Computing the PDOS(w)=sum(|v_i(w)|^2,i=1:N)/3NkT
    pdos = np.sum(np.power(vel_w_abs,2),axis = id_axis)/const
    return(pdos)

def SF(num_den):
    """This function computes the structure factor from number density"""
    num_den_w = np.fft.fft(rho,axis=time_axis)
    StFct = np.power(np.abs(num_den_w),2)
    return(StFct)

def dot(a,b):
    c = 0.0
    c = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    return(c)
    
def VACF(vel):
    #----------------------- Smoothening -------------------------#
    t_damp = np.shape(vel)[1]*np.sqrt(3)
    for j in range(0,np.shape(vel)[1]):
        vel[:,j,:] = vel[:,j,:]*np.exp(-(j/t_damp)**2)
           
    #-------------------------------------------------------------#
    
    vacf     = np.zeros(np.shape(vel)[1])
    vacf_num = np.zeros(np.shape(vel)[1])
    vacf_den = np.zeros(np.shape(vel)[1])
    
    for j in range(0,np.shape(vel)[1]): #np.shape(vel)[1] is the number of snapshots
        for i in range(0,(np.shape(vel)[0]-1)): #np.shape(vel)[0] is the number of the atoms
            vacf_num[j] =+ dot(vel[i,0,:],vel[i,j,:])
            vacf_den[j] =+ dot(vel[i,0,:],vel[i,0,:])
        vacf[j] = vacf_num[j]/vacf_den[j]
    return(vacf)


###########------------------- PWTOOLS --------------------################
def pdos(vel, dt=1.0, m=None, full_out=False, area=1.0, window=True,
         npad=None, tonext=False, mirr=False, method='direct'):
    """Phonon DOS by FFT of the VACF or direct FFT of atomic velocities.
    
    Integral area is normalized to `area`. It is possible (and recommended) to
    zero-padd the velocities (see `npad`). 
    
    Parameters
    ----------
    vel : 3d array (nstep, natoms, 3)
        atomic velocities
    dt : time step
    m : 1d array (natoms,), 
        atomic mass array, if None then mass=1.0 for all atoms is used  
    full_out : bool
    area : float
        normalize area under frequency-PDOS curve to this value
    window : bool 
        use Welch windowing on data before FFT (reduces leaking effect,
        recommended)
    npad : {None, int}
        method='direct' only: Length of zero padding along `axis`. `npad=None`
        = no padding, `npad > 0` = pad by a length of ``(nstep-1)*npad``. `npad
        > 5` usually results in sufficient interpolation.
    tonext : bool
        method='direct' only: Pad `vel` with zeros along `axis` up to the next
        power of two after the array length determined by `npad`. This gives
        you speed, but variable (better) frequency resolution.
    mirr : bool 
        method='vacf' only: mirror one-sided VACF at t=0 before fft
    Returns
    -------
    if full_out = False
        | ``(faxis, pdos)``
        | faxis : 1d array [1/unit(dt)]
        | pdos : 1d array, the phonon DOS, normalized to `area`
    if full_out = True
        | if method == 'direct':
        |     ``(faxis, pdos, (full_faxis, full_pdos, split_idx))``
        | if method == 'vavcf':
        |     ``(faxis, pdos, (full_faxis, full_pdos, split_idx, vacf, fft_vacf))``
        |     fft_vacf : 1d complex array, result of fft(vacf) or fft(mirror(vacf))
        |     vacf : 1d array, the VACF
    
    Examples
    --------
    >>> from pwtools.constants import fs,rcm_to_Hz
    >>> tr = Trajectory(...)
    >>> # freq in [Hz] if timestep in [s]
    >>> freq,dos = pdos(tr.velocity, m=tr.mass, dt=tr.timestep*fs, 
    >>>                 method='direct', npad=1)
    >>> # frequency in [1/cm]
    >>> plot(freq/rcm_to_Hz, dos)
    
    Notes
    -----
    padding (only method='direct'): With `npad` we pad the velocities `vel`
    with ``npad*(nstep-1)`` zeros along `axis` (the time axis) before FFT
    b/c the signal is not periodic. For `npad=1`, this gives us the exact
    same spectrum and frequency resolution as with ``pdos(...,
    method='vacf',mirr=True)`` b/c the array to be fft'ed has length
    ``2*nstep-1`` along the time axis in both cases (remember that the
    array length = length of the time axis influences the freq.
    resolution). FFT is only fast for arrays with length = a power of two.
    Therefore, you may get very different fft speeds depending on whether
    ``2*nstep-1`` is a power of two or not (in most cases it won't). Try
    using `tonext` but remember that you get another (better) frequency
    resolution.
    References
    ----------
    [1] Phys Rev B 47(9) 4863, 1993
    See Also
    --------
    :func:`pwtools.signal.fftsample`
    :func:`pwtools.signal.acorr`
    :func:`direct_pdos`
    :func:`vacf_pdos`
    """
    mass = m
    # assume vel.shape = (nstep,natoms,3)
    axis = 0
    assert vel.shape[-1] == 3
    if mass is not None:
        assert len(mass) == vel.shape[1], "len(mass) != vel.shape[1]"
        # define here b/c may be used twice below
        mass_bc = mass[None,:,None]
    if window:
        sl = [None]*vel.ndim 
        sl[axis] = slice(None) # ':'
        vel2 = vel*(welch(vel.shape[axis])[sl])
    else:
        vel2 = vel
    # handle options which are mutually exclusive
    if method == 'vacf':
        assert npad in [0,None], "use npad={0,None} for method='vacf'"
    # padding
    if npad is not None:
        nadd = (vel2.shape[axis]-1)*npad
        if tonext:
            vel2 = pad_zeros(vel2, tonext=True, 
                             tonext_min=vel2.shape[axis] + nadd, 
                             axis=axis)
        else:    
            vel2 = pad_zeros(vel2, tonext=False, nadd=nadd, axis=axis)
    if method == 'direct': 
        full_fft_vel = np.abs(fft(vel2, axis=axis))**2.0
        full_faxis = np.fft.fftfreq(vel2.shape[axis], dt)
        split_idx = len(full_faxis)//2
        faxis = full_faxis[:split_idx]
        # First split the array, then multiply by `mass` and average. If
        # full_out, then we need full_fft_vel below, so copy before slicing.
        arr = full_fft_vel.copy() if full_out else full_fft_vel
        fft_vel = num.slicetake(arr, slice(0, split_idx), axis=axis, copy=False)
        if mass is not None:
            fft_vel *= mass_bc
        # average remaining axes, summing is enough b/c normalization is done below
        # sums: (nstep, natoms, 3) -> (nstep, natoms) -> (nstep,)
        pdos = num.sum(fft_vel, axis=axis, keepdims=True)
        default_out = (faxis, num.norm_int(pdos, faxis, area=area))
        if full_out:
            # have to re-calculate this here b/c we never calculate the full_pdos
            # normally
            if mass is not None:
                full_fft_vel *= mass_bc
            full_pdos = num.sum(full_fft_vel, axis=axis, keepdims=True)
            extra_out = (full_faxis, full_pdos, split_idx)
            return default_out + extra_out
        else:
            return default_out
    elif method == 'vacf':
        vacf = fvacf(vel2, m=mass)
        if mirr:
            fft_vacf = fft(mirror(vacf))
        else:
            fft_vacf = fft(vacf)
        full_faxis = np.fft.fftfreq(fft_vacf.shape[axis], dt)
        full_pdos = np.abs(fft_vacf)
        split_idx = len(full_faxis)//2
        faxis = full_faxis[:split_idx]
        pdos = full_pdos[:split_idx]
        default_out = (faxis, num.norm_int(pdos, faxis, area=area))
        extra_out = (full_faxis, full_pdos, split_idx, vacf, fft_vacf)
        if full_out:
            return default_out + extra_out
        else:
            return default_out


def vacf_pdos(vel, *args, **kwds):
    """Wrapper for ``pdos(..., method='vacf', mirr=True, npad=None)``"""
    if 'mirr' not in kwds:
        kwds['mirr'] = True
    return pdos(vel, *args, method='vacf', npad=None, **kwds)


def direct_pdos(vel, *args, **kwds):
    """Wrapper for ``pdos(..., method='direct', npad=1)``"""
    if 'npad' not in kwds:
        kwds['npad'] = 1
    if 'pad_tonext' in kwds:
##        warnings.simplefilter('always')
        warnings.warn("'pad_tonext' was renamed 'tonext'",
            DeprecationWarning)
        kwds['tonext'] = kwds['pad_tonext']
        kwds.pop('pad_tonext')
    return pdos(vel, *args, method='direct', **kwds)

def welch(M, sym=1):
    """Welch window. Function skeleton shamelessly stolen from
    scipy.signal.bartlett() and others."""
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1,dtype=float)
    odd = M % 2
    if not sym and not odd:
        M = M+1
    n = np.arange(0,M)
    w = 1.0-((n-0.5*(M-1))/(0.5*(M-1)))**2.0
    if not sym and not odd:
        w = w[:-1]
    return w

def pad_zeros(arr, axis=0, where='end', nadd=None, upto=None, tonext=None,
              tonext_min=None):
    """Pad an nd-array with zeros. Default is to append an array of zeros of 
    the same shape as `arr` to arr's end along `axis`.
    
    Parameters
    ----------
    arr :  nd array
    axis : the axis along which to pad
    where : string {'end', 'start'}, pad at the end ("append to array") or 
        start ("prepend to array") of `axis`
    nadd : number of items to padd (i.e. nadd=3 means padd w/ 3 zeros in case
        of an 1d array)
    upto : pad until arr.shape[axis] == upto
    tonext : bool, pad up to the next power of two (pad so that the padded 
        array has a length of power of two)
    tonext_min : int, when using `tonext`, pad the array to the next possible
        power of two for which the resulting array length along `axis` is at
        least `tonext_min`; the default is tonext_min = arr.shape[axis]
    Use only one of nadd, upto, tonext.
    
    Returns
    -------
    padded array
    Examples
    --------
    >>> # 1d 
    >>> pad_zeros(a)
    array([1, 2, 3, 0, 0, 0])
    >>> pad_zeros(a, nadd=3)
    array([1, 2, 3, 0, 0, 0])
    >>> pad_zeros(a, upto=6)
    array([1, 2, 3, 0, 0, 0])
    >>> pad_zeros(a, nadd=1)
    array([1, 2, 3, 0])
    >>> pad_zeros(a, nadd=1, where='start')
    array([0, 1, 2, 3])
    >>> # 2d
    >>> a=arange(9).reshape(3,3)
    >>> pad_zeros(a, nadd=1, axis=0)
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8],
           [0, 0, 0]])
    >>> pad_zeros(a, nadd=1, axis=1)
    array([[0, 1, 2, 0],
           [3, 4, 5, 0],
           [6, 7, 8, 0]])
    >>> # up to next power of two           
    >>> 2**arange(10)
    array([  1,   2,   4,   8,  16,  32,  64, 128, 256, 512])
    >>> pydos.pad_zeros(arange(9), tonext=True).shape
    (16,)
    """
    if tonext == False:
        tonext = None
    lst = [nadd, upto, tonext]
    assert lst.count(None) in [2,3], "`nadd`, `upto` and `tonext` must be " +\
           "all None or only one of them not None"
    if nadd is None:
        if upto is None:
            if (tonext is None) or (not tonext):
                # default
                nadd = arr.shape[axis]
            else:
                tonext_min = arr.shape[axis] if (tonext_min is None) \
                             else tonext_min
                # beware of int overflows starting w/ 2**arange(64), but we
                # will never have such long arrays anyway
                two_powers = 2**np.arange(30)
                assert tonext_min <= two_powers[-1], ("tonext_min exceeds "
                    "max power of 2")
                power = two_powers[np.searchsorted(two_powers,
                                                  tonext_min)]
                nadd = power - arr.shape[axis]                                                       
        else:
            nadd = upto - arr.shape[axis]
    if nadd == 0:
        return arr
    add_shape = list(arr.shape)
    add_shape[axis] = nadd
    add_shape = tuple(add_shape)
    if where == 'end':
        return np.concatenate((arr, np.zeros(add_shape, dtype=arr.dtype)), axis=axis)
    elif where == 'start':        
        return np.concatenate((np.zeros(add_shape, dtype=arr.dtype), arr), axis=axis)
    else:
        raise Exception("illegal `where` arg: %s" %where)


################################################################
##########------------------ Main -------------------###########
################################################################
        
if __name__ == '__main__':
    
    start = time.time()

    #"""""""""""Initialization""""""""""""
    n_Snp = 10001 #Number of snapshots or time points
    n_atoms = 8000
    n_KP = 10 #Number of K points
    n_Skip = 9 #Number of the header lines to skip
    
    H_type = 2 
    Si_type = 1
       
    #Vector along which SF is to be calculated
    L_simbox = 55e-10
    L_lattice = 5e-10
    k_min = 2*np.pi/L_simbox
    k_max = 2*np.pi/L_lattice
    qVec = np.tile([1,0,0],(n_KP,1))
    aux = np.transpose(np.tile(np.linspace(0,(n_KP-1)*k_min,n_KP),(3,1)))
    k = np.multiply(qVec,aux) #matrix of K points. Each row is one k point
    rho = np.zeros([n_KP,n_Snp])*1j #Defining the number density
    
    #####################################################################
    #------------- declaring the location of the files -----------------#
    #####################################################################
    
#    datadir = "C:\\Users\\mf4yc\\Desktop\\Amorphous Silicon\\SiH\\DumpFiles\\c-Si"
    datadir = "C:\\Users\\mf4yc\\Desktop\\Silicon\\c-Si\\0.1ns_dump\\ens_1.txt"
    filepath = datadir
    file = open(filepath,'r')
#    file_list = glob.glob(os.path.join(os.getcwd(), datadir, "*.dump"))
#    file_list = sorted(file_list,key=keyFunc)
    
    ######################################################################
    #-------- More Initialization by reading in the first snpshot -------#
    ######################################################################
    
#    mySystem = readdata(file_list,0)
#    Hydrogens = mySystem[mySystem["type"]==H_type]
#    Silicons = mySystem[mySystem["type"]==Si_type]
    
#    n_H = len(Hydrogens) #Numebr of H in the system
#    n_Si = len(Silicons) #Number of Si in the system
    
    #defining variables to store the velocity of the whole system, hydrogens
    #and silicons and their absolute values
#    vel_H = np.zeros([len(Hydrogens),len(file_list),3])
#    vel_Si = np.zeros([len(Silicons),len(file_list),3])
    vel_sys = np.zeros([n_atoms,n_Snp,3])
#    vel_H_abs = np.zeros([len(Hydrogens),len(file_list)])
#    vel_Si_abs = np.zeros([len(Silicons),len(file_list)])
    vel_sys_abs = np.zeros([n_atoms,n_Snp])
    column_names = ["atom_id", "type", "x", "y", "z", "vx", "vy", "vz", \
                      "ke","pe", "space"]

    
    ###############################################################
    #---------------- Looping over all snpshots ------------------#
    ###############################################################
    #Computing number density rho(k,t)=Sum(exp(i*X.k))
    for j in range(0,n_Snp): # j here represents the time
    #for j in range(0,100): # j here represents the time
#        mySystem = readdata(file_list,j)
        skipped_head = j*(n_atoms + n_Skip) + n_Skip
        mySystem = pd.read_table(file,header=skipped_head,nrows=8000,\
                                 delimiter=' ', names=column_names)
    

        Hydrogens = mySystem[mySystem["type"] == H_type]
        Silicons = mySystem[mySystem["type"] == Si_type]
        
        vel_sys[:,j,:] = pd.concat([mySystem["vx"],mySystem["vy"],mySystem["vz"]],axis=1)
#        vel_H[:,j,:] = pd.concat([Hydrogens["vx"],Hydrogens["vy"],Hydrogens["vz"]],axis=1)
#        vel_Si[:,j,:] = pd.concat([Silicons["vx"],Silicons["vy"],Silicons["vz"]],axis=1)
        
        vel_sys_abs[:,j] = np.sqrt(np.power(mySystem["vx"],2) +\
                               np.power(mySystem["vy"],2) +\
                               np.power(mySystem["vz"],2))
        
#        vel_H_abs[:,j] = np.sqrt(np.power(Hydrogens["vx"],2) +\
#                               np.power(Hydrogens["vy"],2) +\
#                               np.power(Hydrogens["vz"],2))
#        
#        vel_Si_abs[:,j] = np.sqrt(np.power(Silicons["vx"],2) +\
#                               np.power(Silicons["vy"],2) +\
#                               np.power(Silicons["vz"],2))
        
        print(j)
#        for i in range(0,len(k)): # i here represents the k
#            rho[i,j] = num_den(mySystem,k)
            
#        Vel_H[:,0,j] = Hydrogens["vx"]
#        Vel_H[:,1,j] = Hydrogens["vy"]
#        Vel_H[:,2,j] = Hydrogens["vz"]
#        Vel_Si[:,0,j] = Silicons["vx"]
#        Vel_Si[:,1,j] = Silicons["vy"]
#        Vel_Si[:,2,j] = Silicons["vz"]
        
        mySystem["V"] = np.sqrt(np.power(mySystem["vx"],2) +\
                                np.power(mySystem["vy"],2) +\
                                np.power(mySystem["vz"],2))
#        Hydrogens["V"] = np.sqrt(Hydrogens["vx"]^2+Hydrogens["vy"]^2+Hydrogens["vz"]^2)
#        Silicons["V"] = np.sqrt(Silicons["vx"]^2+Silicons["vy"]^2+Silicons["vz"]^2)
        
        
    
#    StFct = SF(rho) #Obtaining structure factor
#    
#    ph_dos = PDOS(vel_sys) #Obtaining phonon DOS
#    Si_ph_dos = PDOS(vel_Si_abs)
#    H_ph_dos = PDOS(vel_H_abs)
#    
#    ##########################################################
#    #--------------- Frequencey Calculations ----------------#
#    ##########################################################
    timestep = 5e-16
    sampling = 20
    total_time = 1e-10
    
    w=[]
    for i in range(n_Snp):
        w.append((i-n_Snp/2)/(sampling*timestep*n_Snp)*2*np.pi)
    
#    w_max = 2*np.pi/(timestep*sampling)
#    w_min = 2*np.pi/total_time
#    ##########################################################
#    #--------------- Energy Units Conversion ----------------#
#    ##########################################################
    Hz_to_THz = 1e-12
    Hz_to_mev = 4.13567e-12
    Hz_to_cm = 33.35641e-12
#    
    unit_conv = Hz_to_THz
#    w=np.linspace(w_min,w_max,len(file_list))*unit_conv
    
#    
#    end = time.time()
#    print("total run time(s): ",end - start)
#    #Plotting the SF
#    #plt.plot(w[100:900],StFct[19,100:900])
#    
#    #Plotting DOS
#    #plt.plot(w[100:900],ph_dos[100:900])
#    vacf_sys = VACF(vel_sys)
    
    vel = np.zeros([vel_sys.shape[1],vel_sys.shape[0],vel_sys.shape[2]])
    for i in range(vel_sys.shape[0]):
        for j in range(vel_sys.shape[1]):
            vel[j,i,:] = vel_sys[i,j,:]
        

##############------------ pw tools ------------############
