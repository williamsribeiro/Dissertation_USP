import sys
import numpy as np
import emcee
import scipy as sc
import clik
import os
import camb
from emcee.utils import MPIPool
from numpy.linalg import inv
from numpy import loadtxt
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import CubicSpline
from time import time


#THIS CODE RUNS ONLY WITH MPIPOOL PARALLEL PROGRAMMMING

########################################### THIS CODE WORKS WITH DEVEL BRANCH OF CAMB ##############################################

#create folder to write the outputs
if not os.path.exists(os.getcwd()+'/outputs'):
    os.makedirs(os.getcwd()+'/outputs')

####################################################################################################################################
#DATA FOR COSMIC CLOCKS FROM AMENDOLA PAPER
zclock, Hzclock, errHz = loadtxt('/sto/home/willribeiro/Data/Hz_Amendola.dat', usecols=(0,1,2), unpack=True)


####################################################################################################################################
#DATA FOR BAO FROM TXT IN COSMOMC AND MONTEPYTHON

zeff = np.zeros(4)
bao_measur = np.zeros(4)
bao_measur_err = np.zeros(4)

#sdss_6DF_bao
zeff[0], bao_measur[0], bao_measur_err[0] = 0.106, 0.336, 0.015

#sdss_DR11CMASS_bao
zeff[1] = 0.57
bao_measur[1] = 1./13.773 #the survey information given is DV_over_rs
bao_measur_err[1] = 0.134/(13.773**2) #partial derivatives error propagation method

#sdss_DR11LOWZ_bao
zeff[2] = 0.32
bao_measur[2] = 1./8.250
bao_measur_err[2] = 0.170/(8.250**2)

#sdss_MGS_bao
zeff[3] = 0.15
bao_measur[3] = 1./4.465666824
bao_measur_err[3] = 0.1681350461/(4.465666824**2)

#WiggleZ, measurement_type: A(z), which is the acoustic parameter (arXiv:1108.2635)
zeff_wigglez = np.array([0.44, 0.60, 0.73])
Az = np.array([0.474, 0.442, 0.424])
Az_err = np.array([0.034, 0.020, 0.021])
invcovmat_wigglez = np.matrix([[1040.3,-807.5,336.8],[-807.5,3720.3,-1551.9],[336.8,-1551.9,2914.9]])


####################################################################################################################################
#DATA FOR SUPERNOVAE FROM JLA

zcmb, zhel, dz, mb, dmb, x1, dx1, color, dcolor, thirdvar, cov_m_s, cov_m_c, cov_s_c, set_jla = loadtxt('/sto/home/willribeiro/Data/jla_lcparams.txt', usecols=(1,2,3,4,5,6,7,8,9,10,14,15,16,17), unpack=True)
sigma_coh, sigma_lens, zfile = loadtxt('/sto/home/willribeiro/Data/sigma_mu.txt', usecols=(0,1,2), unpack=True)

V0 = sc.genfromtxt('/sto/home/willribeiro/Data/jla_v0_covmatrix.dat')
Va = sc.genfromtxt('/sto/home/willribeiro/Data/jla_va_covmatrix.dat')
Vb = sc.genfromtxt('/sto/home/willribeiro/Data/jla_vb_covmatrix.dat')
V0a = sc.genfromtxt('/sto/home/willribeiro/Data/jla_v0a_covmatrix.dat')
V0b = sc.genfromtxt('/sto/home/willribeiro/Data/jla_v0b_covmatrix.dat')
Vab = sc.genfromtxt('/sto/home/willribeiro/Data/jla_vab_covmatrix.dat')


####################################################################################################################################
#DATA FOR CMB FROM PLANCK LIKELIHOODS

lkl_highl = clik.clik('/sto/home/willribeiro/Data/plik_dx11dr2_HM_v18_TT.clik')
lkl_lowl = clik.clik('/sto/home/willribeiro/Data/lowl_SMW_70_dx11d_2014_10_03_v5c_Ap.clik')


########################################################################################################################################
#DEFINING SOME PARAMETERS

z1, z2, z3, z4, z5, z6, zlim = 0.0, 0.25, 0.50, 0.85, 1.25, 2.0, 2.25 #redshift bins
wlim = -1.0 #limiting w for redshifts higher than zlim

wgamma = 0.00002469 #wgamma for photons, this is well fixed by COBE satellite (Amendola, (2.45))
light = 299792.458 #speed of light in km/s
H0_jla = 70.0 #fiducial H0 value for supernovae (it has no influence on the final results)
cib_index = -1.3 #fixed nuisance from Planck

#Log information from terminal in txt
log_term = open("outputs/log_terminal.txt", "w")
log_term.close()

#######################################################################################################################################
# Define EMCEE parameters

ndim = 31 # ndim is the dimensionality of the hyperparameters
nwalkers = 2*ndim # nwalkers is the number of "chains" used by emcee (must be even)


# define priors for the parameters and nuisances, the range of parameters _r (priors from Planck 2013 and CosmoMC)
ombh2_r = np.array([0.005,0.1]) #Baryon density
omch2_r = np.array([0.001,0.99]) #CDM density
ns_r = np.array([0.8,1.2]) #Scalar spectral index (or tilt)
ln10As_r = np.array([2.0,4.0]) #ln(10^10*A_s), A_s = scalar amplitude
H0_r = np.array([20.0,100.0]) #Hubble parameter
tau_r = np.array([0.01,0.8]) #Optical depth
A_cib_217_r = np.array([0.0,200.0]) #CIB contamination at l=3000 in the 217-GHz Planck map
xi_sz_cib_r = np.array([0.0,1.0]) #SZxCIB cross-correlation
A_sz_r = np.array([0.0,10.0]) #tSZ contamination at 143GHz
ps_A_100_100_r = np.array([0.0,400.0]) #point source contribution in 100x100
ps_A_143_143_r = np.array([0.0,400.0]) #point source contribution in 143x143
ps_A_143_217_r = np.array([0.0,400.0]) #point source contribution in 143x217
ps_A_217_217_r = np.array([0.0,400.0]) #point source contribution in 217x217
ksz_norm_r = np.array([0.0,10.0]) #kSZ contamination
gal545_A_100_r = np.array([0.0,50.0]) #dust residual contamination at l=200 in 100x100
gal545_A_143_r = np.array([0.0,50.0]) #dust residual contamination at l=200 in 143x143
gal545_A_143_217_r = np.array([0.0,100.0]) #dust residual contamination at l=200 in 143x217
gal545_A_217_r = np.array([0.0,400.0]) #dust residual contamination at l=200 in 217x217
calib_100T_r = np.array([0.0,3.0]) #relative calibration between the 100 and 143 spectra
calib_217T_r = np.array([0.0,3.0]) #relative calibration between the 217 and 143 spectra
A_planck_r = np.array([0.9,1.1]) #Planck absolute calibration
alpha_r = np.array([0.0,0.3]) #stretch coefficient for supernovae (nuisance)
beta_r = np.array([1.5,4.0]) #variation in colour coefficient for supernovae (nuisance)
mag_r = np.array([-25.0,-15.0]) #Absolute magnitude for supernovae (nuisance)
delM_r = np.array([-0.3,0.3]) #correction in assuming the absolute magnitude of supernovae related to host stellar mass (nuisance)
w1_r = np.array([-10.0,8.0]) #1st w
w2_r = np.array([-10.0,8.0]) #2nd w
w3_r = np.array([-10.0,8.0]) #3rd w
w4_r = np.array([-10.0,8.0]) #4th w
w5_r = np.array([-10.0,8.0]) #5th w
w6_r = np.array([-10.0,8.0]) #6th w


# This defines the ln-prior likelihood
def lnprior(theta):
    ombh2, omch2, ns, ln10As, H0, tau, A_cib_217, xi_sz_cib, A_sz, ps_A_100_100, ps_A_143_143, ps_A_143_217, ps_A_217_217, ksz_norm, gal545_A_100, gal545_A_143, gal545_A_143_217, gal545_A_217, calib_100T, calib_217T, A_planck, alpha, beta, mag, delM, w1, w2, w3, w4, w5, w6 = theta
    
    #Analyze flat priors
    if not ((ombh2_r[0] < ombh2 < ombh2_r[1]) and (omch2_r[0] < omch2 < omch2_r[1]) and (ns_r[0] < ns < ns_r[1]) and (ln10As_r[0] < ln10As < ln10As_r[1]) and (H0_r[0] < H0 < H0_r[1]) and (tau_r[0] < tau < tau_r[1]) and (A_cib_217_r[0] < A_cib_217 < A_cib_217_r[1]) and (xi_sz_cib_r[0] < xi_sz_cib < xi_sz_cib_r[1]) and (A_sz_r[0] < A_sz < A_sz_r[1]) and (ps_A_100_100_r[0] < ps_A_100_100 < ps_A_100_100_r[1]) and (ps_A_143_143_r[0] < ps_A_143_143 < ps_A_143_143_r[1]) and (ps_A_143_217_r[0] < ps_A_143_217 < ps_A_143_217_r[1]) and (ps_A_217_217_r[0] < ps_A_217_217 < ps_A_217_217_r[1]) and (ksz_norm_r[0] < ksz_norm < ksz_norm_r[1]) and (gal545_A_100_r[0] < gal545_A_100 < gal545_A_100_r[1]) and (gal545_A_143_r[0] < gal545_A_143 < gal545_A_143_r[1]) and (gal545_A_143_217_r[0] < gal545_A_143_217 < gal545_A_143_217_r[1]) and (gal545_A_217_r[0] < gal545_A_217 < gal545_A_217_r[1]) and (calib_100T_r[0] < calib_100T < calib_100T_r[1]) and (calib_217T_r[0] < calib_217T < calib_217T_r[1]) and (A_planck_r[0] < A_planck < A_planck_r[1]) and (alpha_r[0] < alpha < alpha_r[1]) and (beta_r[0] < beta < beta_r[1]) and (mag_r[0] < mag < mag_r[1]) and (delM_r[0] < delM < delM_r[1]) and (w1_r[0] < w1 < w1_r[1]) and (w2_r[0] < w2 < w2_r[1]) and (w3_r[0] < w3 < w3_r[1]) and (w4_r[0] < w4 < w4_r[1]) and (w5_r[0] < w5 < w5_r[1]) and (w6_r[0] < w6 < w6_r[1])):
        return -np.inf
    
    #Gaussian priors
    mu1, sigma1 = 9.5, 3.0 # ksz_norm + 1.6 * A_sz
    mu2, sigma2 = 7.0, 2.0 # gal545_A_100
    mu3, sigma3 = 9.0, 2.0 # gal545_A_143
    mu4, sigma4 = 21.0, 8.5 # gal545_A_143_217
    mu5, sigma5 = 80.0, 20.0 # gal545_A_217
    mu6, sigma6 = 0.999, 0.001 # calib_100T
    mu7, sigma7 = 0.995, 0.002 # calib_217T
    mu8, sigma8 = 1.0000, 0.0025 # A_planck
    
    gauss_prior = -0.5*(ksz_norm + 1.6*A_sz-mu1)**2/sigma1**2 -0.5*(gal545_A_100-mu2)**2/sigma2**2 -0.5*(gal545_A_143-mu3)**2/sigma3**2 -0.5*(gal545_A_143_217-mu4)**2/sigma4**2 -0.5*(gal545_A_217-mu5)**2/sigma5**2 -0.5*(calib_100T-mu6)**2/sigma6**2 -0.5*(calib_217T-mu7)**2/sigma7**2 -0.5*(A_planck-mu8)**2/sigma8**2

    return gauss_prior


# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta):
    ombh2, omch2, ns, ln10As, H0, tau, A_cib_217, xi_sz_cib, A_sz, ps_A_100_100, ps_A_143_143, ps_A_143_217, ps_A_217_217, ksz_norm, gal545_A_100, gal545_A_143, gal545_A_143_217, gal545_A_217, calib_100T, calib_217T, A_planck, alpha, beta, mag, delM, w1, w2, w3, w4, w5, w6 = theta

    start = time()
    
    ############################################# CMB EVALUATION ############################################################################
    
    z = np.arange(0.0, 2.251, 0.001) #redshift subdivisions only for CMB
    a_camb = 1./(1.+z) #scale factor that goes into CAMB
    
    #Cubic spline interpolation
    w_spline = CubicSpline(np.array([z1, z2, z3, z4, z5, z6, zlim]), np.array([w1, w2, w3, w4, w5, w6, wlim]))
    
    #Planck nuisances
    nuisance_params = np.array([A_cib_217, cib_index, xi_sz_cib, A_sz, ps_A_100_100, ps_A_143_143, ps_A_143_217, ps_A_217_217, ksz_norm, gal545_A_100, gal545_A_143, gal545_A_143_217, gal545_A_217, calib_100T, calib_217T, A_planck])
    
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    
    #This function sets up parameters
    pars.DoLensing = True
    pars.NonLinear = 0
    pars.WantVectors = False
    pars.WantTensors = False
    pars.PK_WantTransfer = False
    pars.AccurateReionization = True
    pars.AccuratePolarization = True
    pars.DerivedParameters = False
    a = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS') #pycamb stuff
    w = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS') #pycamb stuff
    a, w = np.copy(a_camb[::-1]), np.copy(w_spline(z)[::-1]) #arrays must be in the reverse order, that's why the [::-1] part
    pars.set_dark_energy_w_a(a=a, w=w, dark_energy_model='ppf') #the size of the arrays must be smaller than 5000    
    pars.set_cosmology(H0=H0, cosmomc_theta=None, ombh2=ombh2, omch2=omch2, omk=0, tau=tau, neutrino_hierarchy='degenerate', num_massive_neutrinos=1, mnu=0.06, nnu=3.046, YHe=None, meffsterile=0.0, standard_neutrino_neff=3.046, TCMB=2.7255, bbn_predictor=None)
    pars.InitPower.set_params(ns=ns, As=(np.exp(ln10As)/(10.**10.)), nrun=0, nrunrun=0.0, r=0.0, nt=None, ntrun=0.0, pivot_scalar=0.05, pivot_tensor=0.05)
    pars.set_for_lmax(lmax=2800, lens_potential_accuracy=0) #because of lensing, one needs to evaluate a bigger lmax
    
    #calculate results for these parameters
    results = camb.get_results(pars)
    
    #get dictionary of CAMB power spectra
    powers = results.get_cmb_power_spectra(pars, spectra=['total'], CMB_unit='muK', raw_cl=True)
    Cls_TT = powers['total'][:2509,0]
    Cls_EE = powers['total'][:30,1]
    Cls_BB = powers['total'][:30,2]
    Cls_TE = powers['total'][:30,3]
    
    #Array that goes into the Likelihood
    cls_params = np.append(Cls_TT, nuisance_params)
    
    #Evaluates the log-likelihood for the Cl's and nuisance parameters for high-l
    log_lkl_highl = lkl_highl(cls_params)[0]
    
    #Rewriting the array to be used in the low-l likelihood
    cls_params = np.append(np.append(np.append(Cls_TT[:30], [Cls_EE, Cls_BB]), Cls_TE), A_planck)
    
    #Evaluates the log-likelihood for the Cl's and nuisance parameters for low-l
    log_lkl_lowl = lkl_lowl(cls_params)[0]
    
    
    
    
    ############################################# BAO EVALUATION ############################################################################
    #SURVEYS UNCORRELATED (6DF, SDSS-MGS, BOSS-LOWZ, BOSS-CMASS)
    z = np.arange(0.001,2.0,0.001) #redshift range for the subsequent experiments
    
    h = H0/100.
    omb = ombh2/h**2
    omc = omch2/h**2
    omr = wgamma*(1.+3.04*(7./8.)*(4./11.)**(4./3.))/(h**2) #photons plus neutrinos
    
    #Eisenstein and Hu fit
    wmatter = ombh2 + omch2
    b1 = 0.313*wmatter**(-0.419)*(1.+0.607*wmatter**(0.674))
    b2 = 0.238*wmatter**(0.223)
    zdrag = 1291.0*wmatter**(0.251)*(1.+b1*ombh2**(b2))/(1+0.659*wmatter**(0.828))    
    zeq = 2.396*(10**4)*wmatter - 1.    
    Rsdrag = (3*ombh2/(4*wgamma))/(1.+zdrag)
    Rseq = (3*ombh2/(4*wgamma))/(1.+zeq)
    ln = np.log((np.sqrt(Rsdrag+Rseq)+np.sqrt(1.+Rsdrag))/(1.+np.sqrt(Rseq)))
    
    #defining function to be integrated for w(z) evolution   
    func = lambda x: 3.*(1.+w_spline(x))/(1.+x)
    
    #integral inside the exponential function
    integral_w = np.zeros(len(z))
  
    for i in range(len(z)):
        if (i==0):
            integral_w[i] = integrate.quad(func, 0., z[i])[0]
        else:
            integral_w[i] = integral_w[i-1] + integrate.quad(func, z[i-1], z[i])[0]
    
    #interpolated integral inside the exponential function
    integr_interp = InterpolatedUnivariateSpline(z, integral_w, ext=1)
    
    #defining function E(z)
    invEz = lambda x: 1./np.sqrt(omr*(1+x)**4+(omb+omc)*(1+x)**3+(1-omb-omc-omr)*np.exp(integr_interp(x)))
      
    integral = np.ones(len(z))
    rbao = np.ones(len(z))
    A_wigglez = np.ones(len(z)) #this row is for WiggleZ evaluation    
    Hz = np.ones(len(z)) #this row is for Cosmic Clocks evaluation
    
    for i in range(len(z)):
        if (i==0):
            integral[i] = integrate.quad(invEz, 0., z[i])[0]            
        else:
            integral[i] = integral[i-1]+integrate.quad(invEz, z[i-1], z[i])[0]

        rbao[i] = (4./3.)*np.sqrt(wgamma/((omb+omc)*ombh2))*(z[i]*invEz(z[i]))**(-1./3.)*(integral[i])**(-2./3.)*ln
        
        #this row is for WiggleZ evaluation
        A_wigglez[i] = np.sqrt(omb+omc)*invEz(z[i])**(1./3.)*((1./z[i])*integral[i])**(2./3.)
        
        #this row is for Cosmic Clocks evaluation
        Hz[i] = H0*np.sqrt(omr*(1+z[i])**4+(omb+omc)*(1+z[i])**3+(1-omb-omc-omr)*np.exp(integral_w[i]))
    
    
    rBAO_interp = InterpolatedUnivariateSpline(z, rbao, ext=1)
    data_surveys = bao_measur-rBAO_interp(zeff)
    chi_uncorr_surveys = np.sum((data_surveys/bao_measur_err)**2)
    
    
    #WIGGLEZ SURVEY    
    A_interp = InterpolatedUnivariateSpline(z, A_wigglez, ext=1)
    diff_wigglez = Az - A_interp(zeff_wigglez)
    chi_wigglez = np.dot(np.dot(diff_wigglez, invcovmat_wigglez), diff_wigglez)
    
    chi_BAO = chi_uncorr_surveys + chi_wigglez
    
    
    
    
    ########################################### COSMIC CLOCKS EVALUATION #################################################################
    
    Hz_interp = InterpolatedUnivariateSpline(z, Hz, ext=1)
    data_clocks = Hzclock-Hz_interp(zclock)
    chi_clocks = np.sum((data_clocks/errHz)**2)
    
    
    
    
    ########################################### SUPERNOVA EVALUATION ####################################################################
    
    #integral of invE(z) interpolated
    integr_interp2 = InterpolatedUnivariateSpline(z, integral, ext=1)    
    lum_dist = light*(1+zhel)/H0_jla*integr_interp2(zcmb)
    
    #this block is made for evaluating delM depending on host stellar mass
    mbtheory = np.zeros(len(thirdvar))
    for i in range(len(thirdvar)):
        if(thirdvar[i] < 10.0):
            mbtheory[i] = 5*np.log10(lum_dist[i]) + 25 - alpha*x1[i] + beta*color[i] + mag
        else:
            mbtheory[i] = 5*np.log10(lum_dist[i]) + 25 - alpha*x1[i] + beta*color[i] + mag + delM

    
    #Covariance matrix evaluation
    Dvec = (5.*dz/(zfile*np.log(10)))**2 + sigma_lens**2 + sigma_coh**2 + dmb**2 + (alpha**2)*(dx1**2) + (beta**2)*(dcolor**2) + 2*alpha*cov_m_s - 2*beta*cov_m_c - 2*alpha*beta*cov_s_c
    Dvec = np.diag(Dvec)
    Cvec = V0 + (alpha**2)*Va + (beta**2)*Vb + 2*alpha*V0a - 2*beta*V0b - 2*alpha*beta*Vab
    Cvec = Cvec.reshape(len(zcmb),len(zcmb))    
    covmat = Dvec + Cvec
    
    difference = mb-mbtheory
    chi_SN = np.dot(np.dot(difference, inv(covmat)), difference)
    
    end = time()
    
    #Counting steps and prints
    #global counter
    
    #Print log information from terminal in txt
    log_term = open("outputs/log_terminal.txt", "a")
    log_term.write('Time for this MC point (in sec) = %.3f' %(end-start) + "\n" + "\n")
    log_term.close()  
    
    return (-0.5*(chi_clocks + chi_BAO + chi_SN) + log_lkl_highl + log_lkl_lowl)


# This defines the ln-likelihood that goes into EMCEE
def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf      
    return lp + lnlike(theta)


pool = MPIPool(loadbalance=True)
if not pool.is_master():
    pool.wait()
    sys.exit(0)


# initialize emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

#Load the position of the walkers after the burn-in step and time for each MCMC point
pos = np.loadtxt("outputs/pos_after_burn_in.txt")
delta_t = np.loadtxt("outputs/delta_t.txt")


log_term = open("outputs/log_terminal.txt", "a")
log_term.write("Time for each MCMC point: %.5g" %delta_t + "\n")


#Repetitions of MCMC
repetitions = 40000
steps_per_repetition = 2
total_steps = repetitions*steps_per_repetition #this is total steps per walker!
dur_MCMC = delta_t*nwalkers*(steps_per_repetition + total_steps) #total duration time of MCMC

mm, ss = divmod(dur_MCMC, 60)
hh, mm = divmod(mm, 60)
log_term.write("Estimated duration of MCMC from now on: " + ("%d:%02d:%02d" % (hh, mm, ss)) + "\n")
log_term.write("Total points to be evaluated from now on: %d" %(nwalkers*(steps_per_repetition + total_steps)) + "\n" + "\n")
log_term.close()

#initial time to record MCMC complete time
timei = time()


# Start MC exploration
# Will save one chain to the file below:
f = open("outputs/chain.txt", "w")
f.close()


sampler.reset()
posf, probf, statef = sampler.run_mcmc(pos, steps_per_repetition)
weights = sampler.flatlnprobability #evaluates the weights for every MCMC point
np.savetxt('outputs/weights.txt', weights, fmt='%1.8e') #saves weights array to txt

accept_num = steps_per_repetition*sampler.acceptance_fraction #array with number of steps accepted for each walker
total_num = steps_per_repetition*np.ones(nwalkers) #array with total number of steps for each walker
np.savetxt('outputs/accept_ratio.txt', accept_num/total_num, fmt='%1.8e') #saves acceptance ratio for every MCMC point

f = open("outputs/chain.txt", "a")
for k in range(sampler.flatchain.shape[0]):
    f.write("  ".join(map(str, sampler.flatchain[k])) + "\n") #write every line in txt file
f.close()


#Saving derived parameters to txt (array is [omLambda, omM, Age/Gyr, chi_square])
der = open("outputs/derived_params.txt", "w")
der.close()
der = open("outputs/derived_params.txt", "a")
timei_d = time()
z = np.arange(0.0, 2.251, 0.001)
a_camb = 1./(1.+z) #scale factor that goes into CAMB
for k in range(sampler.flatchain.shape[0]):
    pars = camb.CAMBparams()
    w_spline = CubicSpline(np.array([z1, z2, z3, z4, z5, z6, zlim]), np.array([sampler.flatchain[k][25], sampler.flatchain[k][26], sampler.flatchain[k][27], sampler.flatchain[k][28], sampler.flatchain[k][29], sampler.flatchain[k][30], wlim]))
    a = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS') #pycamb stuff
    w = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS') #pycamb stuff
    a, w = np.copy(a_camb[::-1]), np.copy(w_spline(z)[::-1]) #arrays must be in the reverse order, that's why the [::-1] part
    pars.set_dark_energy_w_a(a=a, w=w, dark_energy_model='ppf') #the size of the arrays must be smaller than 5000    
    pars.set_cosmology(H0=sampler.flatchain[k][4], cosmomc_theta=None, ombh2=sampler.flatchain[k][0], omch2=sampler.flatchain[k][1], omk=0, tau=sampler.flatchain[k][5], neutrino_hierarchy='degenerate', num_massive_neutrinos=1, mnu=0.06, nnu=3.046, YHe=None, meffsterile=0.0, standard_neutrino_neff=3.046, TCMB=2.7255, bbn_predictor=None)
    omL = pars.omegav
    omM = pars.omegab + pars.omegac
    age = camb.get_age(pars)
    chi = -2.0*sampler.flatlnprobability[k]
    der.write("  ".join(map(str, np.array([omL, omM, age, chi]))) + "\n")
timef_d = time()
mm, ss = divmod((timef_d - timei_d), 60)
hh, mm = divmod(mm, 60)
der.close()
log_term = open("outputs/log_terminal.txt", "a")
log_term.write('End of derived parameters saving. Time spent: ' + ("%d:%02d:%02d" % (hh, mm, ss)) + '\n')
log_term.close()


#Main MC exploration
for nstore in range(repetitions):
    sampler.reset()
    posf, probf, statef = sampler.run_mcmc(posf, steps_per_repetition)
    
    #Saves weights to txt
    f = open("outputs/weights.txt", "a")
    for k in range(sampler.flatlnprobability.shape[0]):
        f.write(str(round(sampler.flatlnprobability[k], 8)) + "\n")
    f.close()

    accept_num += steps_per_repetition*sampler.acceptance_fraction #array with number of steps accepted for each walker
    total_num += steps_per_repetition*np.ones(nwalkers) #array with total number of steps for each walker
    np.savetxt('outputs/accept_ratio.txt', accept_num/total_num, fmt='%1.8e') #saves acceptance ratio for every MCMC point

    #Saves data to txt
    f = open("outputs/chain.txt", "a")
    for k in range(sampler.flatchain.shape[0]):
        f.write("  ".join(map(str, sampler.flatchain[k])) + "\n")
    f.close()
    
    #Saving derived parameters to txt (array is [omLambda, omM, Age/Gyr, chi_square])
    der = open("outputs/derived_params.txt", "a")
    timei_d = time()
    for k in range(sampler.flatchain.shape[0]):
        pars = camb.CAMBparams()
        w_spline = CubicSpline(np.array([z1, z2, z3, z4, z5, z6, zlim]), np.array([sampler.flatchain[k][25], sampler.flatchain[k][26], sampler.flatchain[k][27], sampler.flatchain[k][28], sampler.flatchain[k][29], sampler.flatchain[k][30], wlim]))
        a = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS') #pycamb stuff
        w = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS') #pycamb stuff
        a, w = np.copy(a_camb[::-1]), np.copy(w_spline(z)[::-1]) #arrays must be in the reverse order, that's why the [::-1] part
        pars.set_dark_energy_w_a(a=a, w=w, dark_energy_model='ppf') #the size of the arrays must be smaller than 5000
        pars.set_cosmology(H0=sampler.flatchain[k][4], cosmomc_theta=None, ombh2=sampler.flatchain[k][0], omch2=sampler.flatchain[k][1], omk=0, tau=sampler.flatchain[k][5], neutrino_hierarchy='degenerate', num_massive_neutrinos=1, mnu=0.06, nnu=3.046, YHe=None, meffsterile=0.0, standard_neutrino_neff=3.046, TCMB=2.7255, bbn_predictor=None)
        omL = pars.omegav
        omM = pars.omegab + pars.omegac
        age = camb.get_age(pars)
        chi = -2.0*sampler.flatlnprobability[k]
        der.write("  ".join(map(str, np.array([omL, omM, age, chi]))) + "\n")
    timef_d = time()
    mm, ss = divmod((timef_d - timei_d), 60)
    hh, mm = divmod(mm, 60)
    der.close()
    log_term = open("outputs/log_terminal.txt", "a")
    log_term.write('End of derived parameters saving. Time spent: ' + ("%d:%02d:%02d" % (hh, mm, ss)) + '\n' + '\n')
    log_term.close() 
    
    
    #Saves the position of the walkers in txt after a number of repetitions
    pos_txt = open("outputs/pos_after_burn_in.txt", "w")
    pos_txt.close()
    pos_txt = open("outputs/pos_after_burn_in.txt", "a")
    for k in range(nwalkers):
        pos_txt.write("  ".join(map(str, posf[k,])) + "\n") #write every line in txt file
    pos_txt.close()

    #Repetition number
    np.savetxt('outputs/repetitions_evaluated.txt', np.array([(nstore+2)]), fmt="%i")
    



pool.close()

log_term = open("outputs/log_terminal.txt", "a")
mm, ss = divmod(time() - timei, 60)
hh, mm = divmod(mm, 60)
log_term.write("Real duration of MCMC: " + ("%d:%02d:%02d" % (hh, mm, ss)))
log_term.close()




