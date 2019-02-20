import numpy as np
import emcee
import scipy as sc
import os
from numpy.linalg import inv
from numpy import loadtxt
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import CubicSpline
from time import time

#create folder to write the outputs
if not os.path.exists(os.getcwd()+'/outputs'):
    os.makedirs(os.getcwd()+'/outputs')

####################################################################################################################################
#DATA FOR COSMIC CLOCKS FROM AMENDOLA PAPER
zclock, Hzclock, errHz = loadtxt('Hz_Amendola.dat', usecols=(0,1,2), unpack=True)


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

zcmb, zhel, dz, mb, dmb, x1, dx1, color, dcolor, thirdvar, cov_m_s, cov_m_c, cov_s_c, set_jla = loadtxt('jla_lcparams.txt', usecols=(1,2,3,4,5,6,7,8,9,10,14,15,16,17), unpack=True)
sigma_coh, sigma_lens, zfile = loadtxt('sigma_mu.txt', usecols=(0,1,2), unpack=True)

V0 = sc.genfromtxt('jla_v0_covmatrix.dat')
Va = sc.genfromtxt('jla_va_covmatrix.dat')
Vb = sc.genfromtxt('jla_vb_covmatrix.dat')
V0a = sc.genfromtxt('jla_v0a_covmatrix.dat')
V0b = sc.genfromtxt('jla_v0b_covmatrix.dat')
Vab = sc.genfromtxt('jla_vab_covmatrix.dat')

########################################################################################################################################
#DEFINING SOME PARAMETERS

z1, z2, z3, z4, z5, z6 = 0.0, 0.25, 0.50, 0.85, 1.25, 2.0 #redshift bins

wgamma = 0.00002469 #wgamma for photons, this is well fixed by COBE satellite (Amendola, (2.45))
light = 299792.458 #speed of light in km/s
H0_jla = 70.0 #fiducial H0 value for supernovae (it has no influence on the final results)

#Log information from terminal in txt
log_term = open("outputs/log_terminal.txt", "w")
log_term.close()

counter = 1 #counter of steps

#######################################################################################################################################
# Define EMCEE parameters

ndim = 13 # ndim is the dimensionality of the hyperparameters
nwalkers = 2*ndim # nwalkers is the number of "chains" used by emcee (must be even)


# define priors for the parameters and nuisances, the range of parameters _r (priors from Planck 2013 and CosmoMC)
ombh2_r = np.array([0.005,0.1]) #Baryon density
omch2_r = np.array([0.001,0.99]) #CDM density
H0_r = np.array([20.0,100.0]) #Hubble parameter today
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
    ombh2, omch2, H0, alpha, beta, mag, delM, w1, w2, w3, w4, w5, w6 = theta
    
    #Analyze flat priors
    if not ((ombh2_r[0] < ombh2 < ombh2_r[1]) and (omch2_r[0] < omch2 < omch2_r[1]) and (H0_r[0] < H0 < H0_r[1]) and (alpha_r[0] < alpha < alpha_r[1]) and (beta_r[0] < beta < beta_r[1]) and (mag_r[0] < mag < mag_r[1]) and (delM_r[0] < delM < delM_r[1]) and (w1_r[0] < w1 < w1_r[1]) and (w2_r[0] < w2 < w2_r[1]) and (w3_r[0] < w3 < w3_r[1]) and (w4_r[0] < w4 < w4_r[1]) and (w5_r[0] < w5 < w5_r[1]) and (w6_r[0] < w6 < w6_r[1])):
        return -np.inf

    return 0.0


# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta):
    ombh2, omch2, H0, alpha, beta, mag, delM, w1, w2, w3, w4, w5, w6 = theta

    #Cubic spline interpolation
    w_spline = CubicSpline(np.array([z1, z2, z3, z4, z5, z6]), np.array([w1, w2, w3, w4, w5, w6]))
    
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
    
    
    
    
    #Counting steps and prints
    global counter
    print("Counter = %d" %(counter) + "\n")
    
    #Print log information from terminal in txt
    log_term = open("outputs/log_terminal.txt", "a")
    log_term.write('Counter = %d' %(counter) + "\n")
    log_term.write("chi_clocks = %.2f" %(chi_clocks) + "\n")
    log_term.write("chi_BAO = %.2f" %(chi_BAO) + "\n")
    log_term.write("chi_SN = %.2f" %(chi_SN) + "\n" + "\n")
    log_term.close()  
    
    counter += 1
    
    return (-0.5*(chi_clocks + chi_BAO + chi_SN))


# This defines the ln-likelihood that goes into EMCEE
def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf      
    return lp + lnlike(theta)



#initial positions for the parameters close to the peak of the likelihood (some of the ranges defined by CosmoMC)
ombh2_center, ombh2_width = 0.0221, 0.0001
omch2_center, omch2_width = 0.12, 0.001
H0_center, H0_width = 67.31, 2.0
alpha_center, alpha_width = 0.135, 0.02
beta_center, beta_width = 3.0, 0.3
mag_center, mag_width = -19.05, 0.15
delM_center, delM_width = -0.05, 0.1
w1_center, w1_width = -1.0, 2.0
w2_center, w2_width = -1.0, 2.0
w3_center, w3_width = -1.0, 3.0
w4_center, w4_width = -1.0, 3.0
w5_center, w5_width = -1.0, 3.0
w6_center, w6_width = -1.0, 3.0


ombh2_0 = np.random.uniform((ombh2_center-ombh2_width), (ombh2_center+ombh2_width), nwalkers)
omch2_0 = np.random.uniform((omch2_center-omch2_width), (omch2_center+omch2_width), nwalkers)
H0_0 = np.random.uniform((H0_center-H0_width), (H0_center+H0_width), nwalkers)
alpha_0 = np.random.uniform((alpha_center-alpha_width), (alpha_center+alpha_width), nwalkers)
beta_0 = np.random.uniform((beta_center-beta_width), (beta_center+beta_width), nwalkers)
mag_0 = np.random.uniform((mag_center-mag_width), (mag_center+mag_width), nwalkers)
delM_0 = np.random.uniform((delM_center-delM_width), (delM_center+delM_width), nwalkers)
w1_0 = np.random.uniform((w1_center-w1_width), (w1_center+w1_width), nwalkers)
w2_0 = np.random.uniform((w2_center-w2_width), (w2_center+w2_width), nwalkers)
w3_0 = np.random.uniform((w3_center-w3_width), (w3_center+w3_width), nwalkers)
w4_0 = np.random.uniform((w4_center-w4_width), (w4_center+w4_width), nwalkers)
w5_0 = np.random.uniform((w5_center-w5_width), (w5_center+w5_width), nwalkers)
w6_0 = np.random.uniform((w6_center-w6_width), (w6_center+w6_width), nwalkers)



# initializing the walkers in a tiny ball around the maximum likelihood
par_0 = np.array([ombh2_0, omch2_0, H0_0, alpha_0, beta_0, mag_0, delM_0, w1_0, w2_0, w3_0, w4_0, w5_0, w6_0]).T


# initialize emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

'''
# Do burn-in
timei=time()
burn_steps = 400
pos, prob, state = sampler.run_mcmc(par_0, burn_steps)
timef=time()
delta_t=(timef-timei)/nwalkers/burn_steps



#Save the position of the walkers after the burn-in step and time for each MCMC point
f = open("outputs/pos_after_burn_in.txt", "w")
f.close()
f = open("outputs/pos_after_burn_in.txt", "a")
for k in range(nwalkers):
    f.write("  ".join(map(str, pos[k,])) + "\n") #write every line in txt file
f.close()
np.savetxt('outputs/delta_t.txt', np.array([delta_t]), fmt='%.6f')


'''
#Load the position of the walkers after the burn-in step and time for each MCMC point
pos = np.loadtxt("outputs/pos_after_burn_in.txt")
delta_t = np.loadtxt("outputs/delta_t.txt")


log_term = open("outputs/log_terminal.txt", "a")
print("Time for each MCMC point: %.5g" %delta_t + "\n")
log_term.write("Time for each MCMC point: %.5g" %delta_t + "\n")


#Repetitions of MCMC
repetitions = 30000
steps_per_repetition = 5
total_steps = repetitions*steps_per_repetition #this is total steps per walker!
dur_MCMC = delta_t*nwalkers*(steps_per_repetition + total_steps) #total duration time of MCMC

mm, ss = divmod(dur_MCMC, 60)
hh, mm = divmod(mm, 60)
print("Estimated duration of MCMC from now on: " + ("%d:%02d:%02d" % (hh, mm, ss)) + "\n")
log_term.write("Estimated duration of MCMC from now on: " + ("%d:%02d:%02d" % (hh, mm, ss)) + "\n")

print("Total points to be evaluated from now on: %d" %(nwalkers*(steps_per_repetition + total_steps)) + "\n")
log_term.write("Total points to be evaluated from now on: %d" %(nwalkers*(steps_per_repetition + total_steps)) + "\n" + "\n")
log_term.close()

#initial time to record MCMC complete time
timei = time()


# Start MC exploration
# Will save one chain to the file below:
f = open("outputs/chain.txt", "w")
f.close()


counter = 1 #reset counting execution for CAMB
sampler.reset()
posf, probf, statef = sampler.run_mcmc(pos, steps_per_repetition)
weights = sampler.flatlnprobability #evaluates the weights for every MCMC point
np.savetxt('outputs/weights.txt', weights, fmt='%1.8e') #saves weights array to txt

accept_num = steps_per_repetition*sampler.acceptance_fraction #array with number of steps accepted for each walker
total_num = steps_per_repetition*np.ones(nwalkers) #array with total number of steps for each walker
np.savetxt('outputs/accept_ratio.txt', accept_num/total_num, fmt='%1.8e') #saves acceptance ratio for every MCMC point

flat_sample = sampler.flatchain
f = open("outputs/chain.txt", "a")
for k in range(flat_sample.shape[0]):
    f.write("  ".join(map(str, flat_sample[k])) + "\n") #write every line in txt file
f.close()


#Saving derived parameters to txt (array is [omMatter, omLambda, chi_square])
der = open("outputs/derived_params.txt", "w")
der.close()
der = open("outputs/derived_params.txt", "a")
print('Begin of derived parameters...')
timei_d = time()
for k in range(flat_sample.shape[0]):
    omM = flat_sample[k][0]/(flat_sample[k][2]/100.0)**2 + flat_sample[k][1]/(flat_sample[k][2]/100.0)**2
    omL = 1 - omM    
    chi = -2.0*sampler.flatlnprobability[k]
    der.write("  ".join(map(str, np.array([omM, omL, chi]))) + "\n")
timef_d = time()
mm, ss = divmod((timef_d - timei_d), 60)
hh, mm = divmod(mm, 60)
print('End of derived parameters saving. Time spent: ' + ("%d:%02d:%02d" % (hh, mm, ss)) + '\n')
der.close()
log_term = open("outputs/log_terminal.txt", "a")
log_term.write('End of derived parameters saving. Time spent: ' + ("%d:%02d:%02d" % (hh, mm, ss)) + '\n')
log_term.close()



#GELMAN-RUBIN CONVERGENCE DIAGNOSTIC
cont_loops = 1 #counter of loops in the mcmc exploration
sum_params = np.zeros((nwalkers, ndim)) #total sum of the parameters
sum_params_partial = np.zeros((nwalkers, ndim)) #partial sum of the parameters
for m in range(ndim):
    i = 0
    for j in range(nwalkers):
        cont = 0
        while(cont < steps_per_repetition):            
            sum_params[j,m] += flat_sample[i,m] #sum of the parameters considering different walkers changing in flat_sample
            i += 1
            cont += 1
R_stat = open("outputs/R_statistic.txt", "w") #txt for saving R statistics vector from the convergence diagnostic
R_stat.close()


#Main MC exploration
for nstore in range(repetitions):
    sampler.reset()
    posf, probf, statef = sampler.run_mcmc(posf, steps_per_repetition)
    weights = np.append(weights, sampler.flatlnprobability)
    np.savetxt('outputs/weights.txt', weights, fmt='%1.8e') #saves weights array to txt

    accept_num += steps_per_repetition*sampler.acceptance_fraction #array with number of steps accepted for each walker
    total_num += steps_per_repetition*np.ones(nwalkers) #array with total number of steps for each walker
    np.savetxt('outputs/accept_ratio.txt', accept_num/total_num, fmt='%1.8e') #saves acceptance ratio for every MCMC point

    #Saves data to txt
    f = open("outputs/chain.txt", "a")
    this_sample = sampler.flatchain
    flat_sample = np.vstack((flat_sample,this_sample)) #joins previous chain with a new chain
    for k in range(this_sample.shape[0]):
        f.write("  ".join(map(str, this_sample[k])) + "\n")
    f.close()
    
    #Saving derived parameters to txt (array is [omMatter, omLambda, chi_square])
    der = open("outputs/derived_params.txt", "a")
    print('Begin of derived parameters...')
    timei_d = time()
    for k in range(this_sample.shape[0]):
        omM = this_sample[k][0]/(this_sample[k][2]/100.0)**2 + this_sample[k][1]/(this_sample[k][2]/100.0)**2
        omL = 1 - omM    
        chi = -2.0*sampler.flatlnprobability[k]
        der.write("  ".join(map(str, np.array([omM, omL, chi]))) + "\n")
    timef_d = time()
    mm, ss = divmod((timef_d - timei_d), 60)
    hh, mm = divmod(mm, 60)
    print('End of derived parameters saving. Time spent: ' + ("%d:%02d:%02d" % (hh, mm, ss)) + '\n')
    der.close()
    log_term = open("outputs/log_terminal.txt", "a")
    log_term.write('End of derived parameters saving. Time spent: ' + ("%d:%02d:%02d" % (hh, mm, ss)) + '\n')
    log_term.close() 
    
    
    #Saves the position of the walkers in txt after a number of repetitions
    pos_txt = open("outputs/pos_after_burn_in.txt", "w")
    pos_txt.close()
    pos_txt = open("outputs/pos_after_burn_in.txt", "a")
    for k in range(nwalkers):
        pos_txt.write("  ".join(map(str, posf[k,])) + "\n") #write every line in txt file
    pos_txt.close()
    
    
    #GELMAN-RUBIN CONVERGENCE DIAGNOSTIC
    cont_loops += 1 #counter of loops in the mcmc exploration
    sum_params_partial = np.zeros((nwalkers, ndim)) #partial sum of the parameters
    for m in range(ndim):
        i = 0
        for j in range(nwalkers):
            cont = 0
            while(cont < steps_per_repetition):            
                sum_params_partial[j,m] += this_sample[i,m] #sum of the parameters considering different walkers changing in flat_sample
                i += 1
                cont += 1
    sum_params = sum_params + sum_params_partial #accumulating the sum of the parameters
    mean_params = sum_params/(cont_loops*steps_per_repetition) #mean of the parameters considering all the chains
    total_mean = np.mean(mean_params, axis=0) #mean of mean_params, separated by parameters (axis=0 is mean on the columns)
    
    i = 0
    B = np.zeros(ndim)
    for i in range(ndim):
        B[i] = (1./(nwalkers-1.))*np.sum((mean_params[:,i]-total_mean[i])**2) #between-chain variance
    
    var_partial = np.zeros(ndim) #partial variance
    for m in range(ndim): #it varies the parameter being analyzed
        i = 0
        for k in range(cont_loops): #it varies the loops in mcmc exploration
            for j in range(nwalkers): #it varies the chains
                cont = 0
                while(cont < steps_per_repetition): #it varies each step inside each chain     
                    var_partial[m] += (flat_sample[i,m] - mean_params[j,m])**2 #sum of the partial variances
                    i += 1
                    cont += 1
    
    W = 1./(nwalkers*(cont_loops*steps_per_repetition-1.))*var_partial #mean within-chain variance
    
    V = (cont_loops*steps_per_repetition-1.)/(cont_loops*steps_per_repetition)*W + (nwalkers+1.)/nwalkers*B #pooled variance
    
    R_stat = open("outputs/R_statistic.txt", "a")    
    print('R = ' + ", ".join(str(x) for x in (np.sqrt(V/W))) + '\n') #potential scale reduction factor
    R_stat.write('R = ' + ", ".join(str(x) for x in (np.sqrt(V/W))) + '\n' + '\n')
    R_stat.close()
    log_term = open("outputs/log_terminal.txt", "a")
    log_term.write('R = ' + ", ".join(str(x) for x in (np.sqrt(V/W))) + '\n')
    log_term.close()



log_term = open("outputs/log_terminal.txt", "a")
mm, ss = divmod(time() - timei, 60)
hh, mm = divmod(mm, 60)
print("Real duration of MCMC: " + ("%d:%02d:%02d" % (hh, mm, ss)))
log_term.write("Real duration of MCMC: " + ("%d:%02d:%02d" % (hh, mm, ss)))
log_term.close()




