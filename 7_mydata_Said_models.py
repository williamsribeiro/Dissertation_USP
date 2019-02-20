import numpy as np
import scipy as sc
from matplotlib import pyplot as pl
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint

#RUN WITH PYTHON3!

#Data
zi = np.array([0.0, 0.25, 0.50, 0.85, 1.25, 2.0])

wi_mine = np.array([-1.12, -0.993, -1.33, -0.78, -1.55, -1.7]) #my results from all_data
wup_mine = np.array([0.39, 0.14, 0.50, 0.41, 1.1, 2.3]) #my confidence levels from all_data at 95 per cent
wdown_mine = np.array([0.40, 0.14, 0.48, 0.46, 1.2, 2.8]) #my confidence levels from all_data at 95 per cent
omM, wgamma, h = 0.3, 0.00002469, 0.689 #my results from all_data, except for wgamma
omR = wgamma*(1.+3.04*(7./8.)*(4./11.)**(4./3.))/(h**2)

#Data from paper
wi_najla = np.array([-1.12, -1.16, -1.01, -1.01, -1.21, -1.2]) #results from Najla Said paper
wup_najla = np.array([0.23, 0.17, 0.29, 0.53, 0.87, 1.1]) #Najla Said confidence levels at 95 per cent
wdown_najla = np.array([0.25, 0.18, 0.36, 0.82, 1.12, 1.5]) #Najla Said confidence levels at 95 per cent

####################################################################################################################################
#MODELS TO TEST

#Covariant Galileon
z = np.arange(0.0001,2.2,0.0001)
a = ((1-omM-omR)/(omR)**2)*(1./(1.+z))**8
b = 1.+(1./(1.+z))*omM/omR
c = -1.
omR = (-b + np.sqrt(b**2 - 4.*a*c))/(2.*a)
r2 = a*omR**2
cov_Gal = -(6.+omR)/(3.*(1+r2))




#Quintessence inverse power-law potential (Amendola, p.145)
def xis_lambda(y, N): #ODE to be solved
    x1, x2, x3, lamb = y
    dydt = np.zeros(4)
    dydt[0] = -3.*x1 + np.sqrt(6)/2.*lamb*x2**2 + 0.5*x1*(3.+3*x1**2-3*x2**2+x3**2)
    dydt[1] = -np.sqrt(6)/2.*lamb*x1*x2 + 0.5*x2*(3.+3*x1**2-3*x2**2+x3**2)
    dydt[2] = -2*x3 + 0.5*x3*(3.+3*x1**2-3*x2**2+x3**2)
    dydt[3] = -np.sqrt(6)*lamb**2/n*x1
    return dydt

n = 1 #parameter from potential
y0 = np.array([5.0e-5, 1.0e-8, 0.9999, 1.0e9]) #initial conditions in log10(1+z)=7.21
N = np.linspace(-np.log(10**(7.21)), -np.log(10**(0.001)), 10001) #integration steps and range in N=ln(a)
z_quint = np.exp(-N)-1. #transformation from N to z
sol = odeint(xis_lambda, y0, N) #solutions for x1, x2, x3, lambda
quintessence = (sol[:,0]**2-sol[:,1]**2)/(sol[:,0]**2+sol[:,1]**2) #equation of state evaluation (7.17 in Amendola)




#k-essence: dilatonic ghost condensate model (Amendola, p.181)
def xis(y, N): #ODE to be solved
    x1, x2, x3 = y
    dydt = np.zeros(3)
    dydt[0] = -x1*(6.*(2.*x2-1.)+3.*np.sqrt(6)*lamb*x1*x2)/(2.*(6*x2-1)) + x1/2.*(3.-3.*x1**2+3.*x1**2*x2+x3**2)
    dydt[1] = x2*(3.*x2*(4.-np.sqrt(6)*lamb*x1)-np.sqrt(6)*(np.sqrt(6)-lamb*x1))/(1.-6.*x2)
    dydt[2] = x3/2.*(-1.-3.*x1**2+3.*x1**2*x2+x3**2)
    return dydt

lamb = 0.2 #parameter from action
y0 = np.array([6.0e-11, (1.0e-9+0.5), 0.999]) #initial conditions in log10(1+z)=6.218
N = np.linspace(-np.log(10**(6.218)), -np.log(10**(0.001)), 10001) #integration steps and range in N=ln(a)
z_ghost = np.exp(-N)-1. #transformation from N to z
sol = odeint(xis, y0, N) #solutions for x1, x2, x3
ghost = (1.-sol[:,1])/(1.-3*sol[:,1]) #equation of state evaluation (8.42 in Amendola)



####################################################################################################################################
#PLOT OF THE EQUATION OF STATE EVOLUTION

#trick for plotting asymmetric error bars
err_w_mine = sc.array(sc.ones(len(wup_mine)*2).reshape(len(wup_mine),2))
err_w_mine[:,0] = wdown_mine
err_w_mine[:,1] = wup_mine
err_w_najla = sc.array(sc.ones(len(wup_najla)*2).reshape(len(wup_najla),2))
err_w_najla[:,0] = wdown_najla
err_w_najla[:,1] = wup_najla

#Plotting
pl.figure()
pl.xlabel(r'$z$', fontsize=30)
pl.ylabel(r'$w(z)$', fontsize=30)
pl.plot(zi, wi_mine, '.', color='k', label='This work')
pl.errorbar(zi, wi_mine, yerr = err_w_mine.T, fmt='None', ecolor='black')
pl.plot((zi+0.015), wi_najla, '.', color='b', label='Said $\it{et}$ $\it{al.}$')
pl.axhline(y=-1, color='k', linestyle='--', label='$\Lambda$CDM')
pl.errorbar((zi+0.015), wi_najla, yerr = err_w_najla.T, fmt='None', ecolor='blue')
pl.plot(z, cov_Gal, color='green', linestyle='-', label='Covariant Galileon')
pl.plot(z_quint[::-1], quintessence[::-1], color='red', linestyle='-', label='Tracking quintessence') #interved arrays
pl.plot(z_ghost[::-1], ghost[::-1], color='magenta', linestyle='-', label='Ghost k-essence') #invert arrays for plotting
pl.xlim(-0.05, 2.1)
#pl.ylim(-3.5, 0.5)
pl.legend(loc='lower left', shadow=True, fontsize=15)
pl.tick_params(labelsize=15) #size of the numbers in the graphic
pl.savefig('w_plot_Said_models.png', bbox_inches = "tight")
pl.close()



