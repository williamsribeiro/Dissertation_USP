import numpy as np
import scipy as sc
from matplotlib import pyplot as pl
from getdist import MCSamples, plots

#RUN WITH PYTHON3!

#Read txt's and selects columns of interest
chain = sc.genfromtxt('chain_uncorrelated.txt')
chain_cosmo = chain[:,[0,1,2,3,4,5]]
chain_wis = chain[:,[6,7,8,9,10,11]]
deriv_param = sc.genfromtxt('deriv_param.txt')

#---------------------------------------------------- BEST FIT VALUES -----------------------------------------------------------------
minimum_chi = np.argmin(deriv_param[:,2])
print('Best fit cosmos: ', chain_cosmo[minimum_chi,:])
print('Best fit wis: ', chain_wis[minimum_chi,:])
print('Best fit derived params: ', deriv_param[minimum_chi,:])

#---------------------------------------------- ESTIMATION OF DERIVED PARAMETERS -----------------------------------------------------
#Specify parameter names and labels
names = ['omL','omM','chi']
labels = ["\Omega_\Lambda", "\Omega_m", "chi"]
samples = MCSamples(samples=deriv_param, names = names, labels = labels)

#Return values from chain (limit defines confidence levels: 1-sigma=1, 2-sigma=2, ...)
print(samples.getInlineLatex('omL', limit=1)+"\\\\")
print(samples.getInlineLatex('omL', limit=2)+"\\\\")
print(samples.getInlineLatex('omM', limit=1)+"\\\\")
print(samples.getInlineLatex('omM', limit=2)+"\\\\")
print(samples.getInlineLatex('chi', limit=1)+"\\\\")
print(samples.getInlineLatex('chi', limit=2)+"\\\\")

#Print size of chain
print(len(chain[:,0]))

