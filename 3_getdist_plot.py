import numpy as np
import scipy as sc
from matplotlib import pyplot as pl
from getdist import MCSamples, plots

#RUN WITH PYTHON3!

#Read chain from txt and selects columns of interest
chain = sc.genfromtxt('chain_uncorrelated.txt')
chain_cosmo = chain[:,[0,1,2,3,4,5]]
chain_wis = chain[:,[6,7,8,9,10,11]]

#-------------------------------------------------------- PLOTS ----------------------------------------------------------------------
#Specify parameter names and labels
names = ['ombh2','omch2','ns','As','H0','tau']
labels = ["\Omega_bh^2", "\Omega_ch^2", "n_s", "\ln{(10^{10}A_s)}", "H_0", r"\tau"]
samples = MCSamples(samples=chain_cosmo, names = names, labels = labels, label='BAO+CC+CMB+SN')

#Triangle plot
g = plots.getSubplotPlotter()
g.settings.axes_fontsize = 13 #size of numbers in axes
g.settings.legend_fontsize = 30 #size of legend
g.settings.lab_fontsize = 20 #size of parameter labels
g.triangle_plot([samples], filled=True, colors=['blue'], line_args=[{'color':'blue'}], legend_loc = 'upper right')
pl.savefig('BAO_CC_CMB_SN_cosmo.png')

#Specify parameter names and labels
names = ['w1','w2','w3','w4','w5','w6']
labels = ['w_1','w_2','w_3','w_4','w_5','w_6']
samples = MCSamples(samples=chain_wis, names = names, labels = labels, label='BAO+CC+CMB+SN')

#Triangle plot
g = plots.getSubplotPlotter()
g.settings.axes_fontsize = 13 #size of numbers in axes
g.settings.legend_fontsize = 30 #size of legend
g.settings.lab_fontsize = 30 #size of parameter labels
g.triangle_plot([samples], filled=True, colors=['blue'], line_args=[{'color':'blue'}], legend_loc = 'upper right')
pl.savefig('BAO_CC_CMB_SN_wis.png')


#---------------------------------------------------- PARAMETER ESTIMATION------------------------------------------------------------
#Specify parameter names and labels
names = ['ombh2','omch2','ns','As','H0','tau']
labels = ["\Omega_bh^2", "\Omega_ch^2", "n_s", "\ln{(10^{10}A_s)}", "H_0", r"\tau"]
samples = MCSamples(samples=chain_cosmo, names = names, labels = labels, label='BAO+CC+CMB+SN')

#Return values from chain (limit defines confidence levels: 1-sigma=1, 2-sigma=2, ...)
print(samples.getInlineLatex('ombh2', limit=1)+"\\\\")
print(samples.getInlineLatex('ombh2', limit=2)+"\\\\")
print(samples.getInlineLatex('omch2', limit=1)+"\\\\")
print(samples.getInlineLatex('omch2', limit=2)+"\\\\")
print(samples.getInlineLatex('ns', limit=1)+"\\\\")
print(samples.getInlineLatex('ns', limit=2)+"\\\\")
print(samples.getInlineLatex('As', limit=1)+"\\\\")
print(samples.getInlineLatex('As', limit=2)+"\\\\")
print(samples.getInlineLatex('H0', limit=1)+"\\\\")
print(samples.getInlineLatex('H0', limit=2)+"\\\\")
print(samples.getInlineLatex('tau', limit=1)+"\\\\")
print(samples.getInlineLatex('tau', limit=2)+"\\\\")

#Specify parameter names and labels
names = ['w1','w2','w3','w4','w5','w6']
labels = ['w_1','w_2','w_3','w_4','w_5','w_6']
samples = MCSamples(samples=chain_wis, names = names, labels = labels, label='BAO+CC+CMB+SN')

#Return values from chain (limit defines confidence levels: 1-sigma=1, 2-sigma=2, ...)
print(samples.getInlineLatex('w1', limit=1)+"\\\\")
print(samples.getInlineLatex('w1', limit=2)+"\\\\")
print(samples.getInlineLatex('w2', limit=1)+"\\\\")
print(samples.getInlineLatex('w2', limit=2)+"\\\\")
print(samples.getInlineLatex('w3', limit=1)+"\\\\")
print(samples.getInlineLatex('w3', limit=2)+"\\\\")
print(samples.getInlineLatex('w4', limit=1)+"\\\\")
print(samples.getInlineLatex('w4', limit=2)+"\\\\")
print(samples.getInlineLatex('w5', limit=1)+"\\\\")
print(samples.getInlineLatex('w5', limit=2)+"\\\\")
print(samples.getInlineLatex('w6', limit=1)+"\\\\")
print(samples.getInlineLatex('w6', limit=2))


#Print size of chain
print(len(chain[:,0]))




