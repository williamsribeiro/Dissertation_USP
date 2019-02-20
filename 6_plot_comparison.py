import numpy as np
import scipy as sc
from matplotlib import pyplot as pl
from getdist import MCSamples, plots

#RUN WITH PYTHON3!

#Read chain from txt and selects columns of interest
all_data = sc.genfromtxt('chain_uncorrelated_all_data.txt', usecols = (6,7,8,9,10,11))
planck = sc.genfromtxt('chain_uncorrelated_planck.txt', usecols = (6,7,8,9,10,11))

#-------------------------------------------------------- PLOTS ----------------------------------------------------------------------
#Specify parameter names and labels
names = ['w1','w2','w3','w4','w5','w6']
labels = ['w_1','w_2','w_3','w_4','w_5','w_6']
samples1 = MCSamples(samples = planck, names = names, labels = labels, label='CMB')
samples2 = MCSamples(samples = all_data, names = names, labels = labels, label='BAO+CC+CMB+SN')

#Triangle plot
g = plots.getSubplotPlotter()
g.settings.axes_fontsize = 13 #size of numbers in axes
g.settings.legend_fontsize = 30 #size of legend
g.settings.lab_fontsize = 30 #size of parameter labels
g.settings.tight_gap_fraction = 0.04 #plot scale, standard=0.13
g.triangle_plot([samples1, samples2], filled=True, colors=['green','blue'], line_args=[{'color':'green'},{'color':'blue'}], legend_loc = 'upper right')

pl.savefig('comparison_wis.png')

#Print size of chain
print(len(all_data[:,0]))
print(len(planck[:,0]))


