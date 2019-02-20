import numpy as np
import scipy as sc
from matplotlib import pyplot as pl
from scipy.interpolate import CubicSpline

#RUN WITH PYTHON3!

#Data
z = np.arange(0.001,2.0,0.001)
zi = np.array([0.0, 0.25, 0.50, 0.85, 1.25, 2.0])

wi = np.array([-1.12, -0.993, -1.33, -0.78, -1.55, -1.7])

wup_1sig = np.array([0.2, 0.072, 0.25, 0.25, 0.71, 1.6])
wdown_1sig = np.array([0.2, 0.072, 0.25, 0.17, 0.49, 1.0])

wup_2sig = np.array([0.39, 0.14, 0.50, 0.41, 1.1, 2.3])
wdown_2sig = np.array([0.40, 0.14, 0.48, 0.46, 1.2, 2.8])


####################################################################################################################################
#PLOT OF THE EQUATION OF STATE EVOLUTION

#Cubic spline interpolation
w = CubicSpline(zi, wi)

#trick for plotting asymmetric error bars
err_w_1sig = sc.array(sc.ones(len(wup_1sig)*2).reshape(len(wup_1sig),2))
err_w_1sig[:,0] = wdown_1sig
err_w_1sig[:,1] = wup_1sig
err_w_2sig = sc.array(sc.ones(len(wup_2sig)*2).reshape(len(wup_2sig),2))
err_w_2sig[:,0] = wdown_2sig
err_w_2sig[:,1] = wup_2sig

#Plotting
pl.figure()
pl.xlabel(r'$z$', fontsize=30)
pl.ylabel(r'$w(z)$', fontsize=30)
pl.axhline(y=-1, color='k', linestyle='--')
pl.plot(z, w(z), '-', color='k', label='Spline')
pl.plot(zi, wi, '.', color='k')
pl.errorbar(zi, wi, yerr = err_w_1sig.T, fmt='None', ecolor='blue', label=r'$1\sigma$')
pl.errorbar((zi+0.01), wi, yerr = err_w_2sig.T, fmt='None', ecolor='red', label=r'$2\sigma$')
pl.xlim(-0.05, 2.1)
pl.ylim(-5.0, 1.5)
pl.legend(loc='lower left', shadow=True, fontsize=18, ncol=1)
pl.tick_params(labelsize=15) #size of the numbers in the graphic
pl.savefig('w_plot_all_data.png', bbox_inches = "tight")
pl.close()



