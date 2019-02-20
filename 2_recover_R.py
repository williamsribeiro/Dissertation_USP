import numpy as np
import scipy as sc
from time import time

#Data
chain2 = sc.genfromtxt('/home/williams/Downloads/Dissertation/Final_Results/all_data/outputs2/chain.txt')
chain3 = sc.genfromtxt('/home/williams/Downloads/Dissertation/Final_Results/all_data/outputs3/chain.txt')
chain4 = sc.genfromtxt('/home/williams/Downloads/Dissertation/Final_Results/all_data/outputs4/chain.txt')
chain5 = sc.genfromtxt('/home/williams/Downloads/Dissertation/Final_Results/all_data/outputs5/chain.txt')
chain = np.vstack((chain3,chain4))
chain = np.vstack((chain2,chain))
chain = np.vstack((chain,chain5))
print(len(chain[:,0]))

ndim = 31 #parameter space dimension
nwalkers = 2*ndim #number of walkers
steps_per_repetition = 2 #number of steps at each bloc in the original code
bloc = nwalkers*steps_per_repetition #number of rows to select in bloc in the chain

flat_sample = chain[:bloc,:] #selecting the first bloc rows
last_row = bloc-1

start = time()

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

R_number = 6867 #number of times R is evaluated in the txt

#while (np.any(last_row) == True):
while (cont_loops <= R_number):
    last_row += bloc #increments rows in blocs
    
    this_sample = chain[(last_row+1-bloc):(last_row+1),:]
    
    flat_sample = np.vstack((flat_sample,this_sample)) #joins previous chain with a new chain
    
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
    
    if (cont_loops % 20 == 0):
        print('Bloc = %d' %(cont_loops))
        mm, ss = divmod((time()-start), 60)
        hh, mm = divmod(mm, 60)
        print('Time = ' + ("%d:%02d:%02d" % (hh, mm, ss)) + "\n")




print('R = ' + ", ".join(str(x) for x in (np.sqrt(V/W))) + '\n') #potential scale reduction factor


