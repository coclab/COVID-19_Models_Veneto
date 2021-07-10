# Train ABmodel parameters on Veneto data and plot fitted curves


###########################################################################
# import libraries
import time, enum, math
import numpy as np
import pandas as pd
import random
import networkx as nx
import scipy.optimize
import pylab as plt

from multiprocessing import Pool # multiprocess 

###########################################################################
# import custom functions
import ABmodel 
#from ABmodel_utils import *

############################################################################
# import real observations
real_observations = pd.read_csv( 'Real_curves_dayly.csv', index_col = 0 ) # DAYLY

#import fitted parameters det model
det_params = pd.read_csv( 'param_estimates_DET_8P.csv', index_col = 1 ) 
det_params = det_params.iloc[:,1:]

#############################################################################

# optimization and simulation global settings

n_sim = 10                # number of model simulations
N_abm = 20000             # number of agents
num_nodes_E = 1           # number of agents in E
num_nodes_Ias = 1         # number of agents in Ias
num_nodes_Isy = 1         # number of agents in Isy
num_nodes_Iho = 0         # number of agents in Iho
num_nodes_Q = 0           # number of agents in Q
num_nodes_R = 0           # number of agents in R
num_nodes_D = 0           # number of agents In D

col_to_consider = ['INFECTED_HO', 'DECEASED', 'QUARANTINED', 'POSITIVE']

# max number of iterations of opt. algorith 
max_it = 250

# set period to fit (as string, from 1 to 8) 
p = '6'
  
#############################################################################
# define optimization problem and functions

def mse_on_single_array(v, w):
    """
    Mean squared error between two arrays.
    :param v: first array.
    :param w: second array.
    :return: the mean squared error between v and w.
    """
    if len(v) != len(w):
        raise ValueError("v and w must have the same length.")
    return np.square(np.subtract(v, w)).mean()
    

def weights_array(v, weight = 0.5):
    '''
    Given an array and a weight to be given for positive values, change array
    :param v -> Array that we want to weight
    :param weight -> A float in (0,1) indicating the weight to give to positive value'''
    v = v.astype(float)
    v = (v<0)*1 + (v>=0)*weight # overestimations are encouraged with smaller weight in the error function!
    return v
    
    
def mse_on_df(v, w, weighted = False, weights_comp = []):
    """
    Mean squared error between two dataframe, row by row.
    :param v: first dataframe.
    :param w: second dataframe.
    :return: the mean squared error between v and w given by the squared error between each couple of rows.
    """
    
    if not len(weights_comp): 
        weights_comp = v.shape[1]*[1/v.shape[1]]
        
    weights_comp = np.array(weights_comp)
    
    if v.shape != w.shape:
        raise ValueError("v and w must have the same shape.", v.shape, ' and ',w.shape, ' given')
    else:
        v = v.to_numpy()
        w = w.to_numpy()
        diff = v-w
        err_rel = diff/(w + 1*(w==0))
        
        for i in range(len(weights_comp)):
          err_rel[:,i] = err_rel[:,i]*weights_comp[i]
            
        if not weighted:
            return np.square(err_rel).mean()
        else:
            z = weights_array(err_rel)
            return np.sum(z * (err_rel)**2) / (np.sum(z) * (err_rel.size))


def _mseABM(target_profile, start, end, 
         known_pars,
         last_change_times,
         last_change_pars,
         col_to_consider = ['INFECTED_HO', 'DECEASED'],
         n_sim = n_sim,                # number of model simulations
         N_abm = N_abm,               # number of agents
         num_nodes_E = num_nodes_E,          # number of agents in E
         num_nodes_Ias = num_nodes_Ias,        # number of agents in Ias
         num_nodes_Isy = num_nodes_Isy,        # number of agents in Isy
         num_nodes_Iho = num_nodes_Iho,        # number of agents in Iho
         num_nodes_Q = num_nodes_Q,          # number of agents in Q
         num_nodes_R = num_nodes_R,          # number of agents in R
         num_nodes_D = num_nodes_D ):
    
    
        """
        Generate a MSE error with the inputted target profile.
        :param target_profile: the target time series of observed new cases.
        : ...
        :return: a callable function computing the mse between estimated and real observed new cases from the unknown parameters.
        """

        def target_series_mseABM(unknown_pars):                               
                             
            """
            MSE between target profile and the estimated profile made by estimated_observed_new_cases.
            :param unknown_pars: trainable parameters of the model function.
            :return: the mse between the estimated and target series.
            """
            
            new_change_times = last_change_times + [start]
            new_change_pars = last_change_pars + [list(unknown_pars) + list(known_pars)]
            days = end
            

            # multiple ABM simulation exploiting parallel computing
            
            cols = ['INFECTED_AS', 'INFECTED_SY', 'INFECTED_HO', 'DECEASED', 'QUARANTINED']

            sim_records = dict([(c, []) for c in cols])
            
            
            if __name__ == "__main__":
                with Pool(processes=n_sim) as pool: # number of process = number simulations (only when small!)
                    #apply_async
                    result_objects = [pool.apply_async(ABmodel.run_model, args=(N_abm, 
                                         num_nodes_E, num_nodes_Ias, num_nodes_Isy, num_nodes_Iho, num_nodes_Q, num_nodes_R, num_nodes_D, 
                                         days, new_change_times, new_change_pars, 'Solutions', False)) for run in range(n_sim)]
                    
                    # result_objects is a list of pool.ApplyResult objects
                    results = [r.get() for r in result_objects]
                
            for r in results:
                for c in cols:
                    rec = list(r[c].values)
                    sim_records[c].append(rec)  

            
            
            solutions = pd.DataFrame()
            for c in cols:
                solutions[c] = np.mean(sim_records[c], axis = 0) * (4900000/N_abm)
            
            
            solutions['POSITIVE'] = solutions['INFECTED_AS'] + solutions['INFECTED_SY'] + solutions['INFECTED_HO'] + solutions['QUARANTINED']
            
            w_dict = {'DECEASED': 1, 'INFECTED_HO' : 1, 'QUARANTINED': 0.05, 'POSITIVE': 0.01}
            
            
            MSE_rel = mse_on_df(solutions.iloc[start:end, :][col_to_consider], 
                                target_profile.iloc[start:end, :][col_to_consider],
                                weighted = True, 
                                weights_comp = [w_dict[c] for c in col_to_consider])
            
            return MSE_rel
         
        return target_series_mseABM
    
   



def fit_parametersABM(
            last_change_times,
            last_change_pars,
            unknown_pars_guess,
            known_pars,
            bounds,
            observed_cases,
            start, 
            end,
            col_to_consider = ['INFECTED_HO', 'DECEASED'], # QUARANTINED
            tol = 0.000001,
            max_it = max_it,
            verbose = True
            ):
        """
        Estimate the optimal parameters minimizing the mse with numerical optimization.
        :return: The estimated parameters in a scipy OptimizeResult object.
        """

        # set bounds and constraints
        lower_bounds = [ b[0] for b in bounds ]
        upper_bounds = [ b[1] for b in bounds ]
            
        id_mat = np.identity(len(upper_bounds)).astype(int) # identity matrix


        
        opt_param = scipy.optimize.minimize(fun = _mseABM(observed_cases, start, end,
                                                             known_pars,
                                                             last_change_times,
                                                             last_change_pars,
                                                             col_to_consider), 
                                            x0 = unknown_pars_guess, 
                                            method='COBYLA',  # https://handwiki.org/wiki/COBYLA
                                            # bounds=scipy.optimize.Bounds(lower_bounds, upper_bounds), # not supported in COBYLA
                                            constraints = scipy.optimize.LinearConstraint( id_mat, lower_bounds, upper_bounds),
                                            tol=tol, 
                                            options={'rhobeg': 1.0,
                                                      'maxiter': max_it,
                                                      'disp': verbose, 
                                                      'catol': 0.0002})
        '''
        # optimizer variant
        opt_param = scipy.optimize.minimize(fun = _mseABM(observed_cases, start, end,
                                                             known_pars,
                                                             last_change_times,
                                                             last_change_pars,
                                                             col_to_consider), 
                                            x0 = unknown_pars_guess, 
                                            method='trust-constr', 
                                            bounds=scipy.optimize.Bounds(lower_bounds, upper_bounds), 
                                            # constraints = ()
                                            tol=tol, 
                                            options={'xtol': 1e-08, 
                                                     'gtol': 1e-08, 
                                                     'barrier_tol': 1e-08, 
                                                     'maxiter': max_it, 
                                                     'verbose': 1, 
                                                     'disp': verbose})
        '''
        

        return opt_param
    
    

#############################################################################

# set start - end days
start = int(det_params.loc['start', 'Period_'+p])
end = int(det_params.loc['end', 'Period_'+p])
days = end



# guesses from last period fitting and DET model fitting

last_change_times = []
last_change_pars = []

if p == '1': # fitting PERIOD 1

    # create DF where save fitting results
    
    abm_params = pd.DataFrame()
    abm_params['coeff_name'] = ['start','end',
                               'ptrans_community_as', 'ptrans_other_as', 'ptrans_house_as', 'ptrans_work_as', 'ptrans_work_colleague_as', 'ptrans_school_as', 'ptrans_school_class_as', 'ptrans_school_friend_as', 'ptrans_community_sy', 'ptrans_other_sy', 'ptrans_house_sy', 
                                'ptrans_work_sy', 'ptrans_work_colleague_sy', 'ptrans_school_sy', 'ptrans_school_class_sy', 'ptrans_school_friend_sy', 'ptrans_quarantine', 'ptrans_ho']
    
    # observed (DET fitting)
    # probability to be quarantined 
    pquar_as = float(det_params.loc['pquar_as', 'Period_'+p])
    pquar_sy = float(det_params.loc['pquar_sy', 'Period_'+p])
    
    # days of quartine (by law)
    quarantine_time = 14
    
    #hospitalization rate (days from syntomatic to hospitalization)
    time_to_hospitalization = 1/float(det_params.loc['psi', 'Period_'+p])
    
    # death rate (days from infection to death)
    time_to_death = 1/float(det_params.loc['mu', 'Period_'+p])
    
    # quarantined rate (days from infection to quarantine)
    time_to_quarantine_as = 1/float(det_params.loc['phi_as', 'Period_'+p])
    time_to_quarantine_sy = 1/float(det_params.loc['phi_sy', 'Period_'+p])
    
    
    # initial guesses
    ptrans_community_as= 0.005
    ptrans_other_as = 0.007
    ptrans_house_as = 0.01
    ptrans_work_as = 0.01
    ptrans_work_colleague_as = 0.03
    ptrans_school_as = 0.007
    ptrans_school_class_as = 0.01
    ptrans_school_friend_as = 0.03
    
    # transmission probability Symptomatic
    ptrans_community_sy = 0.005
    ptrans_other_sy = 0.007
    ptrans_house_sy = 0.01
    ptrans_work_sy = 0.01
    ptrans_work_colleague_sy = 0.03
    ptrans_school_sy = 0.007
    ptrans_school_class_sy = 0.01
    ptrans_school_friend_sy = 0.03
    
    # transmission probability Quarantined
    ptrans_quarantine = 0.0001
    
    # transmission probability Hospitalized
    ptrans_ho = 0.0001
    
    
    unknown_pars_guess = ptrans_community_as, ptrans_other_as, ptrans_house_as, ptrans_work_as, ptrans_work_colleague_as, ptrans_school_as, ptrans_school_class_as, ptrans_school_friend_as, ptrans_community_sy, ptrans_other_sy, ptrans_house_sy, ptrans_work_sy, ptrans_work_colleague_sy, ptrans_school_sy, ptrans_school_class_sy, ptrans_school_friend_sy, ptrans_quarantine, ptrans_ho
    known_pars = pquar_as, pquar_sy, quarantine_time, time_to_hospitalization, time_to_death, time_to_quarantine_as, time_to_quarantine_sy 


else: # fitting PERIOD > 1

    abm_params = pd.read_csv( 'param_estimates_ABM_8P.csv', index_col = 0 ) 
    
    abm_params.iloc[:2,1:] = abm_params.iloc[:2,1:].astype(int)
    abm_params.iloc[2:,1:] = abm_params.iloc[2:,1:].astype(float)
    
    
    for i in range(1, int(p)+1):
        
    
        # observed (DET fitting)
        
        # probability to be quarantined 
        pquar_as = float(det_params.loc['pquar_as', 'Period_'+str(i)])
        pquar_sy = float(det_params.loc['pquar_sy', 'Period_'+str(i)])
        
        # days of quartine (by law)
        quarantine_time = 14
        
        #hospitalization rate (days from syntomatic to hospitalization)
        time_to_hospitalization = 1/float(det_params.loc['psi', 'Period_'+str(i)])
        
        # death rate (days from infection to death)
        time_to_death = 1/float(det_params.loc['mu', 'Period_'+str(i)])
        
        # quarantined rate (days from infection to quarantine)
        time_to_quarantine_as = 1/float(det_params.loc['phi_as', 'Period_'+str(i)])
        
        time_to_quarantine_sy = 1/float(det_params.loc['phi_sy', 'Period_'+str(i)])
        
        known_pars = pquar_as, pquar_sy, quarantine_time, time_to_hospitalization, time_to_death, time_to_quarantine_as, time_to_quarantine_sy 
    
        
        if i < int(p):
                
            # fitted parameters last periods        
            last_start, last_end, ptrans_community_as, ptrans_other_as, ptrans_house_as, ptrans_work_as, ptrans_work_colleague_as, ptrans_school_as, ptrans_school_class_as, ptrans_school_friend_as, ptrans_community_sy, ptrans_other_sy, ptrans_house_sy, ptrans_work_sy, ptrans_work_colleague_sy, ptrans_school_sy, ptrans_school_class_sy, ptrans_school_friend_sy, ptrans_quarantine, ptrans_ho = abm_params.loc[:, 'Period_'+str(i)].values
            
            unknown_pars_guess = ptrans_community_as, ptrans_other_as, ptrans_house_as, ptrans_work_as, ptrans_work_colleague_as, ptrans_school_as, ptrans_school_class_as, ptrans_school_friend_as, ptrans_community_sy, ptrans_other_sy, ptrans_house_sy, ptrans_work_sy, ptrans_work_colleague_sy, ptrans_school_sy, ptrans_school_class_sy, ptrans_school_friend_sy, ptrans_quarantine, ptrans_ho
        
            last_change_times.append(last_start)
            
            last_change_pars = last_change_pars + [list(unknown_pars_guess) + list(known_pars)]
            
    

# set bound ( recall we want to train probabilities!)
if p in ['2', '3', '4','7']:
    # closed school
    bounds = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,0.00000001),(0,0.00000001),(0,0.00000001),
	      (0,1),(0,1),(0,1),(0,1),(0,1),(0,0.00000001),(0,0.00000001),(0,0.00000001),(0,1),(0,1)]

else:
    # loose bound 
    bounds = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),
              (0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]




# solo per 6

last_start, last_end, ptrans_community_as, ptrans_other_as, ptrans_house_as, ptrans_work_as, ptrans_work_colleague_as, ptrans_school_as, ptrans_school_class_as, ptrans_school_friend_as, ptrans_community_sy, ptrans_other_sy, ptrans_house_sy, ptrans_work_sy, ptrans_work_colleague_sy, ptrans_school_sy, ptrans_school_class_sy, ptrans_school_friend_sy, ptrans_quarantine, ptrans_ho = abm_params.loc[:, 'Period_'+str(2)].values

unknown_pars_guess = ptrans_community_as, ptrans_other_as, ptrans_house_as, ptrans_work_as, ptrans_work_colleague_as, ptrans_school_as, ptrans_school_class_as, ptrans_school_friend_as, ptrans_community_sy, ptrans_other_sy, ptrans_house_sy, ptrans_work_sy, ptrans_work_colleague_sy, ptrans_school_sy, ptrans_school_class_sy, ptrans_school_friend_sy, ptrans_quarantine, ptrans_ho



# optimize 

t0 = time.time()
opt_param = fit_parametersABM(last_change_times = last_change_times,
            last_change_pars = last_change_pars,
            unknown_pars_guess = unknown_pars_guess,
            known_pars = known_pars,
            observed_cases = real_observations,
            start = start, 
            end = end,
            col_to_consider = col_to_consider, 
            bounds = bounds)
t1 = time.time()
print('\nTraining time: ',t1 - t0, ' seconds')

unknown_pars_opt = opt_param['x']

# save optimal parameters
abm_params['Period_'+p] = [start, end] + list(unknown_pars_opt)
abm_params.to_csv('param_estimates_ABM_8P.csv')



# inspect solution
print('\nOptimal parameters: ', unknown_pars_opt)

# plot 

new_change_times = last_change_times + [start]
new_change_pars = last_change_pars + [list(unknown_pars_opt) + list(known_pars)]


cols = ['INFECTED_AS', 'INFECTED_SY', 'INFECTED_HO', 'DECEASED', 'QUARANTINED', 'POSITIVE']

sim_records = dict([(c, []) for c in cols])

# parallel computing simulation with optimized pars
if __name__ == "__main__":
    with Pool(processes=n_sim) as pool: # number of process = number simulations (only when small!)
        #apply_async
        result_objects = [pool.apply_async(ABmodel.run_model, args=(N_abm, 
                             num_nodes_E, num_nodes_Ias, num_nodes_Isy, num_nodes_Iho, num_nodes_Q, num_nodes_R, num_nodes_D, 
                             days, new_change_times, new_change_pars, 'Solutions', False)) for run in range(n_sim)]
        
        # result_objects is a list of pool.ApplyResult objects
        results = [r.get() for r in result_objects]
        
for r in results:
    r['POSITIVE'] = r['INFECTED_AS'] + r['INFECTED_SY'] + r['INFECTED_HO'] + r['QUARANTINED']
    
    for c in cols:
        rec = list(r[c].values)
        sim_records[c].append(rec)
            
            
               
real_sub = real_observations.iloc[1:end, :]    

for c in col_to_consider:
    plt.figure(figsize=(8,5))
    plt.plot(range(1,end),real_sub[c].values, label = c)
    ma = np.mean(sim_records[c] , axis = 0) * (4900000/N_abm)
    mstd = np.std(sim_records[c], axis = 0) * (4900000/N_abm)
    ma = ma[1:end]
    mstd = mstd[1:end]
    plt.plot(range(1,end), ma, label = c+'_EST')
    # approx CI for n = 10, t = 2.228
    plt.fill_between(range(1,end), ma - (2.228/(n_sim**0.5)) * mstd, ma + (2.228/(n_sim**0.5)) * mstd, color = 'orange', alpha=0.2);


    plt.xlabel('days')
    plt.ylabel('cases')
    plt.legend()
    
    for t in new_change_times:
        if t>1:
            plt.axvline(t, color="grey", linestyle="--", lw = 0.7)
            
    plt.savefig('ABM_fit_'+c+p+'.png')