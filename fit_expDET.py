# Train DETmodel parameters on Veneto data, plot fitted curves and experiment prediction


# import libraries
import numpy as np
import pandas as pd
import pylab as plt
from typing import Any, Callable, Tuple
import scipy.optimize
from scipy.optimize import OptimizeResult, Bounds, LinearConstraint

###########################################################################
# import custom functions

from DETmodel import main

###########################################################################
# import real observations

real_observations = pd.read_csv( 'Real_curves_dayly.csv', index_col = 0 ) # DAYLY

###########################################################################
# define optimization problem and functions

def mse_on_single_array(v: np.ndarray, w: np.ndarray) -> float:
    """
    Mean squared error between two arrays.
    :param v: first array.
    :param w: second array.
    :return: the mean squared error between v and w.
    """
    if len(v) != len(w):
        raise ValueError("v and w must have the same length.")
    return np.square(np.subtract(v, w)).mean()
    
def weights_array(v : np.array, weight : float = 0.5) -> np.array:
    '''
    Given an array and a weight to be given for positive values, change array
    :param v -> Array that we want to weight
    :param weight -> A float in (0,1) indicating the weight to give to positive value'''
    v = v.astype(float)
    v = (v<0)*1 + (v>=0)*weight # sovrastime pesano meno in errore!
    return v
    
    
def mse_on_df(v: pd.DataFrame, w: pd.DataFrame, weighted : bool = False, weights_comp: list = []) -> float:
    """
    Mean squared error between two dataframe, row by row.
    :param v: first dataframe.
    :param w: second dataframe.
    :return: the mean squared error between v and w given by the squared error between each couple of rows.
    """
    
    if not len(weights_comp): # if []
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
        
        
def inspect_estimated_curves(solution, real, col_to_consider = ['INFECTED_HO', 'DECEASED']):
    
    solution_sub = solution[col_to_consider]
    real_sub = real[col_to_consider]
    #print('MSE: ', mse_on_df(solution_sub, real_sub))
    
    tot_obs = pd.DataFrame()
    for c in col_to_consider:
        tot_obs[c] = real[c].values
        tot_obs[c+'_EST'] = solution[c].values

    tot_obs.plot( xlabel = 'days', ylabel = 'cases')


def _mse(target_profile: pd.DataFrame, start: int, end: int, time_scale:int, 
         known_pars: list, 
         col_to_consider: list = ['INFECTED_HO', 'DECEASED']) -> Callable:
    
    
        """
        Generate a MSE error with the inputted target profile.
        :param target_profile: the target time series of observed new cases.
        :return: a callable function computing the mse between estimated and real observed new cases from the unknown parameters.
        """

        def target_series_mse(unknown_pars: Tuple )  -> float:                               
                             
            """
            MSE between target profile and the estimated profile made by estimated_observed_new_cases.
            :param pars: parameters of the model function.
            :param col_to_consider: a list containing the colnames to consider for calculating MSE
            :return: the mse between the estimated and target series.
            """
            
            initE, initIas, initIsy, initIho, initQ, initR, initD, initN, gamma_as, gamma_sy, gamma_ho, gamma_quar, p_sy, p_ho, pdie_sy, pdie_ho = known_pars
            beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = unknown_pars
            
            days = (end - start)*time_scale
            
            solutions = main(initE, initIas, initIsy, initIho, initQ, initR, initD, initN, 
                             beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu,
                             days, 
                             p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy, 
                             output = 'Solutions')
            
            
            solutions['POSITIVE'] = solutions['INFECTED_AS'] + solutions['INFECTED_SY'] + solutions['INFECTED_HO'] + solutions['QUARANTINED']
            
            w_dict = {'DECEASED': 1, 'INFECTED_HO' : 1, 'QUARANTINED': 0.05, 'POSITIVE': 0.01}
            
            MSE_rel = mse_on_df(solutions.iloc[0:(end - start)*time_scale:time_scale, :][col_to_consider], 
                                target_profile.iloc[start:end, :][col_to_consider],
                                weighted = True, 
                                weights_comp = [w_dict[c] for c in col_to_consider])
            
            return MSE_rel
         
        return target_series_mse
    
def fit_parameters(
            known_pars : list,
            unknown_pars_guess : list,
            observed_cases: pd.DataFrame,
            start: int, 
            end: int,
            time_scale: int = 1,
            col_to_consider: list = ['INFECTED_HO', 'DECEASED'], # QUARANTINED
            bounds: list = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1)],
            tol: float = 0.1,
            max_it: int = 150,
            verbose: bool = True
            ) -> OptimizeResult:
        """
        Estimate the optimal parameters minimizing the mse.
        :return: The estimated parameters in a scipy OptimizeResult object.
        """
        
        # beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy
        
        # set bounds and constraints
        lower_bounds = [ b[0] for b in bounds ]
        upper_bounds = [ b[1] for b in bounds ]

        opt_param = scipy.optimize.minimize(fun = _mse(observed_cases, start, end, time_scale, 
                                                       known_pars, col_to_consider), 
                                            x0 = unknown_pars_guess, 
                                            method='trust-constr', 
                                            bounds=Bounds(lower_bounds, upper_bounds), 
                                            # constraints = ()
                                            tol=tol, 
                                            options={'xtol': 1e-10, 
                                                     'gtol': 1e-08, 
                                                     'barrier_tol': 1e-08, 
                                                     'maxiter': max_it, 
                                                     'verbose': 0, 
                                                     'disp': verbose})
        

        return opt_param

###########################################################################
# fit Period 'Start' 17/02/2020 - 11/03/2020
col_to_consider = ['INFECTED_HO', 'DECEASED', 'QUARANTINED', 'POSITIVE']

start = 1
end = 24
time_scale = 1
days = (end - start)*time_scale

# observed
initN = 4900000
initE = 100
initIas = 6
initIsy = 3
initIho = 0
initQ = 0
initR = 0
initD = 0
gamma_as, gamma_sy, gamma_ho, gamma_quar = 1/18, 1/18, 1/26, 1/14 
p_as, p_sy = 0.645, 0.355
p_ho = 0.223 
pdie_sy = 0.024 
pdie_ho = 0.271 

# guesses
beta_as, beta_sy, beta_ho, beta_quar  = 3/14, 6/14, 1/100, 1/100 
epsilon = 1/5 
phi_as, phi_sy = 1/5, 1/3 
psi = 1/10 
mu = 1/12 
pquar_as, pquar_sy = 0.00001, 0.00001  

pquar_sy_max = 0.3 # 1 - p_ho - pdie_sy   # pquar_sy < 1 - p_ho - pdie_sy = 1 - 0.255 = 0.745

# bounds
bounds = [(0.5, 2), (0.5, 2), (1/1000,1/5), (1/1000,1/5), 
          (1/10,1),
          (1/10,1), (1/10,1), 
          (1/14,1), 
          (1/30,1),
          (0.00001,0.2), (0.00001,pquar_sy_max)]

known_pars = initE, initIas, initIsy, initIho, initQ, initR, initD, initN, gamma_as, gamma_sy, gamma_ho, gamma_quar, p_sy, p_ho, pdie_sy, pdie_ho 
unknown_pars_guess = beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy 

opt_param1 = fit_parameters(known_pars = known_pars,
            unknown_pars_guess = unknown_pars_guess,
            observed_cases = real_observations,
            start = start, 
            end = end,
            time_scale = time_scale,
            col_to_consider = col_to_consider, 
            bounds = bounds,
            tol = 0.0001,
            max_it = 1000,
            verbose = True
            )

# inspect solution
beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param1['x']

solutiondf1 = main(initE, initIas, initIsy, initIho, initQ, initR, initD, initN, 
         beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu,
         days, 
         p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy, 
         output = 'Solutions')

solutiondf1['POSITIVE'] = solutiondf1['INFECTED_AS'] + solutiondf1['INFECTED_SY'] + solutiondf1['INFECTED_HO'] + solutiondf1['QUARANTINED']
      

'''    
inspect_estimated_curves(solutiondf1.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['INFECTED_HO'])
inspect_estimated_curves(solutiondf1.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['DECEASED'])
inspect_estimated_curves(solutiondf1.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['QUARANTINED'])
inspect_estimated_curves(solutiondf1.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['POSITIVE'])
'''

_, initE1, initIas1, initIsy1, initIho1, initQ1, initR1, initD1 = solutiondf1.iloc[-1,:-1].values

###########################################################################
# Period 'Lockdown' 11/03/2020 - 11/05/2020
start = 24
end = 85
time_scale = 1
days = (end - start)*time_scale

# observed
initN = 4900000
initE, initIas, initIsy, initIho, initQ, initR, initD = initE1, initIas1, initIsy1, initIho1, initQ1, initR1, initD1
gamma_as, gamma_sy, gamma_ho, gamma_quar = 1/18, 1/18, 1/26, 1/14 
p_as, p_sy = 0.645, 0.355 
p_ho = 0.223 
pdie_sy = 0.024 
pdie_ho = 0.271 

# starting solution
beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param1['x']

pquar_sy_max = 1 - p_ho - pdie_sy   # pquar_sy < 1 - p_ho - pdie_sy

# bounds
bounds = [(1/100, 1), (1/100, 1), (1/1000,1/5), (1/1000,1/5), 
          (1/10,1),
          (1/10,1), (1/10,1), 
          (1/14,1), 
          (1/30,1),
          (0.00001,0.8), (0.00001,pquar_sy_max)]

known_pars = initE, initIas, initIsy, initIho, initQ, initR, initD, initN, gamma_as, gamma_sy, gamma_ho, gamma_quar, p_sy, p_ho, pdie_sy, pdie_ho 
unknown_pars_guess = beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy 

opt_param2 = fit_parameters(known_pars = known_pars,
            unknown_pars_guess = unknown_pars_guess,
            observed_cases = real_observations,
            start = start, 
            end = end,
            time_scale = time_scale,
            col_to_consider = col_to_consider, 
            bounds = bounds,
            tol = 0.0001,
            max_it = 2500,
            verbose = True)

beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param2['x']

solutiondf2 = main(initE, initIas, initIsy, initIho, initQ, initR, initD, initN, 
         beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu,
         days, 
         p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy, 
         output = 'Solutions')

solutiondf2['POSITIVE'] = solutiondf2['INFECTED_AS'] + solutiondf2['INFECTED_SY'] + solutiondf2['INFECTED_HO'] + solutiondf2['QUARANTINED']

'''        
inspect_estimated_curves(solutiondf2.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['INFECTED_HO'])
inspect_estimated_curves(solutiondf2.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['DECEASED'])
inspect_estimated_curves(solutiondf2.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['QUARANTINED'])
inspect_estimated_curves(solutiondf2.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['POSITIVE'])
​'''

_, initE2, initIas2, initIsy2, initIho2, initQ2, initR2, initD2 = solutiondf2.iloc[-1,:-1].values


###########################################################################
#Period 'Post Lockdown' 11/05/2020 - 15/06/2020
start = 85
end = 120
time_scale = 1
days = (end - start)*time_scale


# observed
initN = 4900000
initE, initIas, initIsy, initIho, initQ, initR, initD = initE2, initIas2, initIsy2, initIho2, initQ2, initR2, initD2
gamma_as, gamma_sy, gamma_ho, gamma_quar = 1/18, 1/18, 1/26, 1/14 
p_as, p_sy = 0.645, 0.355 
p_ho = 0.223 
pdie_sy = 0.024
pdie_ho = 0.271 

# starting solution
beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param2['x']

pquar_sy_max = 1 - p_ho - pdie_sy   # pquar_sy < 1 - p_ho - pdie_sy

# bounds
bounds = [(beta_as, 2), (beta_sy, 2), (1/1000,1/10), (1/1000,1/10), 
          (1/10,1),
          (1/10,1), (1/10,1), 
          (1/14,1), 
          (1/30,1),
          (0.00001,0.8), (0.00001,pquar_sy_max)]

known_pars = initE, initIas, initIsy, initIho, initQ, initR, initD, initN, gamma_as, gamma_sy, gamma_ho, gamma_quar, p_sy, p_ho, pdie_sy, pdie_ho 
unknown_pars_guess = beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy 

opt_param3 = fit_parameters(known_pars = known_pars,
            unknown_pars_guess = unknown_pars_guess,
            observed_cases = real_observations,
            start = start, 
            end = end,
            time_scale = time_scale,
            col_to_consider = col_to_consider,
            bounds = bounds,
            tol = 0.000001,
            max_it = 1500,
            verbose = True
            )

beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param3['x']

solutiondf3 = main(initE, initIas, initIsy, initIho, initQ, initR, initD, initN, 
         beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu,
         days, 
         p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy, 
         output = 'Solutions')

solutiondf3['POSITIVE'] = solutiondf3['INFECTED_AS'] + solutiondf3['INFECTED_SY'] + solutiondf3['INFECTED_HO'] + solutiondf3['QUARANTINED']

'''            
inspect_estimated_curves(solutiondf3.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['INFECTED_HO'])
inspect_estimated_curves(solutiondf3.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['DECEASED'])
inspect_estimated_curves(solutiondf3.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['QUARANTINED'])
inspect_estimated_curves(solutiondf3.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['POSITIVE'])
​'''


_, initE3, initIas3, initIsy3, initIho3, initQ3, initR3, initD3 = solutiondf3.iloc[-1,:-1].values


###########################################################################
# Period 'Summer' 15/06/2020 - 6/09/2020
start = 120
end = 203
time_scale = 1
days = (end - start)*time_scale

# observed
initN = 4900000
initE, initIas, initIsy, initIho, initQ, initR, initD = initE3, initIas3, initIsy3, initIho3, initQ3, initR3, initD3
gamma_as, gamma_sy, gamma_ho, gamma_quar = 1/18, 1/18, 1/26, 1/14 
p_as, p_sy = 0.645, 0.355 
p_ho = 0.223 
pdie_sy = 0.024 
pdie_ho = 0.271 

# starting solution
beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param3['x']

pquar_sy_max = 1 - p_ho - pdie_sy   # pquar_sy < 1 - p_ho - pdie_sy

# bounds
bounds = [(0.5, 1), (0.5, 1), (1/1000,1/10), (1/1000,1/10), 
          (1/10,1),
          (1/10,1), (1/10,1), 
          (1/14,1), 
          (1/30,1),
          (0.00001,0.8), (0.00001,pquar_sy_max)]

known_pars = initE, initIas, initIsy, initIho, initQ, initR, initD, initN, gamma_as, gamma_sy, gamma_ho, gamma_quar, p_sy, p_ho, pdie_sy, pdie_ho 
unknown_pars_guess = beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy 

opt_param4 = fit_parameters(known_pars = known_pars,
            unknown_pars_guess = unknown_pars_guess,
            observed_cases = real_observations,
            start = start, 
            end = end,
            time_scale = time_scale,
            col_to_consider = col_to_consider, 
            bounds = bounds,
            tol = 0.000001,
            max_it = 500,
            verbose = True
            )

beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param4['x']

solutiondf4 = main(initE, initIas, initIsy, initIho, initQ, initR, initD, initN, 
         beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu,
         days, 
         p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy, 
         output = 'Solutions')

solutiondf4['POSITIVE'] = solutiondf4['INFECTED_AS'] + solutiondf4['INFECTED_SY'] + solutiondf4['INFECTED_HO'] + solutiondf4['QUARANTINED']
   
'''         
inspect_estimated_curves(solutiondf4.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['INFECTED_HO'])
inspect_estimated_curves(solutiondf4.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['DECEASED'])
inspect_estimated_curves(solutiondf4.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['QUARANTINED'])
inspect_estimated_curves(solutiondf4.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['POSITIVE'])
​'''

_, initE4, initIas4, initIsy4, initIho4, initQ4, initR4, initD4 = solutiondf4.iloc[-1,:-1].values


###########################################################################
#Period 'School reopening' 6/09/2020 - 3/11/2020
start = 203
end = 261
time_scale = 1
days = (end - start)*time_scale


# observed
initN = 4900000
initE, initIas, initIsy, initIho, initQ, initR, initD = initE4, initIas4, initIsy4, initIho4, initQ4, initR4, initD4
gamma_as, gamma_sy, gamma_ho, gamma_quar = 1/18, 1/18, 1/26, 1/14 
p_as, p_sy = 0.645, 0.355 
p_ho = 0.223 
pdie_sy = 0.024 
pdie_ho = 0.271

# starting solution
beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param4['x']

pquar_sy_max = 1 - p_ho - pdie_sy   # pquar_sy < 1 - p_ho - pdie_sy

# bounds
bounds = [(0.6, 1), (0.6, 1), (1/1000,1/10), (1/1000,1/10), 
          (1/10,1),
          (1/10,1), (1/10,1), 
          (1/14,1), 
          (1/30,1),
          (0.00001,0.8), (0.00001,pquar_sy_max)] 

known_pars = initE, initIas, initIsy, initIho, initQ, initR, initD, initN, gamma_as, gamma_sy, gamma_ho, gamma_quar, p_sy, p_ho, pdie_sy, pdie_ho 
unknown_pars_guess = beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy 

opt_param5 = fit_parameters(known_pars = known_pars,
            unknown_pars_guess = unknown_pars_guess,
            observed_cases = real_observations,
            start = start, 
            end = end,
            time_scale = time_scale,
            col_to_consider = col_to_consider, 
            bounds = bounds,
            tol = 0.0001,
            max_it = 500,
            verbose = True
            )

beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param5['x']

solutiondf5 = main(initE, initIas, initIsy, initIho, initQ, initR, initD, initN, 
         beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu,
         days, 
         p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy, 
         output = 'Solutions')

solutiondf5['POSITIVE'] = solutiondf5['INFECTED_AS'] + solutiondf5['INFECTED_SY'] + solutiondf5['INFECTED_HO'] + solutiondf5['QUARANTINED']
  
'''          
inspect_estimated_curves(solutiondf5.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['INFECTED_HO'])
inspect_estimated_curves(solutiondf5.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['DECEASED'])
inspect_estimated_curves(solutiondf5.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['QUARANTINED'])
inspect_estimated_curves(solutiondf5.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['POSITIVE'])
​'''

_, initE5, initIas5, initIsy5, initIho5, initQ5, initR5, initD5 = solutiondf5.iloc[-1,:-1].values


###########################################################################
# Period 'Second Peak' 3/11/2020 - 24/12/2021
start = 261
end = 312 #338
time_scale = 1
days = (end - start)*time_scale


# observed
initN = 4900000
initE, initIas, initIsy, initIho, initQ, initR, initD = initE5, initIas5, initIsy5, initIho5, initQ5, initR5, initD5
gamma_as, gamma_sy, gamma_ho, gamma_quar = 1/18, 1/18, 1/26, 1/14 
p_as, p_sy = 0.645, 0.355 
p_ho = 0.223 
pdie_sy = 0.024
pdie_ho = 0.271

# starting solution
beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param5['x']

pquar_sy_max = 1 - p_ho - pdie_sy   # pquar_sy < 1 - p_ho - pdie_sy

# bounds
bounds = [(0.2, 2), (0.2, 2), (1/1000,1/10), (1/1000,1/10), 
          (1/10,1),
          (1/10,1), (1/10,1), 
          (1/14,1), 
          (1/30,1),
          (0.00001,0.9), (0.00001,pquar_sy_max)]

known_pars = initE, initIas, initIsy, initIho, initQ, initR, initD, initN, gamma_as, gamma_sy, gamma_ho, gamma_quar, p_sy, p_ho, pdie_sy, pdie_ho 
unknown_pars_guess = beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy 


opt_param6 = fit_parameters(known_pars = known_pars,
            unknown_pars_guess = unknown_pars_guess,
            observed_cases = real_observations,
            start = start, 
            end = end,
            time_scale = time_scale,
            col_to_consider = col_to_consider, 
            bounds = bounds,
            tol = 0.00000001,
            max_it = 500,
            verbose = True)

beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param6['x']

solutiondf6 = main(initE, initIas, initIsy, initIho, initQ, initR, initD, initN, 
         beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu,
         days, 
         p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy, 
         output = 'Solutions')

solutiondf6['POSITIVE'] = solutiondf6['INFECTED_AS'] + solutiondf6['INFECTED_SY'] + solutiondf6['INFECTED_HO'] + solutiondf6['QUARANTINED']
  
'''          
inspect_estimated_curves(solutiondf6.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['INFECTED_HO'])
inspect_estimated_curves(solutiondf6.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['DECEASED'])
inspect_estimated_curves(solutiondf6.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['QUARANTINED'])
inspect_estimated_curves(solutiondf6.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['POSITIVE'])
​'''

_, initE6, initIas6, initIsy6, initIho6, initQ6, initR6, initD6 = solutiondf6.iloc[-1,:-1].values

###########################################################################
# Period 'Christmas holidays' 24/12/2020 - 19/1/2021
start = 312
end = 338
time_scale = 1
days =  (end - start) * time_scale


# observed
initN = 4900000
initE, initIas, initIsy, initIho, initQ, initR, initD = initE6, initIas6, initIsy6, initIho6, initQ6, initR6, initD6
gamma_as, gamma_sy, gamma_ho, gamma_quar = 1/18, 1/18, 1/26, 1/14 
p_as, p_sy = 0.645, 0.355 
p_ho = 0.223 
pdie_sy = 0.024
pdie_ho = 0.271

# starting solution
beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param6['x']

pquar_sy_max = 1 - p_ho - pdie_sy   # pquar_sy < 1 - p_ho - pdie_sy

# bounds
bounds = [(0.1, 2), (0.1, 2), (1/1000,1/10), (1/1000,1/10), 
          (1/10,1),
          (1/10,1), (1/10,1), 
          (1/14,1), 
          (1/30,1),
          (0.00001,0.9), (0.00001,pquar_sy_max)]

known_pars = initE, initIas, initIsy, initIho, initQ, initR, initD, initN, gamma_as, gamma_sy, gamma_ho, gamma_quar, p_sy, p_ho, pdie_sy, pdie_ho 
unknown_pars_guess = beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy 

opt_param7 = fit_parameters(known_pars = known_pars,
            unknown_pars_guess = unknown_pars_guess,
            observed_cases = real_observations,
            start = start, 
            end = end,
            time_scale = time_scale,
            col_to_consider = col_to_consider, 
            bounds = bounds,
            tol = 0.00000001,
            max_it = 500,
            verbose = True
            )

beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param7['x']

solutiondf7 = main(initE, initIas, initIsy, initIho, initQ, initR, initD, initN, 
         beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu,
         days, 
         p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy, 
         output = 'Solutions')

solutiondf7['POSITIVE'] = solutiondf7['INFECTED_AS'] + solutiondf7['INFECTED_SY'] + solutiondf7['INFECTED_HO'] + solutiondf7['QUARANTINED']

'''          
inspect_estimated_curves(solutiondf7.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['INFECTED_HO'])
inspect_estimated_curves(solutiondf7.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['DECEASED'])
inspect_estimated_curves(solutiondf7.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['QUARANTINED'])
inspect_estimated_curves(solutiondf7.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['POSITIVE'])
​'''

_, initE7, initIas7, initIsy7, initIho7, initQ7, initR7, initD7 = solutiondf7.iloc[-1,:-1].values

###########################################################################
# Period 'Post Christmas holidays' 19/1/2020 - 13/2/2021
start = 338
end = 363 #379
time_scale = 1
days = (end - start)*time_scale


# observed
initN = 4900000
initE, initIas, initIsy, initIho, initQ, initR, initD = initE7, initIas7, initIsy7, initIho7, initQ7, initR7, initD7
gamma_as, gamma_sy, gamma_ho, gamma_quar = 1/18, 1/18, 1/26, 1/14 
p_as, p_sy = 0.645, 0.355 
p_ho = 0.223 
pdie_sy = 0.024 
pdie_ho = 0.271

# starting solution
beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param6['x']

pquar_sy_max = 1 - p_ho - pdie_sy   # pquar_sy < 1 - p_ho - pdie_sy

# bounds
# bounds
bounds = [(1/100, 1), (1/100, 1), (1/1000,1/10), (1/1000,1/10), 
          (1/10,1),
          (1/10,1), (1/10,1), 
          (1/14,1), 
          (1/30,1),
          (0.00001,0.9), (0.00001,pquar_sy_max)]

known_pars = initE, initIas, initIsy, initIho, initQ, initR, initD, initN, gamma_as, gamma_sy, gamma_ho, gamma_quar, p_sy, p_ho, pdie_sy, pdie_ho 
unknown_pars_guess = beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy 

opt_param8 = fit_parameters(known_pars = known_pars,
            unknown_pars_guess = unknown_pars_guess,
            observed_cases = real_observations,
            start = start, 
            end = end,
            time_scale = time_scale,
            col_to_consider = col_to_consider,
            bounds = bounds,
            tol = 0.0001,
            max_it = 500,
            verbose = True
            )

beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = opt_param8['x']

solutiondf8 = main(initE, initIas, initIsy, initIho, initQ, initR, initD, initN, 
         beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu,
         days, 
         p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy, 
         output = 'Solutions')


solutiondf8['POSITIVE'] = solutiondf8['INFECTED_AS'] + solutiondf8['INFECTED_SY'] + solutiondf8['INFECTED_HO'] + solutiondf8['QUARANTINED']

'''         
inspect_estimated_curves(solutiondf8.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['INFECTED_HO'])
inspect_estimated_curves(solutiondf8.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['DECEASED'])
inspect_estimated_curves(solutiondf8.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['QUARANTINED'])
inspect_estimated_curves(solutiondf8.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['POSITIVE'])
​'''


###########################################################################
# concatenated all solutions

estimates = pd.DataFrame()
estimates['coeff_name'] = ['start','end','start_date','end_date','beta_as', 'beta_sy', 'beta_ho', 'beta_quar', 'epsilon', 'phi_as', 'phi_sy', 'psi', 'mu', 'pquar_as', 'pquar_sy']
estimates['Period_1'] = [1,24, '2020-02-17', '2020-03-11' ] + list(opt_param1['x'])
estimates['Period_2'] = [24,85, '2020-03-11', '2020-05-11'] + list(opt_param2['x'])
estimates['Period_3'] = [85,120, '2020-05-11', '2020-06-15'] + list(opt_param3['x'])
estimates['Period_4'] = [120, 203, '2020-06-15', '2020-09-06'] + list(opt_param4['x'])
estimates['Period_5'] = [203, 261, '2020-09-06', '2020-11-03'] + list(opt_param5['x'])
estimates['Period_6'] = [261, 312, '2020-11-03', '2020-12-24' ] + list(opt_param6['x'])
estimates['Period_7'] = [312, 338, '2020-12-24', '2021-01-19'] + list(opt_param7['x'])
estimates['Period_8'] = [338, 363, '2021-01-19', '2021-02-13'] + list(opt_param8['x'])

# export sols
#estimates.to_csv('param_estimates_DET_8P.csv')
#estimates.to_csv('param_estimates_DET_7P.csv')
sol_tot = pd.concat([solutiondf1, solutiondf2, solutiondf3, solutiondf4, solutiondf5, solutiondf6, solutiondf7, solutiondf8])

tot_obs = real_observations.iloc[1:363].copy()
tot_obs['INFECTED_AS_EST'] = sol_tot['INFECTED_AS'].values
tot_obs['INFECTED_SY_EST'] = sol_tot['INFECTED_SY'].values
tot_obs['INFECTED_HO_EST'] = sol_tot['INFECTED_HO'].values
tot_obs['QUARANTINED_EST'] = sol_tot['QUARANTINED'].values
tot_obs['DECEASED_EST'] = sol_tot['DECEASED'].values
tot_obs['POSITIVE_EST'] = sol_tot['POSITIVE'].values
tot_obs['SUSCEPTIBLE_EST'] = sol_tot['SUSCEPTIBLE'].values
tot_obs['REMOVED_EST'] = sol_tot['REMOVED'].values


#tot_obs = tot_obs.iloc[1:300]

tot_obs[['POSITIVE', 'POSITIVE_EST',]].plot( xlabel = 'days', ylabel = 'cases', figsize=(8,5))
plt.axvline(24, color="grey", linestyle="--", lw = 0.7)
plt.axvline(85, color="grey", linestyle="--", lw = 0.7)
plt.axvline(120, color="grey", linestyle="--", lw = 0.7)
plt.axvline(203, color="grey", linestyle="--", lw = 0.7)
plt.axvline(261, color="grey", linestyle="--", lw = 0.7)
plt.axvline(312, color="grey", linestyle="--", lw = 0.7)
plt.axvline(338, color="grey", linestyle="--", lw = 0.7)
plt.savefig('P_8.png')


tot_obs[['QUARANTINED', 'QUARANTINED_EST']].plot( xlabel = 'days', ylabel = 'cases', figsize=(8,5))
plt.axvline(24, color="grey", linestyle="--", lw = 0.7)
plt.axvline(85, color="grey", linestyle="--", lw = 0.7)
plt.axvline(120, color="grey", linestyle="--", lw = 0.7)
plt.axvline(203, color="grey", linestyle="--", lw = 0.7)
plt.axvline(261, color="grey", linestyle="--", lw = 0.7)
plt.axvline(312, color="grey", linestyle="--", lw = 0.7)
plt.axvline(338, color="grey", linestyle="--", lw = 0.7)
plt.savefig('Q_8.png')


tot_obs[['DECEASED', 'DECEASED_EST']].plot( xlabel = 'days', ylabel = 'cases', figsize=(8,5))
plt.axvline(24, color="grey", linestyle="--", lw = 0.7)
plt.axvline(85, color="grey", linestyle="--", lw = 0.7)
plt.axvline(120, color="grey", linestyle="--", lw = 0.7)
plt.axvline(203, color="grey", linestyle="--", lw = 0.7)
plt.axvline(261, color="grey", linestyle="--", lw = 0.7)
plt.axvline(312, color="grey", linestyle="--", lw = 0.7)
plt.axvline(338, color="grey", linestyle="--", lw = 0.7)
plt.savefig('D_8.png')


tot_obs[['INFECTED_HO', 'INFECTED_HO_EST']].plot( xlabel = 'days', ylabel = 'cases', figsize=(8,5) )
plt.axvline(24, color="grey", linestyle="--", lw = 0.7)
plt.axvline(85, color="grey", linestyle="--", lw = 0.7)
plt.axvline(120, color="grey", linestyle="--", lw = 0.7)
plt.axvline(203, color="grey", linestyle="--", lw = 0.7)
plt.axvline(261, color="grey", linestyle="--", lw = 0.7)
plt.axvline(312, color="grey", linestyle="--", lw = 0.7)
plt.axvline(338, color="grey", linestyle="--", lw = 0.7)
plt.savefig('I_HO_8.png')


###########################################################################
# Predict with fitted Deterministic model

# import fitted parameters 
det_params = pd.read_csv( 'param_estimates_DET_8P.csv', index_col = 0 ) # DAYLY
det_params = det_params.iloc[4:,:]
det_params.iloc[:,1:] = det_params.iloc[:,1:].astype(float)

time_points = [1,24,85,120,203,261,312,338,363]


# observed
initN = 4900000
initE = 100
initIas = 6
initIsy = 3
initIho = 0
initQ = 0
initR = 0
initD = 0
gamma_as, gamma_sy, gamma_ho, gamma_quar = 1/18, 1/18, 1/26, 1/14 
p_as, p_sy = 0.645, 0.355
p_ho = 0.223 
pdie_sy = 0.024 
pdie_ho = 0.271 

inits = [[initE, initIas, initIsy, initIho, initQ, initR, initD, initN]]

sol_tot = pd.DataFrame()

for i in range(len(time_points)-1):
    start = time_points[i]
    end = time_points[i+1]
    time_scale = 1
    days = (end - start)*time_scale
    
    
    # get opt parameter
    beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = det_params.iloc[:,i+1].values
    
    solutiondf = main(initE, initIas, initIsy, initIho, initQ, initR, initD, initN, 
             beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu,
             days, 
             p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy, 
             output = 'Solutions')
    
    solutiondf['POSITIVE'] = solutiondf['INFECTED_AS'] + solutiondf['INFECTED_SY'] + solutiondf['INFECTED_HO'] + solutiondf['QUARANTINED']
    sol_tot = pd.concat([sol_tot, solutiondf])
    
    # inspect_estimated_curves(solutiondf.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['INFECTED_HO'])
    # inspect_estimated_curves(solutiondf.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['DECEASED'])
    # inspect_estimated_curves(solutiondf.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['QUARANTINED'])
    # inspect_estimated_curves(solutiondf.iloc[0:(end-start)*time_scale:time_scale,:], real_observations.iloc[start:end, :], col_to_consider = ['POSITIVE'])
    _, initE, initIas, initIsy, initIho, initQ, initR, initD = solutiondf.iloc[-1,:-1].values
    
    inits.append([initE, initIas, initIsy, initIho, initQ, initR, initD, initN])
    
   
    
# predict starting from last observations/fitting

last_p = 1# last period 

predict_days = 12 # days to predict

#start = time_points[-last_p -1]
#end = time_points[-last_p] + predict_days
start = time_points[-1]-1
end = start + predict_days

time_scale = 1
days = (end - start)*time_scale

#initE, initIas, initIsy, initIho, initQ, initR, initD, initN = inits[-last_p -1]
initE, initIas, initIsy, initIho, initQ, initR, initD, initN = inits[ -1]


# adjust with real ones where known
initIho, initQ, initD, _ = real_observations.iloc[start,:].values


beta_as, beta_sy, beta_ho, beta_quar, epsilon, phi_as, phi_sy, psi, mu, pquar_as, pquar_sy = det_params.iloc[:,-last_p].values



solutiondf = main(initE, initIas, initIsy, initIho, initQ, initR, initD, initN, 
         beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu,
         days, 
         p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy, 
         output = 'Solutions')



solutiondf['POSITIVE'] = solutiondf['INFECTED_AS'] + solutiondf['INFECTED_SY'] + solutiondf['INFECTED_HO'] + solutiondf['QUARANTINED']




D_real_start = int(real_observations.iloc[start, :]['DECEASED'])

D_real_end = int(real_observations.iloc[end -1, :]['DECEASED'])

D_pred = int(solutiondf.iloc[(end-start) -1,:]['DECEASED'])

I_HO_real_start = int(real_observations.iloc[start, :]['INFECTED_HO'])

I_HO_real_end = int(real_observations.iloc[end -1 , :]['INFECTED_HO'])

I_HO_pred = int(solutiondf.iloc[(end-start) -1,:]['INFECTED_HO'])





# plot
fig, ax = plt.subplots(1,1, figsize=(8,5))

ax.plot(range(start,end),solutiondf.iloc[0:(end-start)*time_scale:time_scale,:]['INFECTED_HO'].values, color='grey', linestyle='dashed', label = 'Predicted')
ax.plot(range(start-15,end),real_observations.iloc[start-15:end, :]['INFECTED_HO'].values, label = 'Real')
plt.scatter([start, end-1, end-1], [real_observations.iloc[start, :]['INFECTED_HO'], real_observations.iloc[end-1, :]['INFECTED_HO'], solutiondf.iloc[(end-start) -1,:]['INFECTED_HO']], c='grey')
ax.legend()

ax.text(0.62, 0.4, str(I_HO_real_start), fontsize = 10, horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
ax.text(0.89, 0.25, str(I_HO_real_end), fontsize = 10, horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
ax.text(0.89, 0.05, str(I_HO_pred), fontsize = 10, horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

ax.set_title('Hospitalizations Prediction')
ax.set_xlabel('days')
ax.set_ylabel('cases')
plt.savefig('Det_pred_I_HO_8.png')


fig, ax = plt.subplots(1,1, figsize=(8,5))

ax.plot(range(start,end),solutiondf.iloc[0:(end-start)*time_scale:time_scale,:]['DECEASED'].values, color='grey', linestyle='dashed', label = 'Predicted')
ax.plot(range(start-15,end),real_observations.iloc[start-15:end, :]['DECEASED'].values, label = 'Real')
plt.scatter([start, end-1, end-1], [real_observations.iloc[start, :]['DECEASED'], real_observations.iloc[end-1, :]['DECEASED'], solutiondf.iloc[(end-start) -1,:]['DECEASED']], c='grey')
ax.legend()

ax.text(0.62, 0.62, str(D_real_start), fontsize = 10, horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
ax.text(0.89, 0.96, str(D_pred), fontsize = 10,  horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
ax.text(0.89, 0.79, str(D_real_end), fontsize = 10, horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

ax.set_title('Deceases Prediction')
ax.set_xlabel('days')
ax.set_ylabel('cases')
plt.savefig('Det_pred_D_8.png')


    