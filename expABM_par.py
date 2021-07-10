# Experiment pandemic scenarious in Veneto using ABmodel

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

#import fitted parameters DET model
det_params = pd.read_csv( 'param_estimates_DET_8P.csv', index_col = 1 ) 
det_params = det_params.iloc[:,1:]

# import fitted parameters AB model
abm_params = pd.read_csv( 'param_estimates_ABM_8P.csv', index_col = 0 ) 

abm_params.iloc[:2,1:] = abm_params.iloc[:2,1:].astype(int)
abm_params.iloc[2:,1:] = abm_params.iloc[2:,1:].astype(float)

col_to_consider = ['INFECTED_HO', 'DECEASED', 'QUARANTINED', 'POSITIVE']

#############################################################################

# simulation global settings

n_sim = 10               # number of model simulations
N_abm = 20000             # number of agents
num_nodes_E = 1          # number of agents in E
num_nodes_Ias = 1        # number of agents in Ias
num_nodes_Isy = 1        # number of agents in Isy
num_nodes_Iho = 0        # number of agents in Iho
num_nodes_Q = 0          # number of agents in Q
num_nodes_R = 0          # number of agents in R
num_nodes_D = 0           # number of agents In D

 
#############################################################################

# set start - end days
start = int(det_params.loc['start', 'Period_1'])
end = int(det_params.loc['end', 'Period_5'])
days = end

end2 = end + 30
days2 = end2

# retrieve ABM fitting and DET model fitting


# initialize containers
change_times = []
change_pars = []
change_pars1 = []
change_pars2 = []
change_pars3 = []
change_pars5 = []


# observed (DET fitting)
for i in range(1, 5+ 1):
    
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

    if i == 1:
        known_pars1 = known_pars
        

    # fitted ABM parameters       
    last_start, last_end, ptrans_community_as, ptrans_other_as, ptrans_house_as, ptrans_work_as, ptrans_work_colleague_as, ptrans_school_as, ptrans_school_class_as, ptrans_school_friend_as, ptrans_community_sy, ptrans_other_sy, ptrans_house_sy, ptrans_work_sy, ptrans_work_colleague_sy, ptrans_school_sy, ptrans_school_class_sy, ptrans_school_friend_sy, ptrans_quarantine, ptrans_ho = abm_params.loc[:, 'Period_'+str(i)].values
    
    fitted_pars = ptrans_community_as, ptrans_other_as, ptrans_house_as, ptrans_work_as, ptrans_work_colleague_as, ptrans_school_as, ptrans_school_class_as, ptrans_school_friend_as, ptrans_community_sy, ptrans_other_sy, ptrans_house_sy, ptrans_work_sy, ptrans_work_colleague_sy, ptrans_school_sy, ptrans_school_class_sy, ptrans_school_friend_sy, ptrans_quarantine, ptrans_ho

    change_times.append(last_start)
    
    
    # SIMULATION 0: use originally fitted variable params
    change_pars = change_pars + [list(fitted_pars) + list(known_pars)]  
    
    
    # SIMULATION 1: assume no measure - use always parameters of Period_1
    last_start1, last_end1, ptrans_community_as1, ptrans_other_as1, ptrans_house_as1, ptrans_work_as1, ptrans_work_colleague_as1, ptrans_school_as1, ptrans_school_class_as1, ptrans_school_friend_as1, ptrans_community_sy1, ptrans_other_sy1, ptrans_house_sy1, ptrans_work_sy1, ptrans_work_colleague_sy1, ptrans_school_sy1, ptrans_school_class_sy1, ptrans_school_friend_sy1, ptrans_quarantine1, ptrans_ho1 = abm_params.loc[:, 'Period_1'].values
    fitted_pars1 = ptrans_community_as1, ptrans_other_as1, ptrans_house_as1, ptrans_work_as1, ptrans_work_colleague_as1, ptrans_school_as1, ptrans_school_class_as1, ptrans_school_friend_as1, ptrans_community_sy1, ptrans_other_sy1, ptrans_house_sy1, ptrans_work_sy1, ptrans_work_colleague_sy1, ptrans_school_sy1, ptrans_school_class_sy1, ptrans_school_friend_sy1, ptrans_quarantine1, ptrans_ho1
    
    change_pars1 = change_pars1 + [list(fitted_pars1) + list(known_pars1)] 
    
    
    # SIMULATION 2: no school closure march-june - use school prob. transmission of Period_1 also in Period_2 and Period_3
    if i == 2 or i == 3:
        fitted_pars2 = ptrans_community_as, ptrans_other_as, ptrans_house_as, ptrans_work_as, ptrans_work_colleague_as, ptrans_school_as1, ptrans_school_class_as1, ptrans_school_friend_as1, ptrans_community_sy, ptrans_other_sy, ptrans_house_sy, ptrans_work_sy, ptrans_work_colleague_sy, ptrans_school_sy1, ptrans_school_class_sy1, ptrans_school_friend_sy1, ptrans_quarantine, ptrans_ho
    else:
        fitted_pars2 = fitted_pars

    change_pars2 = change_pars2 + [list(fitted_pars2) + list(known_pars)] 
    
    
    # SIMULATION 3: no school reopening from september - set school prob. transmission of Period_5, Period_6 and Period_8 to 0
    if i == 5 or i == 6 or i == 8:
        fitted_pars3 = ptrans_community_as, ptrans_other_as, ptrans_house_as, ptrans_work_as, ptrans_work_colleague_as, 0.0, 0.0, 0.0, ptrans_community_sy, ptrans_other_sy, ptrans_house_sy, ptrans_work_sy, ptrans_work_colleague_sy, 0.0, 0.0, 0.0, ptrans_quarantine, ptrans_ho
    else:
        fitted_pars3 = fitted_pars
    
    change_pars3 = change_pars3 + [list(fitted_pars3) + list(known_pars)] 
    
    
    # SIMULATION 5: repeat lockdown after summer - use prob. transmission of Period_2 also in Period_5
    if i == 5:
        last_start2, last_end2, ptrans_community_as2, ptrans_other_as2, ptrans_house_as2, ptrans_work_as2, ptrans_work_colleague_as2, ptrans_school_as2, ptrans_school_class_as2, ptrans_school_friend_as2, ptrans_community_sy2, ptrans_other_sy2, ptrans_house_sy2, ptrans_work_sy2, ptrans_work_colleague_sy2, ptrans_school_sy2, ptrans_school_class_sy2, ptrans_school_friend_sy2, ptrans_quarantine2, ptrans_ho2 = abm_params.loc[:, 'Period_2'].values
        fitted_pars5 = ptrans_community_as2, ptrans_other_as2, ptrans_house_as2, ptrans_work_as2, ptrans_work_colleague_as2, ptrans_school_as2, ptrans_school_class_as2, ptrans_school_friend_as2, ptrans_community_sy2, ptrans_other_sy2, ptrans_house_sy2, ptrans_work_sy2, ptrans_work_colleague_sy2, ptrans_school_sy2, ptrans_school_class_sy2, ptrans_school_friend_sy2, ptrans_quarantine2, ptrans_ho2
    else:
        fitted_pars5 = fitted_pars
    
    change_pars5 = change_pars5 + [list(fitted_pars5) + list(known_pars)]
    
    
#############################################################################
# RUN SIMULATIONS


cols = ['INFECTED_AS', 'INFECTED_SY', 'INFECTED_HO', 'DECEASED', 'QUARANTINED', 'POSITIVE', 'EXPOSED', 'REMOVED']


# SIMULATION 0
sim_records = dict([(c, []) for c in cols])
            
if __name__ == "__main__":
    with Pool(processes=n_sim) as pool: 
        #apply_async
        result_objects = [pool.apply_async(ABmodel.run_model, args=(N_abm, 
                             num_nodes_E, num_nodes_Ias, num_nodes_Isy, num_nodes_Iho, num_nodes_Q, num_nodes_R, num_nodes_D, 
                             days, change_times, change_pars, 'Solutions', False)) for run in range(n_sim)]
        
        # result_objects is a list of pool.ApplyResult objects
        results = [r.get() for r in result_objects]
        
for r in results:
    r['POSITIVE'] = r['INFECTED_AS'] + r['INFECTED_SY'] + r['INFECTED_HO'] + r['QUARANTINED']
    
    for c in cols:
        rec = list(r[c].values)
        sim_records[c].append(rec)
        
     
# SIMULATION 1
sim_records1 = dict([(c, []) for c in cols])

if __name__ == "__main__":
    with Pool(processes=n_sim) as pool: 
        #apply_async
        result_objects = [pool.apply_async(ABmodel.run_model, args=(N_abm, 
                             num_nodes_E, num_nodes_Ias, num_nodes_Isy, num_nodes_Iho, num_nodes_Q, num_nodes_R, num_nodes_D, 
                             days2, change_times, change_pars1, 'Solutions', False)) for run in range(n_sim)]
        
        # result_objects is a list of pool.ApplyResult objects
        results = [r.get() for r in result_objects]
        
for r in results:
    r['POSITIVE'] = r['INFECTED_AS'] + r['INFECTED_SY'] + r['INFECTED_HO'] + r['QUARANTINED']
    
    for c in cols:
        rec = list(r[c].values)
        sim_records1[c].append(rec)


# SIMULATION 2 
sim_records2 = dict([(c, []) for c in cols])
         
if __name__ == "__main__":
    with Pool(processes=n_sim) as pool: 
        #apply_async
        result_objects = [pool.apply_async(ABmodel.run_model, args=(N_abm, 
                             num_nodes_E, num_nodes_Ias, num_nodes_Isy, num_nodes_Iho, num_nodes_Q, num_nodes_R, num_nodes_D, 
                             days2, change_times, change_pars2, 'Solutions', False)) for run in range(n_sim)]
        
        # result_objects is a list of pool.ApplyResult objects
        results = [r.get() for r in result_objects]
        
for r in results:
    r['POSITIVE'] = r['INFECTED_AS'] + r['INFECTED_SY'] + r['INFECTED_HO'] + r['QUARANTINED']
    
    for c in cols:
        rec = list(r[c].values)
        sim_records2[c].append(rec)


# SIMULATION 3
sim_records3 = dict([(c, []) for c in cols])
          
if __name__ == "__main__":
    with Pool(processes=n_sim) as pool: 
        #apply_async
        result_objects = [pool.apply_async(ABmodel.run_model, args=(N_abm, 
                             num_nodes_E, num_nodes_Ias, num_nodes_Isy, num_nodes_Iho, num_nodes_Q, num_nodes_R, num_nodes_D, 
                             days2, change_times, change_pars3, 'Solutions', False)) for run in range(n_sim)]
        
        # result_objects is a list of pool.ApplyResult objects
        results = [r.get() for r in result_objects]
        
for r in results:
    r['POSITIVE'] = r['INFECTED_AS'] + r['INFECTED_SY'] + r['INFECTED_HO'] + r['QUARANTINED']
    
    for c in cols:
        rec = list(r[c].values)
        sim_records3[c].append(rec)
        
        
        
# SIMULATION 5
sim_records5 = dict([(c, []) for c in cols])

if __name__ == "__main__":
    with Pool(processes=n_sim) as pool:
        #apply_async
        result_objects = [pool.apply_async(ABmodel.run_model, args=(N_abm, 
                             num_nodes_E, num_nodes_Ias, num_nodes_Isy, num_nodes_Iho, num_nodes_Q, num_nodes_R, num_nodes_D, 
                             days2, change_times, change_pars5, 'Solutions', False)) for run in range(n_sim)]
        
        # result_objects is a list of pool.ApplyResult objects
        results = [r.get() for r in result_objects]
        
for r in results:
    r['POSITIVE'] = r['INFECTED_AS'] + r['INFECTED_SY'] + r['INFECTED_HO'] + r['QUARANTINED']
    
    for c in cols:
        rec = list(r[c].values)
        sim_records5[c].append(rec)

##########################################################################            
# PLOT scenarious          

               
real_sub = real_observations.iloc[1:end2, :]    
  
    
for c in col_to_consider:
    
    plt.figure(figsize=(8,5))
    
    # sim 1
    ma1 = np.mean(sim_records1[c] , axis = 0) * (4900000/N_abm)
    mstd1 = np.std(sim_records1[c], axis = 0) * (4900000/N_abm)
    ma1 = ma1[1:end2]
    mstd1 = mstd1[1:end2]
    plt.plot(range(1,end2), ma1, label = 'Simulated no measures', color = 'red')
    # approx CI for n = 10, t = 2.228
    plt.fill_between(range(1,end2), ma1 - (2.228/(n_sim**0.5)) * mstd1, ma1 + (2.228/(n_sim**0.5)) * mstd1, color = 'red', alpha=0.2);

    # sim 2
    ma2 = np.mean(sim_records2[c] , axis = 0) * (4900000/N_abm)
    mstd2 = np.std(sim_records2[c], axis = 0) * (4900000/N_abm)
    ma2 = ma2[1:end2]
    mstd2 = mstd2[1:end2]
    plt.plot(range(1,end2), ma2, label = 'Simulated no school closure March-June', color = 'orchid')
    # approx CI for n = 10, t = 2.228
    plt.fill_between(range(1,end2), ma2 - (2.228/(n_sim**0.5)) * mstd2, ma2 + (2.228/(n_sim**0.5)) * mstd2, color = 'orchid', alpha=0.2);

    # sim 3
    ma3 = np.mean(sim_records3[c] , axis = 0) * (4900000/N_abm)
    mstd3 = np.std(sim_records3[c], axis = 0) * (4900000/N_abm)
    ma3 = ma3[1:end2]
    mstd3 = mstd3[1:end2]
    plt.plot(range(1,end2), ma3, label = 'Simulated no school reopening September', color = 'orange')
    # approx CI for n = 10, t = 2.228
    plt.fill_between(range(1,end2), ma3 - (2.228/(n_sim**0.5)) * mstd3, ma3 + (2.228/(n_sim**0.5)) * mstd3, color = 'orange', alpha=0.2);

    # sim 5
    ma5 = np.mean(sim_records5[c] , axis = 0) * (4900000/N_abm)
    mstd5 = np.std(sim_records5[c], axis = 0) * (4900000/N_abm)
    ma5 = ma5[1:end2]
    mstd5 = mstd5[1:end2]
    plt.plot(range(1,end2), ma5, label = 'Simulated second lockdown after summer', color = 'limegreen')
    # approx CI for n = 10, t = 2.228
    plt.fill_between(range(1,end2), ma5 - (2.228/(n_sim**0.5)) * mstd5, ma5 + (2.228/(n_sim**0.5)) * mstd5, color = 'limegreen', alpha=0.2);


    # real
    plt.plot(range(1,end2),real_sub[c].values, label = 'Real ')


    plt.xlabel('days')
    plt.ylabel('cases')
    plt.legend(prop={'size': 8})   
    plt.title(c)
    
    for t in change_times:
        if t>1:
            plt.axvline(t, color="grey", linestyle="--", lw = 0.7)
            
    plt.savefig('ABM_exp_'+c+'.png')
    
    
    
    
    
for c in col_to_consider:
    
    plt.figure(figsize=(8,5))
    
    # sim 1
    ma1 = np.mean(sim_records1[c] , axis = 0) * (4900000/N_abm)
    mstd1 = np.std(sim_records1[c], axis = 0) * (4900000/N_abm)
    ma1 = ma1[1:end2]
    mstd1 = mstd1[1:end2]
    plt.plot(range(1,end2), ma1, label = 'Simulated no measures', color = 'red')
    # approx CI for n = 10, t = 2.228
    plt.fill_between(range(1,end2), ma1 - (2.228/(n_sim**0.5)) * mstd1, ma1 + (2.228/(n_sim**0.5)) * mstd1, color = 'red', alpha=0.2);

    # sim 5
    ma5 = np.mean(sim_records5[c] , axis = 0) * (4900000/N_abm)
    mstd5 = np.std(sim_records5[c], axis = 0) * (4900000/N_abm)
    ma5 = ma5[1:end2]
    mstd5 = mstd5[1:end2]
    plt.plot(range(1,end2), ma5, label = 'Simulated second lockdown after summer', color = 'limegreen')
    # approx CI for n = 10, t = 2.228
    plt.fill_between(range(1,end2), ma5 - (2.228/(n_sim**0.5)) * mstd5, ma5 + (2.228/(n_sim**0.5)) * mstd5, color = 'limegreen', alpha=0.2);


    # real
    plt.plot(range(1,end2),real_sub[c].values, label = 'Real ')


    plt.xlabel('days')
    plt.ylabel('cases')
    plt.legend(prop={'size': 8})   
    plt.title(c)
    
    for t in change_times:
        if t>1:
            plt.axvline(t, color="grey", linestyle="--", lw = 0.7)
            
    plt.savefig('ABM_exp_'+c+'1.png')
    
    
for c in col_to_consider:
    
    plt.figure(figsize=(8,5))
    
    # sim 2
    ma2 = np.mean(sim_records2[c] , axis = 0) * (4900000/N_abm)
    mstd2 = np.std(sim_records2[c], axis = 0) * (4900000/N_abm)
    ma2 = ma2[1:end2]
    mstd2 = mstd2[1:end2]
    plt.plot(range(1,end2), ma2, label = 'Simulated no school closure March-June', color = 'orchid')
    # approx CI for n = 10, t = 2.228
    plt.fill_between(range(1,end2), ma2 - (2.228/(n_sim**0.5)) * mstd2, ma2 + (2.228/(n_sim**0.5)) * mstd2, color = 'orchid', alpha=0.2);

    # sim 3
    ma3 = np.mean(sim_records3[c] , axis = 0) * (4900000/N_abm)
    mstd3 = np.std(sim_records3[c], axis = 0) * (4900000/N_abm)
    ma3 = ma3[1:end2]
    mstd3 = mstd3[1:end2]
    plt.plot(range(1,end2), ma3, label = 'Simulated no school reopening September', color = 'orange')
    # approx CI for n = 10, t = 2.228
    plt.fill_between(range(1,end2), ma3 - (2.228/(n_sim**0.5)) * mstd3, ma3 + (2.228/(n_sim**0.5)) * mstd3, color = 'orange', alpha=0.2);

    # real
    plt.plot(range(1,end2),real_sub[c].values, label = 'Real ')


    plt.xlabel('days')
    plt.ylabel('cases')
    plt.legend(prop={'size': 8})   
    plt.title(c)
    
    for t in change_times:
        if t>1:
            plt.axvline(t, color="grey", linestyle="--", lw = 0.7)
            
    plt.savefig('ABM_exp_'+c+'2.png')
    
    
    
'''
    # sim 0
    ma = np.mean(sim_records[c] , axis = 0) * (4900000/N_abm)
    mstd = np.std(sim_records[c], axis = 0) * (4900000/N_abm)
    ma = ma[1:end]
    mstd = mstd[1:end]
    plt.plot(range(1,end), ma, label = 'Fitted', color = 'orange')
    # approx CI for n = 10, t = 2.228
    plt.fill_between(range(1,end), ma - (2.228/(n_sim**0.5)) * mstd, ma + (2.228/(n_sim**0.5)) * mstd, color = 'orange', alpha=0.2)
    
'''

#####################################################################
# FORECAST 


cols = ['INFECTED_AS', 'INFECTED_SY', 'INFECTED_HO', 'DECEASED', 'QUARANTINED', 'POSITIVE', 'EXPOSED', 'REMOVED']


# use as initial values last solutions

num_nodes_E = int(np.mean(sim_records['EXPOSED'] , axis = 0)[-1])
num_nodes_Ias = int(np.mean(sim_records['INFECTED_AS'] , axis = 0)[-1])
num_nodes_Isy = int(np.mean(sim_records['INFECTED_SY'] , axis = 0)[-1])
# num_nodes_Iho = int(np.mean(sim_records['INFECTED_HO'] , axis = 0)[-1])
# num_nodes_Q = int(np.mean(sim_records['QUARANTINED'] , axis = 0)[-1])
num_nodes_R = int(np.mean(sim_records['REMOVED'] , axis = 0)[-1])
# num_nodes_D = int(np.mean(sim_records['DECEASED'] , axis = 0)[-1])


# adjust with real ones where known
num_nodes_Iho_t, num_nodes_Q_t, num_nodes_D_t, _ = real_observations.iloc[end,:].values 
num_nodes_Iho = int(num_nodes_Iho_t * (N_abm/4900000))
num_nodes_Q = int(num_nodes_Q_t * (N_abm/4900000))
num_nodes_D = int(num_nodes_D_t * (N_abm/4900000))


predict_days = 12 # days to predict



# SIMULATION 4: forecast
sim_records4 = dict([(c, []) for c in cols])

if __name__ == "__main__":
    with Pool(processes=n_sim) as pool:
        #apply_async
        result_objects = [pool.apply_async(ABmodel.run_model, args=(N_abm, 
                             num_nodes_E, num_nodes_Ias, num_nodes_Isy, num_nodes_Iho, num_nodes_Q, num_nodes_R, num_nodes_D, 
                             days + predict_days, change_times, change_pars, 'Solutions', False)) for run in range(n_sim)]
        
        # result_objects is a list of pool.ApplyResult objects
        results = [r.get() for r in result_objects]

        
for r in results:
    r['POSITIVE'] = r['INFECTED_AS'] + r['INFECTED_SY'] + r['INFECTED_HO'] + r['QUARANTINED']
    
    for c in cols:
        rec = list(r[c].values)
        sim_records4[c].append(rec)
        
        

# PLOT prediction results

ma4i = np.mean(sim_records4['INFECTED_HO'] , axis = 0) * (4900000/N_abm)
mstd4i = np.std(sim_records4['INFECTED_HO'], axis = 0) * (4900000/N_abm)

ma4i = ma4i[-(predict_days+2):]
mstd4i = mstd4i[-(predict_days+2):]

ma4i = ma4i - ma4i[0] + num_nodes_Iho_t


ma4d = np.mean(sim_records4['DECEASED'] , axis = 0) * (4900000/N_abm)
mstd4d = np.std(sim_records4['DECEASED'], axis = 0) * (4900000/N_abm)

ma4d = ma4d[-(predict_days+2):]
mstd4d = mstd4d[-(predict_days+2):]

ma4d = ma4d - ma4d[0] + num_nodes_D_t


print('Iho0, Iho1 true, Iho1 pred', real_observations.iloc[end, :]['INFECTED_HO'], real_observations.iloc[end+predict_days+1, :]['INFECTED_HO'], ma4i[-1])
print('D0, D1 true, D1 pred', real_observations.iloc[end, :]['DECEASED'], real_observations.iloc[end+predict_days+1, :]['DECEASED'], ma4d[-1])


fig, ax = plt.subplots(1,1, figsize=(8,5))
ax.plot(range(end,end+predict_days+2),ma4i, color='grey', linestyle='dashed', label = 'Predicted')
ax.plot(range(end-15,end+predict_days+2),real_observations.iloc[end-15:end+predict_days+2, :]['INFECTED_HO'].values, label = 'Real')
plt.fill_between(range(end,end+predict_days+2), ma4i - (2.228/(n_sim**0.5)) * mstd4i, ma4i + (2.228/(n_sim**0.5)) * mstd4i, color = 'grey', alpha=0.2);
plt.scatter([end, end+predict_days+1, end+predict_days+1], [real_observations.iloc[end, :]['INFECTED_HO'], real_observations.iloc[end+predict_days+1, :]['INFECTED_HO'], ma4i[-1]], c='grey')
ax.legend()

ax.set_title('Hospitalizations Prediction')
ax.set_xlabel('days')
ax.set_ylabel('cases')

plt.savefig('ABM_pred_INFECTED_HO.png')


fig, ax = plt.subplots(1,1, figsize=(8,5))
ax.plot(range(end,end+predict_days+2),ma4d, color='grey', linestyle='dashed', label = 'Predicted')
ax.plot(range(end-15,end+predict_days+2),real_observations.iloc[end-15:end+predict_days+2, :]['DECEASED'].values, label = 'Real')
plt.fill_between(range(end,end+predict_days+2), ma4d - (2.228/(n_sim**0.5)) * mstd4d, ma4d + (2.228/(n_sim**0.5)) * mstd4d, color = 'grey', alpha=0.2);
plt.scatter([end, end+predict_days+1, end+predict_days+1], [real_observations.iloc[end, :]['DECEASED'], real_observations.iloc[end+predict_days+1, :]['DECEASED'], ma4d[-1]], c='grey')
ax.legend()

ax.set_title('Deceases Prediction')
ax.set_xlabel('days')
ax.set_ylabel('cases')

plt.savefig('ABM_pred_DECEASED.png')

