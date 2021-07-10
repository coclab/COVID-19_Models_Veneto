# Utility functions for ABmodel

# imports
import time, enum, math
import numpy as np
import pandas as pd
import random
import networkx as nx
from scipy.sparse import csr_matrix
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pylab as plt
import pickle


############################################################################################################################

# Set parameters from literature data

# age distribution
# age group '0-2', '3-5','6-10','11-13','14-18', '19-24', '25-44', '45-64', '65-79', '80-84',' 85 +' 
age_veneto_tot = [105598, 118043, 222031, 141385, 232380, 285686, 1117212, 1521131, 777127, 181729, 176811] # from ISTAT
prob_age_group = np.array(age_veneto_tot)/np.sum(age_veneto_tot)


# gender proportion
prob_gender = [0.51,0.49] # F / M # from ISTAT - veneto


# family composition (fonte https://www.istat.it/it/files//2020/05/05_Veneto_Scheda.pdf)
prob_house_sizes = np.array([30.2, 29.6, 19.2, 15.0, 6.0])/100  # 1,2,3,4,5+


# presence of chronic pathologies
#2019 italia - https://www.statista.com/statistics/573014/number-of-people-affected-by-chronic-diseases-italy/#statisticContainer  
# 24555000 / 60360000
# by ages, 2014 Italian general practice registry - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6351821/
prob_pathological_by_age = {0 : [0.72,0.28], # NO / YES 
                     1 : [0.72,0.28],
                     2 : [0.72,0.28],
                     3 : [0.72,0.28],
                     4 : [0.72,0.28],
                     5 : [0.72,0.28],
                     6 : [0.72,0.28],    #< 45
                     7 : [0.41, 0.59],   # 45 - 65
                     8 : [0.16, 0.84],   # 65 - 80
                     9 : [0.084, 0.916], #> 80
                     10: [0.084, 0.916]}

  
# mean and std latent days by age (from https://www.tandfonline.com/doi/full/10.1080/09603123.2021.1905781) table 1 - 2
# std = np.sqrt( 3.41 * 2.28**2)
latent_days_by_age = {0: [7.0, 4.21],
                        1: [7.0, 4.21],
                        2: [7.0, 4.21],
                        3: [7.0, 4.21], 
                        4: [7.0, 4.21], # <= 18
                        5: [7.7, 4.21], 
                        6: [7.7, 4.21], 
                        7: [7.7, 4.21], # <= 64
                        8: [9, 4.21],
                        9: [9, 4.21],
                        10: [9, 4.21]} 



# Set parameters for observed from literature data
# estimates by age and pathology ( + status)


# probabilities probdict[(age, pat)] = (p_as, p_sy, p_ho, pdie_sy, pdie_ho)
probdict = {(0, 'NO'): (0.7847058823529411,
                          0.21529411764705886,
                          0.14936247723105764,
                          0.0,
                          0.0),
                         (0, 'SI'): (0.85, 0.15000000000000002, 0.6666666664444444, 0.0, 0.0),
                         (1, 'NO'): (0.8658574540927482,
                          0.13414254590725183,
                          0.018561484918750436,
                          0.0,
                          0.0),
                         (1, 'SI'): (0.9, 0.09999999999999998, 0.0, 0.0, 0.0),
                         (2, 'NO'): (0.8827593892789479,
                          0.11724061072105207,
                          0.016159695817475134,
                          0.0028517110266132587,
                          0.0),
                         (2, 'SI'): (0.95, 0.050000000000000044, 0.9999999989999999, 0.0, 0.0),
                         (3, 'NO'): (0.853431562621171,
                          0.14656843737882896,
                          0.014991181657835105,
                          0.0,
                          0.0),
                         (3, 'SI'): (0.78125, 0.21875, 0.1428571428367347, 0.0, 0.0),
                         (4, 'NO'): (0.7831858407079646,
                          0.2168141592920354,
                          0.010383100608660802,
                          0.0,
                          0.0),
                         (4, 'SI'): (0.7866666666666666,
                          0.21333333333333337,
                          0.31249999998046873,
                          0.0,
                          0.0),
                         (5, 'NO'): (0.7342540895288471,
                          0.2657459104711529,
                          0.019166973829705277,
                          0.0,
                          0.019230769230584317),
                         (5, 'SI'): (0.6759776536312849,
                          0.3240223463687151,
                          0.20689655172057075,
                          0.0,
                          0.0),
                         (6, 'NO'): (0.668356652517948,
                          0.33164334748205204,
                          0.040213312265453674,
                          0.0001580090855224113,
                          0.020628683693496436),
                         (6, 'SI'): (0.6405172413793103,
                          0.3594827586206897,
                          0.3021582733805704,
                          0.0,
                          0.04761904761866969),
                         (7, 'NO'): (0.633603409912405,
                          0.36639659008759495,
                          0.11733203505355104,
                          0.0011274534925434037,
                          0.07097619567589626),
                         (7, 'SI'): (0.5198237885462555,
                          0.48017621145374445,
                          0.545147271849085,
                          0.004345726702074193,
                          0.07705934455263325),
                         (8, 'NO'): (0.5720825479203897,
                          0.4279174520796103,
                          0.3802631578947151,
                          0.016990846681921225,
                          0.21994884910482623),
                         (8, 'SI'): (0.36688211757463834,
                          0.6331178824253616,
                          0.8035974720462793,
                          0.016042780748655304,
                          0.25105868118557106),
                         (9, 'NO'): (0.5231510961520968,
                          0.47684890384790324,
                          0.5776421213276273,
                          0.06619610835557684,
                          0.3622853368558909),
                         (9, 'SI'): (0.3664317745035234,
                          0.6335682254964766,
                          0.8483316481285659,
                          0.04448938321532408,
                          0.3873659117992999),
                         (10, 'NO'): (0.5255556177573756,
                          0.4744443822426244,
                          0.5666076696164523,
                          0.21203539823006345,
                          0.502290712203144),
                         (10, 'SI'): (0.43931012040351447,
                          0.5606898795964855,
                          0.7881601857221194,
                          0.13639001741141243,
                          0.5486008836520261)}

# recovery rate  gammadict[(age, pat)] = (mean_asintom, mean_sintom, mean_ricov, std_asintom, std_sintom, std_ricov)
gammadict = { (0, 'NO'): (14.605211433705845,
              15.78475499092559,
              19.275229357798164,
              12.006537766548494,
              12.147025954774971,
              19.3045147087761),
             (0, 'SI'): (16.957142857142856,
              12.375,
              13.5,
              8.666837870581649,
              5.629958132298016,
              8.504900548115382),
             (1, 'NO'): (14.605211433705845,
              15.78475499092559,
              19.275229357798164,
              12.006537766548494,
              12.147025954774971,
              19.3045147087761),
             (1, 'SI'): (16.957142857142856,
              12.375,
              13.5,
              8.666837870581649,
              5.629958132298016,
              8.504900548115382),
             (2, 'NO'): (14.605211433705845,
              15.78475499092559,
              19.275229357798164,
              12.006537766548494,
              12.147025954774971,
              19.3045147087761),
             (2, 'SI'): (16.957142857142856,
              12.375,
              13.5,
              8.666837870581649,
              5.629958132298016,
              8.504900548115382),
              (3, 'NO'): (14.605211433705845,
              15.78475499092559,
              19.275229357798164,
              12.006537766548494,
              12.147025954774971,
              19.3045147087761),
             (3, 'SI'): (16.957142857142856,
              12.375,
              13.5,
              8.666837870581649,
              5.629958132298016,
              8.504900548115382),
             (4, 'NO'): (16.29608269858542,
              17.64340770791075,
              20.0,
              13.53334056640032,
              14.248170812412955,
              22.577643809751272),
             (4, 'SI'): (19.948275862068964,
              16.545454545454547,
              24.0,
              15.737342353257032,
              4.435394827152062,
              21.644860821913362),
             (5, 'NO'): (16.59998526377837,
              16.720050441361916,
              21.161290322580644,
              14.77271451350533,
              12.082986014102936,
              16.95857505459076),
             (5, 'SI'): (20.389830508474578,
              22.75,
              22.272727272727273,
              14.838753776724,
              35.63844605491803,
              13.016073978668906),
             (6, 'NO'): (16.776425382105444,
              17.011923509561306,
              20.391025641025642,
              14.622761005920315,
              12.323322752790855,
              18.565302713106746),
             (6, 'SI'): (26.979536152796726,
              22.893617021276597,
              33.621848739495796,
              42.237500059306974,
              26.80726043036904,
              54.05834757293187),
             (7, 'NO'): (17.40169060881462,
              17.72051216552781,
              23.449789238780063,
              14.161579069809294,
              11.973449679913715,
              18.506930291908994),
             (7, 'SI'): (28.100904977375567,
              22.07431693989071,
              27.366346153846155,
              41.806645501085704,
              18.181944131306985,
              31.775400169533025),
             (8, 'NO'): (18.76971439711473,
              18.748343651569456,
              25.15568987077949,
              13.60927999026464,
              12.154608649880677,
              16.63507136333955),
             (8, 'SI'): (25.610154905335627,
              23.825214899713465,
              29.852820932134097,
              27.19629731365802,
              17.79257385037159,
              29.590156109740956),
             (9, 'NO'): (22.26783949397114,
              20.46637335009428,
              26.592762780011487,
              17.434314831018217,
              13.205724672835494,
              19.444117817671412),
             (9, 'SI'): (29.492857142857144,
              23.346938775510203,
              31.451292246520875,
              27.594883357766726,
              12.381481050606274,
              30.777137368211374),
             (10, 'NO'): (25.03426531826294,
              21.56602564102564,
              28.41854990583804,
              18.444231814927424,
              13.6038329730276,
              19.796906997717198),
             (10, 'SI'): (33.95886312640239,
              26.6,
              37.214405360134,
              35.64443857184119,
              13.216558529164791,
              42.40703523915584)}

##########################################################################################################################

# functions for agent personalized parameters depending on age, pathologies...

def personalized_death_prob(agent): # (age_group, gender, pathological):
    
    age, pat, status = agent.age_group, agent.pathological, agent.state
    
    p_as, p_sy, p_ho, pdie_sy, pdie_ho = probdict[(age, pat)] 
    
    if status == State.INFECTED_SY:
        return pdie_sy

    elif status == State.INFECTED_HO:
        return pdie_ho
    
    else: 
        return 0
    
    # LOGISTIC MODEL
    #clf_d = pickle.load(open('death_model.sav', 'rb'))
    #gender = int(gender == 'M')
    #pathological = int(pathological == 'YES')
    #clf_d.predict_proba(np.array([age_group, gender, pathological]).reshape(1, -1))[0][1]
    


def personalized_clinic_status_prob(agent): # (age_group, gender, pathological):
    
    age, pat, status = agent.age_group, agent.pathological, agent.state
    
    p_as, p_sy, p_ho, pdie_sy, pdie_ho = probdict[(age, pat)] 
    
    if status == State.EXPOSED:
        return [p_as, p_sy]

    elif status == State.INFECTED_SY:
        return p_ho
    
    else: 
        return 0
    
    
    
    # LOGISTIC MODEL
    #clf_s = pickle.load(open('clinic_status_model.sav', 'rb'))
    #gender = int(gender == 'M')
    #pathological = int(pathological == 'YES')
    #clf_s.predict_proba(np.array([age_group, gender, pathological]).reshape(1, -1))[0]

 
    
def get_recovery_time(agent): # days (1/rate) from I to R
    
    age, pat, status = agent.age_group, agent.pathological, agent.state
    
    mean_asintom, mean_sintom, mean_ricov, std_asintom, std_sintom, std_ricov = gammadict[(age, pat)]
    
    if status == State.INFECTED_AS:
        mean = mean_asintom
        std = std_asintom
    
    elif status == State.INFECTED_SY:
        mean = mean_sintom
        std = std_sintom

    elif status == State.INFECTED_HO:
        mean = mean_ricov
        std = std_ricov

    #age = agent.age_group
    #mean, std = recovery_days_by_age[age]
  
    sample = int(np.random.gamma(shape = (mean/std)**2, scale = std*std/mean ))
    
    return sample
    

def get_latent_time(agent): # days (1/rate) from E to I
    age = agent.age_group
    mean, std = latent_days_by_age[age]
    
    sample = int(np.random.gamma(shape = (mean/std)**2, scale = std*std/mean ))
    
    return sample

                    
##################################################################################################################################

class State(enum.IntEnum):
    SUSCEPTIBLE = 0
    EXPOSED = 1
    INFECTED_AS = 2 
    INFECTED_SY = 3
    INFECTED_HO = 4
    QUARANTINED = 5
    REMOVED = 6
    DECEASED = 7

    
    


# function to create model population as network of agents
                

def check_all_child(node_list, G):
    for idx in node_list:
        if G.nodes[idx]['age_group'] > 4:
            return False
    return True



def get_contacts(i, i_pot_cont, mean_cont_i , sd_cont_i):
    '''
    Find contacts of the node i
    :param i -> The index of the node on which we are trying to infer contacts
    :param i_pot_cont -> The list of potential conctact of i
    :param mean_cont_i -> The average conctacts I would expect for node i
    :param sd_cont_i -> The contacts' sd I would expect for node i
    :return -> A list with the actual close contacts of i 
    '''
    n_conct = int(np.random.normal(loc=mean_cont_i, scale=sd_cont_i))  
    while n_conct < 0:
        n_conct =  int(np.random.normal(loc=mean_cont_i, scale=sd_cont_i))  
    if n_conct == 0:
        friends = []
    else:
        friends = random.sample(i_pot_cont, k=min((n_conct,len(i_pot_cont))))
    return friends+[i]




def create_world_net(num_nodes = 100, 
                     min_school = 100, max_school= 500, min_class = 15, max_class = 30, mean_friens = 2, std_friens = 1.5, 
                     min_work = 4, max_work = 16, mean_colleagues = 3, std_colleagues = 2.5, other_min_edges = 10):
    
    '''
    Example of desired node list format for NetworkX
   
        [
            ('A', {'age': 34, 'gender': 'M', 'pathological': 'YES'}),
            ('B', {'age': 67, 'gender': 'F', 'pathological': 'NO'}),
            ('C', {'age': 12, 'gender': 'M', 'pathological': 'YES'})
        ]


    Example of desired edge list format for NetworkX

        [
          ('A', 'B', {'contact_type': 'house'}), 
          ('A', 'C', {'contact_type': 'school'})
        ]
     '''

    
    # create empty NetworkX object graph
    G = nx.Graph() 


    # CREATE NODES

    model_nodes = [(i , # node id 
                    { 'age_group': int(np.random.choice(range(0,11), p=prob_age_group))           # node attributes
                     }) 
                   for i in range(num_nodes)]
    
    child = []
    adults = []
    for n in model_nodes:
        if n[1]['age_group']<=4:
            child.append(n[0]) # append idx
        else:
            adults.append(n[0]) # append idx



    # add nodes, i.e. subjects
    G.add_nodes_from(model_nodes)

    
    # CREATE EDGES
    model_edges = []
    
    # OTHER CONNECTIONS
    G_barabasi = nx.barabasi_albert_graph(len(G.nodes), m = other_min_edges, seed = 42)
    edges = list(G_barabasi.edges())
    pot_edges = [(g[0],g[1], {'contact_type' : 'other'}) for g in edges]
    model_edges += [g for g in pot_edges]
    
    
    # WORK CONNECTIONS (complete graph)
    
    lista_eta = [6,7]
    nodes_age_e = [i for i in range(num_nodes) if (G.nodes[i]['age_group'] in lista_eta)]
    
    for i in nodes_age_e:
        if i not in nodes_age_e:
            pass
        else:
            nodes_age_e.remove(i)

            work_size = int(np.random.uniform(min_work,max_work))

            if len(nodes_age_e) > 0 :

                # assign work contacts
                work_ids = random.sample(nodes_age_e, np.min((work_size - 1, len(nodes_age_e))))
                nodes_age_e = list(set(nodes_age_e).difference(set(work_ids)))
                
                G_complete = nx.complete_graph(work_ids+[i])
                model_edges += [(g[0], g[1], {'contact_type': 'work'}) for g in list(G_complete.edges())]


                colleagues = get_contacts(i, work_ids, mean_colleagues, std_colleagues)
                work_ids = list(set(work_ids).difference(set(colleagues)))
                
                if len(colleagues)>1: #otherwise create loop
                    G_complete = nx.complete_graph(colleagues)
                    model_edges += [(g[0], g[1], {'contact_type': 'work_colleague'}) for g in list(G_complete.edges())]


                while len(work_ids) > 0: 
                     j = random.sample(work_ids, k = 1)[0]
                     colleagues = get_contacts(j, work_ids, mean_colleagues, std_colleagues)
                     work_ids = list(set(work_ids).difference(set(colleagues)))
                        
                     if len(colleagues)>1: #otherwise create loop
                         G_complete = nx.complete_graph(colleagues)
                         model_edges += [(g[0], g[1], {'contact_type': 'work_colleague'}) for g in list(G_complete.edges())]
                        
                        
                        
    # SCHOOL CONNECTIONS 
    
    
    lista_eta = range(1,6)

    nodes_age_e = [i for i in range(num_nodes) if (G.nodes[i]['age_group'] in lista_eta)]

    for i in nodes_age_e:
        if i not in nodes_age_e:
            pass
        else:
            nodes_age_e.remove(i)
            school_size = int(np.random.uniform(min_school, max_school))
            if len(nodes_age_e) > 0 :
                school_ids = random.sample(nodes_age_e, np.min((school_size - 1, len(nodes_age_e))))
                G_sch_complete = nx.complete_graph(school_ids+[i])
                nodes_age_e = list(set(nodes_age_e).difference(set(school_ids)))
                model_edges += [(g[0], g[1], {'contact_type': 'school'}) for g in list(G_sch_complete.edges())]
                nodes_age_e_school = list(G_sch_complete.nodes())
                for j in nodes_age_e_school:
                    if j in nodes_age_e_school:
                       
                        nodes_age_e_school.remove(j)
                        class_size = int(np.random.uniform(min_class,max_class))

                        if len(nodes_age_e_school) > 0 :

                            # assign class
                            class_ids = random.sample(nodes_age_e_school, np.min((class_size - 1, len(nodes_age_e_school))))
                            nodes_age_e_school = list(set(nodes_age_e_school).difference(set(class_ids)))
                            G_complete = nx.complete_graph(class_ids+[j])
                            model_edges += [(g[0], g[1], {'contact_type': 'school_class'}) for g in list(G_complete.edges())]

                            friends = get_contacts(j, class_ids, mean_friens, std_friens)
                            class_ids = list(set(class_ids).difference(set(friends)))
                            
                            if len(friends)>1: #otherwise create loop
                                G_complete = nx.complete_graph(friends)
                                model_edges += [(g[0], g[1], {'contact_type': 'school_friend'}) for g in list(G_complete.edges())]


                            while len(class_ids) > 0: 
                                 j = random.sample(class_ids, k = 1)[0]
                                 friends = get_contacts(j, class_ids, mean_friens, std_friens)
                                 class_ids = list(set(class_ids).difference(set(friends)))
                                    
                                 if len(friends)>1: #otherwise create loop
                                    G_complete = nx.complete_graph(friends)
                                    model_edges += [(g[0], g[1], {'contact_type': 'school_friend'}) for g in list(G_complete.edges())]


    # HOUSE CONNECTIONS (complete graph)
      
    tot = 0
    fam1 = 0
    fam2 = 0
    fam3 = 0
    fam4 = 0
    fam5 = 0

    while tot < num_nodes:
        size = np.random.choice([1,2,3,4,5], p=prob_house_sizes)

        if size == 1: fam1+=1
        elif size == 2: fam2+=2
        elif size == 3: fam3+=3
        elif size == 4: fam4+=4
        else: fam5+=5

        tot = fam1 + fam2 + fam3 + fam4 + fam5  
    
    # split nodes in the different families
    
    node_in_family = adults[fam1::] + child # remaining adults are single
    random.shuffle(node_in_family)
    
    node_ids_2 = node_in_family[0:fam2]
    node_ids_3 = node_in_family[fam2:fam2+fam3]
    node_ids_4 = node_in_family[fam2+fam3:fam2+fam3+fam4]
    node_ids_5 = node_in_family[fam2+fam3+fam4:fam2+fam3+fam4+fam5]
    
    
    fail = 0
    
    for n, s in zip([node_ids_2, node_ids_3, node_ids_4, node_ids_5] , range(2,6)):
        
        i = 0
        aux = [len(n)+1, len(n)]
        while len(n)>= s:
            i += 1
        
            # assign household components
            house_ids = random.sample(n, s)
            
            if not check_all_child(house_ids, G):
                if aux[i] != aux[i-1]:
                    n = list(set(n) - set(house_ids))
                else:    
                    model_nodes = [( num_nodes + fail, # node id 
                    { 'age_group': np.random.choice(range(5,8), p=np.array(prob_age_group[5:8])/np.sum(prob_age_group[5:8]))  
                     })]
    
                    G.add_nodes_from(model_nodes)
                    house_ids.append(fail+num_nodes)
                    fail+=1
            
            G_complete = nx.complete_graph(house_ids)
            model_edges += [(i[0], i[1], {'contact_type': 'house'}) for i in list(G_complete.edges())]
                    
            n = list(set(n) - set(house_ids))
            
            aux.append(len(n))
                           
    # create empty NetworkX object graph
    G = nx.Graph() 

    # add nodes, i.e. subjects
    G.add_nodes_from(model_nodes)

    # add edges, i.e. subjects' contacts
    G.add_edges_from(model_edges)
    
    return G
    


##################################################################################################################################

# functions to keep trace, plot or describe ABM simulation results

def update_contagion_matrix(infector_agent, infected_agent):
                        # update the model matrix containing infection info
                    infector_agent.model.row.append(infected_agent.unique_id)
                    infector_agent.model.col.append(infected_agent.unique_id)
                    infector_agent.model.data.append(np.ceil(infector_agent.model.schedule.time) + 2) # ceiling of the number 
                                                                                                      # because in multistaged 
                                                                                                      # scheduler time is divided 
                                                                                                      # in portion, not integer
                    
                    
                    infector_agent.model.row.append(infector_agent.unique_id)
                    infector_agent.model.col.append(infected_agent.unique_id)
                    infector_agent.model.data.append(np.ceil(infector_agent.model.schedule.time) + 2)


def get_column_data(model):
    #pivot the model dataframe to get states count at each step
    agent_state = model.datacollector.get_agent_vars_dataframe()
    X = pd.pivot_table(agent_state.reset_index(),index='Step',columns='State',aggfunc=np.size,fill_value=0)    

    translate = {0 : 'SUSCEPTIBLE',
             1: 'EXPOSED',
             2: 'INFECTED_AS',
             3: 'INFECTED_SY', 
             4:   'INFECTED_HO',
             5: 'QUARANTINED',
             6: 'REMOVED',
             7: 'DECEASED'}
    

    labels = [translate[c[1]] for c in X.columns]
    
    X.columns = labels
    return X
    
def plot_states(model,ax, drop_S_R = False):    
    steps = model.schedule.steps
    X = get_column_data(model)
    
    if drop_S_R:
        X = X.drop(['SUSCEPTIBLE', 'REMOVED'], axis = 1) # to drop susc and rem from visualization
    
    color_dict = { 'SUSCEPTIBLE' : "lightblue",
    'EXPOSED': "purple",
    'INFECTED_AS' : 'yellow' ,
    'INFECTED_SY' : 'orange',
    'INFECTED_HO' : 'red',
    'QUARANTINED' : "grey",
    'REMOVED' : "green",
    'DECEASED': "black"}
    

    X.plot(ax=ax,lw=3, color=[color_dict.get(x, '#333333') for x in X.columns], alpha=0.8)
    return 


def plot_grid(model,fig,layout='kamada-kawai',title=''):
    graph = model.G
    if layout == 'kamada-kawai':      
        pos = nx.kamada_kawai_layout(graph)  
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    else:
        pos = nx.spring_layout(graph, iterations=5, seed=8)  
    plt.clf()
    ax=fig.add_subplot()
    
    cmap = ListedColormap(["lightblue","purple", 'yellow', "orange", 'red', "grey" ,"green", "black"])
    states = [int(i.state) for i in model.grid.get_all_cell_contents()]
    colors = [cmap(i) for i in states]
    
    cmap_ed = {'house': 'orange', 'work':'grey' , 'school_class':'lightgreen', 'work_colleague': 'black',
               'school_friend':'green', 'other': 'purple', 'school': 'yellow' }
    
    wmap_ed = {'house': 1, 'work':0.4 , 'school_class':0.4, 'work_colleague': 0.8,
               'school_friend':0.8, 'other': 0.4, 'school': 0.2 }
    
    ed_colors = [cmap_ed[graph[u][v]['contact_type']] for u,v in graph.edges() ]

    ed_widths = [wmap_ed[graph[u][v]['contact_type']] for u,v in graph.edges() ]
    
    nx.draw(graph, pos, node_size=80, edge_color=ed_colors, width = ed_widths, node_color=colors, #with_labels=True,
            alpha=0.9,font_size=14,ax=ax)


    ax.set_title(title)
    return


def final_size(row, col, data, dims):
    
    a = csr_matrix((data, (row, col)), shape=(dims, dims)).toarray()
    diag = np.diagonal(a)
    
    return 100 * np.sum(diag > 0)/dims


def R_t(row, col, data, dims, t):
    
    a = csr_matrix((data, (row, col)), shape=(dims, dims)).toarray()
    diag = np.diagonal(a)
    
    infector_id = [idx for idx in range(len(diag)) if diag[idx] == t]
    
    if len(infector_id) == 0:
        r_t = 0
    else:
        r_t = (np.sum(a[infector_id, :]>0) - len(infector_id))/len(infector_id)
    
    
    return r_t



def generation_time_tot(row, col, data, dims):
    
    a = csr_matrix((data, (row, col)), shape=(dims, dims)).toarray()
    diag = np.diagonal(a)
    
    a_no_diag = a - np.diag(diag)
    
    infector_id =[idx for idx in range(len(a)) if np.sum(a_no_diag[idx, :]) > 0]
    
    if len(infector_id) == 0:
        return np.nan
        
    else: 
        gt = []

        for idx in infector_id:
            time_diff = a_no_diag[idx, :] - diag[idx]
            time_diff = time_diff[time_diff > 0]

            gt+= list(time_diff)

        return np.mean(gt)
    
    
def generation_time_t(row, col, data, dims, t):
    
    a = csr_matrix((data, (row, col)), shape=(dims, dims)).toarray()
    diag = np.diagonal(a)
    
    a_no_diag = a - np.diag(diag)
    
    infector_id =[idx for idx in range(len(a)) if (np.sum(a_no_diag[idx, :]) > 0 and diag[idx] == t) ]
    
    if len(infector_id) == 0:
        return np.nan
        
    else:
        gt = []

        for idx in infector_id:
            time_diff = a_no_diag[idx, :] - diag[idx]
            time_diff = time_diff[time_diff > 0]

            gt+= list(time_diff)

        return np.mean(gt)
    
    
def compute_averagedegree(G):
    aux = 0
    for i in G.nodes():
        aux += G.degree[i]
    return aux/len(G.nodes)
        