# Network agent based infection model

# imports
from ABmodel_utils import *
import time, enum, math
import numpy as np
import pandas as pd
import random
import networkx as nx
from mesa import Agent, Model
from mesa.time import StagedActivation 
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector



    
def initialize_EXPOSED(agent, infector_agent = None, time = None):
    
    """ Initialize attributes of exposed agent """
    
    agent.state = State.EXPOSED
    agent.latent_time = get_latent_time(agent)
    agent.infection_time = agent.model.schedule.time

    try: update_contagion_matrix(infector_agent, agent)
    except: # case without infector id (start)
    
        # update matrix of contagion
        agent.model.row.append(agent.unique_id)
        agent.model.col.append(agent.unique_id)
        agent.model.data.append(time + 2)
    
    
def initialize_INFECTED(agent):
    
    """ Initialize attributes of infected agent """
    
    if agent.state == State.INFECTED_AS:
        
        # recovery time
        agent.recovery_time = get_recovery_time(agent)
        
        # prob to be quarantined
        pquar = agent.model.pquar_as

        # quarantined flag
        agent.quarantined = np.random.choice([0,1], p=[1-pquar, pquar]) # 1 will be quarantined
                                                                                             

    elif agent.state == State.INFECTED_SY:
        
        # recovery time
        agent.recovery_time = get_recovery_time(agent)
        
        # prob to be hospitalized
        phosp = personalized_clinic_status_prob(agent)
        
        # prob to die
        pdie = personalized_death_prob(agent)
        
        # prob to be quarantined
        pquar = agent.model.pquar_sy
        
        # prob to remove
        if (pquar +  phosp + pdie) >= 1:          # the quar probability is a model parameter, the others depend on the agent
                                                  # recall that phosp + pdie always <= 1 by def
                                                  # in case the sum of probs is > 1, restrict prob to quarantine and set prob
                                                  # to remove to 0
            pquar = 1 - (phosp + pdie)
            
            prem = 0
            
        else:
            prem = 1 - (pquar + phosp + pdie)
             
            

        # get mutally exclusive status: quarantined, removed, hospitalized, deceased 
        status = np.array(np.random.choice(['1,1,0','0,1,0','0,1,1','0,0,0'], 
                                           p=[pquar, prem, phosp, pdie]).split(",")).astype(int)
        
        agent.quarantined, agent.alive, agent.hospitalized = status  # if quarantined = 1,  will be quarantined
                                                                     # if alive = 0 will die 
                                                                     # if hospitalized = 1, will be hospitalized
                                                                 
    elif agent.state == State.INFECTED_HO:
        
        # recovery time
        agent.recovery_time = get_recovery_time(agent)  # updated for hospitalized
        
        # prob to die
        pdie = personalized_death_prob(agent)
        
        # alive flag
        agent.alive = np.random.choice([0,1], p=[pdie, 1 - pdie]) # 0 will die

          

def random_susc_contact(a, mean_cont = 5, sd_cont = 1): # a is MyAgent object

    """ Return random susceptiple agents to potentially infect """
    
    neighbors_pos = a.model.grid.get_neighbors(a.pos, include_center=False)
    neighbors_nodes_id = [agent.unique_id for agent in a.model.grid.get_cell_list_contents(neighbors_pos)]
    
    
    all_nodes_not_neigh_id = list(set(range(a.model.num_nodes)) - set(neighbors_nodes_id)) # all random connection 
                                                                                           # not already node contacts in graph
    l = len(all_nodes_not_neigh_id)
    k = int(np.random.normal(loc=mean_cont, scale=sd_cont))   #random.randint(np.min((min_cont, l)), np.min((max_cont, l)))
    k = np.min([np.max([k,0]), l])
    rcontacts_id = random.sample(all_nodes_not_neigh_id, k)
    
    all_nodes_agent = a.model.grid.get_all_cell_contents()
    rcontacts = [all_nodes_agent[i] 
                     for i in rcontacts_id ]
    
    # return only susceptible random connections
    rcontacts_susc = [agent 
                     for agent in rcontacts
                     if agent.state is State.SUSCEPTIBLE ]
    
    return rcontacts_susc



    
###############################################################################################################


# DEFINE AGENT

class MyAgent(Agent):
    
    """ define the agent of the epidemic model."""
    def __init__(self, unique_id, model, age_group):
        
        super().__init__(unique_id, model) # inherit 'mesa agent' class attribute and model attributes
        
        # agent attributes
        self.age_group = age_group
        self.pathological = str(np.random.choice(['NO','SI'], p=prob_pathological_by_age[age_group]))
        
        # initialize the time of infection to zero
        self.infection_time = 0
        
        # initialize state to susceptible
        self.state = State.SUSCEPTIBLE  
        
    
    def update_status(self):
        """Check and update infection status"""
        
        
        if self.state == State.INFECTED_AS: 
            
            
            t = self.model.schedule.time-self.infection_time
        
     
            # infected asymptomatic subjects can be uncovered and quarantined
            if self.quarantined == 1:
                if t >= self.model.time_to_quarantine_as: # check rate: otherwise stay infectious               
                    self.state = State.QUARANTINED
                    self.start_quarantine_time = self.model.schedule.time 
            else:
                # otherwise infected asymptomatic subjects recovers after a certain time    
                if t >= self.recovery_time:   
                    self.state = State.REMOVED
        
        
   
        elif self.state == State.INFECTED_SY: 
            
            
            t = self.model.schedule.time-self.infection_time
        
            # infected symptatic subjects can die due to the disease with a certain rate
            if self.alive == 0:
                if t >= self.model.time_to_death: # check rate: otherwise stay infectious
                    self.state = State.DECEASED 
                
            
            # infected symptatic subjects can be uncovered and quarantined
            elif self.quarantined == 1:
                if t >= self.model.time_to_quarantine_sy: # check rate: otherwise stay infectious               
                    self.state = State.QUARANTINED
                    self.start_quarantine_time = self.model.schedule.time 
                    
                    
            # infected symptatic subjects can be hospitalized
            elif self.hospitalized == 1:
                if t >= self.model.time_to_hospitalization: # check rate: otherwise stay infectious               
                    self.state = State.INFECTED_HO
                    initialize_INFECTED(self)
                    
            else:
                # otherwise (all types) subjects recovers after a certain time    
                if t >= self.recovery_time:   
                    self.state = State.REMOVED
                    
        elif self.state == State.INFECTED_HO: 
            
            
            t = self.model.schedule.time-self.infection_time
        
            # infected hospitalized subjects can die due to the disease with a certain rate
            if self.alive == 0:
                if t >= self.model.time_to_death: # check rate: otherwise stay infectious
                    self.state = State.DECEASED 
                
            else:
                # otherwise infected hospitalized subjects recovers after a certain time    
                if t >= self.recovery_time:   
                    self.state = State.REMOVED
                    
                

            
        elif self.state == State.EXPOSED: 
            t = self.model.schedule.time-self.infection_time
            
            # subjects exposed to virus become infective after a certain time
            if t >= self.latent_time:   
                
                prob_inf = personalized_clinic_status_prob(self) 
                
                self.state = np.random.choice([State.INFECTED_AS, State.INFECTED_SY], 
                                              p=prob_inf) 
                initialize_INFECTED(self)
                
        elif self.state == State.QUARANTINED: 
            
            # quarantined subjects move to removed when the isolation period ends
            t = self.model.schedule.time-self.start_quarantine_time
            if t >= self.model.quarantine_time:
                self.state = State.REMOVED
                

    def infect(self, allowed_contacts, trans_probs, random_contacts_mean = 0):
        
        """ Find allowed close contacts and infect """
        
        # find susceptible contacts in GRAPH (work, school, house...)
        if allowed_contacts != ['random']: # check that also other type are allowed
            neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
            susceptible_neighbors = [
                                        agent
                                        for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)
                                        if agent.state is State.SUSCEPTIBLE
                                    ]

            for a in susceptible_neighbors:
                # get the connection type from the original society graph
                contact_type = self.model.G[self.unique_id][a.unique_id]['contact_type']

                if contact_type in allowed_contacts:
                    # infect depending on the probability for the specific type of connection
                    ptrans = trans_probs[allowed_contacts.index(contact_type)]
                    if self.random.random() < ptrans:
                            initialize_EXPOSED(a, self)
                            self.model.inf_counter[contact_type] +=1
                    
        # find susceptible contacts RANDOM
        if 'random' in allowed_contacts:
            susceptible_random_contacts = random_susc_contact(self, mean_cont = random_contacts_mean) 
                    
            if len(susceptible_random_contacts):
                for a in susceptible_random_contacts:
                    # infect depending on the probability 
                    ptrans = trans_probs[allowed_contacts.index('random')]
                    if self.random.random() < ptrans:
                        initialize_EXPOSED(a, self)
                        self.model.inf_counter['random'] +=1 
        
                    

        
    def day_contacts(self):
        
        """ Simulate workind day contacts potentially infective """
        
        if self.state == State.INFECTED_AS:
            self.infect(allowed_contacts = ['house', 'work', 'work_colleague', 'school', 
                                            'school_class', 'school_friend', 'other', 'random'],
                       trans_probs = [self.model.ptrans_house_as, 
                                      self.model.ptrans_work_as, self.model.ptrans_work_colleague_as,
                                      self.model.ptrans_school_as, self.model.ptrans_school_class_as, self.model.ptrans_school_friend_as, 
                                      self.model.ptrans_other_as, self.model.ptrans_community_as], 
                       random_contacts_mean = 5)
            
            

        elif self.state == State.INFECTED_SY:
            self.infect(allowed_contacts = ['house', 'work', 'work_colleague', 'school', 
                                            'school_class', 'school_friend', 'other', 'random'],
                       trans_probs = [self.model.ptrans_house_sy, 
                                      self.model.ptrans_work_sy, self.model.ptrans_work_colleague_sy,
                                      self.model.ptrans_school_sy, self.model.ptrans_school_class_sy, self.model.ptrans_school_friend_sy, 
                                      self.model.ptrans_other_sy, self.model.ptrans_community_sy], 
                       random_contacts_mean = 5)
            
        elif self.state == State.INFECTED_HO: 
            self.infect(allowed_contacts = ['other'],
                        trans_probs = [self.model.ptrans_ho])
            
        elif self.state == State.QUARANTINED: 
            self.infect(allowed_contacts = ['house'],
                        trans_probs = [self.model.ptrans_quarantine])
            
    
            
         
    '''    
    def night_contacts(self):
        if self.state == State.INFECTED_AS:
            self.infect(allowed_contacts = ['house', 'other', 'random'],
                       trans_probs = [self.model.ptrans_house_as, self.model.ptrans_other_as, self.model.ptrans_community_as], 
                       random_contacts_mean = 3)
            
        if self.state == State.INFECTED_SY:
            self.infect(allowed_contacts = ['house', 'other', 'random'],
                       trans_probs = [self.model.ptrans_house_sy, self.model.ptrans_other_sy, self.model.ptrans_community_sy], 
                       random_contacts_mean = 3)
            
        if self.state == State.INFECTED_HO: 
            self.infect(allowed_contacts = ['other'],
                        trans_probs = [self.model.ptrans_ho])
            
        if self.state == State.QUARANTINED: 
            self.infect(allowed_contacts = ['house'],
                        trans_probs = [self.model.ptrans_quarantine])
     '''
        

        
   
    
    
###############################################################################################################

# DEFINE AGENT BASED MODEL

class NetworkInfectionModel(Model):
    """ Define network model for infection spread."""
    
    def __init__(self, 
                 
                 # number of agents
                 num_nodes=100, 
                 
                 # initial number of agents in each compartment
                 num_nodes_E=0, 
                 num_nodes_Ias=1, 
                 num_nodes_Isy=1, 
                 num_nodes_Iho=0, 
                 num_nodes_Q=0,
                 num_nodes_R=0, 
                 num_nodes_D=0, 
                 
                 
                 # transmission probability Asymptomatic
                 ptrans_community_as= 0.005,
                 ptrans_other_as = 0.007,
                 ptrans_house_as = 0.01,
                 ptrans_work_as = 0.01,
                 ptrans_work_colleague_as = 0.03,
                 ptrans_school_as = 0.007,
                 ptrans_school_class_as = 0.01,
                 ptrans_school_friend_as = 0.03,
                 
                 
                 
                 # transmission probability Symptomatic
                 ptrans_community_sy = 0.005,
                 ptrans_other_sy = 0.007,
                 ptrans_house_sy = 0.01,
                 ptrans_work_sy = 0.01,
                 ptrans_work_colleague_sy = 0.03,
                 ptrans_school_sy = 0.007,
                 ptrans_school_class_sy = 0.01,
                 ptrans_school_friend_sy = 0.03,
                 
                 # transmission probability Quarantined
                 ptrans_quarantine = 0.0001,
                 
                 # transmission probability Hospitalized
                 ptrans_ho = 0.0001,
                 
                 # probability to be quarantined 
                 pquar_as = 0.07, 
                 pquar_sy = 0.10,
                 
                 # days of quartine (by law)
                 quarantine_time = 14,
                 
                 #hospitalization rate (days from syntomatic to hospitalization)
                 time_to_hospitalization = 1/0.81,
        
                 # death rate (days from infection to death)
                 time_to_death = 1/0.04,
            
                 # quarantined rate (days from infection to quarantine)
                 time_to_quarantine_as = 1/0.49,
                
                 time_to_quarantine_sy = 1/0.26):
                 
            
            
        
        
        
        
        # INITIALIZE MODEL 
        
        # initialize model attributes
        self.num_nodes = num_nodes  
        
        self.ptrans_community_as = ptrans_community_as
        self.ptrans_other_as = ptrans_other_as
        self.ptrans_house_as = ptrans_house_as        
        self.ptrans_work_as = ptrans_work_as
        self.ptrans_work_colleague_as = ptrans_work_colleague_as
        self.ptrans_school_as = ptrans_school_as 
        self.ptrans_school_class_as = ptrans_school_class_as
        self.ptrans_school_friend_as = ptrans_school_friend_as
        
        self.ptrans_community_sy = ptrans_community_sy
        self.ptrans_other_sy = ptrans_other_sy
        self.ptrans_house_sy = ptrans_house_sy        
        self.ptrans_work_sy = ptrans_work_sy
        self.ptrans_work_colleague_sy = ptrans_work_colleague_sy
        self.ptrans_school_sy = ptrans_school_sy
        self.ptrans_school_class_sy = ptrans_school_class_sy
        self.ptrans_school_friend_sy = ptrans_school_friend_sy
        
        self.ptrans_quarantine = ptrans_quarantine
        self.ptrans_ho = ptrans_ho
        
        self.pquar_as = pquar_as
        self.pquar_sy = pquar_sy
        
        self.quarantine_time = quarantine_time
        
        self.time_to_death = time_to_death
        self.time_to_hospitalization = time_to_hospitalization
        self.time_to_quarantine_as = time_to_quarantine_as
        self.time_to_quarantine_sy = time_to_quarantine_sy

        # initialize contagion matrix trace
        self.row = []
        self.col = []
        self.data = []
        
        # initialize infection counter
        
        self.inf_counter = {'house':0, 
                       'work':0,
                       'school':0,
                       'school_class' : 0, 
                       'work_colleague':0,
                       'school_friend':0,
                       'random':0,
                       'other':0}
        
        
        # create meta population graph
        self.G = create_world_net(num_nodes,
                     min_school = 100, max_school= 500, min_class = 15, max_class = 30, mean_friens = 2, std_friens = 1.5, 
                     min_work = 4, max_work = 16, mean_colleagues = 3, std_colleagues = 2.5, other_min_edges = 10)
    
        
        # ABM model settings
        self.grid = NetworkGrid(self.G)
        self.schedule = StagedActivation(model = self,  stage_list = ['update_status', 'day_contacts'], #, 'night_contacts'],  
                                         shuffle = True, shuffle_between_stages = False) #RandomActivation(model = self)
        self.running = True
        self.datacollector = DataCollector(agent_reporters={"State": "state"})

        

        # CREATE AGENTS 
        
        # sample agents id to set with status != SUSCEPTIBLE
        not_susceptible_tot = num_nodes_E + num_nodes_Ias + num_nodes_Isy + num_nodes_Iho + num_nodes_R + num_nodes_Q + num_nodes_D 
        not_susceptible_agents = random.sample(range(self.num_nodes), not_susceptible_tot)
        
        #make some agents exposed at start
        exposed_agents = not_susceptible_agents[0:num_nodes_E]
        
        #make some agents infected at start
        infected_agents_as = not_susceptible_agents[num_nodes_E:num_nodes_E+num_nodes_Ias]
        infected_agents_sy = not_susceptible_agents[num_nodes_E+num_nodes_Ias:num_nodes_E+num_nodes_Ias+num_nodes_Isy]
        infected_agents_ho = not_susceptible_agents[num_nodes_E+num_nodes_Ias+num_nodes_Isy:
                                                    num_nodes_E+num_nodes_Ias+num_nodes_Isy+num_nodes_Iho]
        
        #make some agents recovered at start
        recovered_agents = not_susceptible_agents[num_nodes_E+num_nodes_Ias+num_nodes_Isy+num_nodes_Iho:
                                                  num_nodes_E+num_nodes_Ias+num_nodes_Isy+num_nodes_Iho+num_nodes_R]
        
        #make some agents quarantined at start
        quarantined_agents = not_susceptible_agents[num_nodes_E+num_nodes_Ias+num_nodes_Isy+num_nodes_Iho+num_nodes_R:
                                                    num_nodes_E+num_nodes_Ias+num_nodes_Isy+num_nodes_Iho+num_nodes_R+num_nodes_Q]
        
        #make some agents deceased at start
        deceased_agents = not_susceptible_agents[num_nodes_E+num_nodes_Ias+num_nodes_Isy+num_nodes_Iho+num_nodes_R+num_nodes_Q:
                                                 num_nodes_E+num_nodes_Ias+num_nodes_Isy+num_nodes_Iho+num_nodes_R+num_nodes_Q+num_nodes_D]

        
        for i, node in enumerate(self.G.nodes()):
            a = MyAgent(i, self, self.G.nodes[i]['age_group']) 
            self.schedule.add(a)
            #add agent
            self.grid.place_agent(a, node)
            
            # check initial status (default SUSCEPTIBLE )
            if i in exposed_agents:
                initialize_EXPOSED(a, time = -1)
            
            if i in infected_agents_as: 
                a.state = State.INFECTED_AS
                initialize_INFECTED(a)
                
            if i in infected_agents_sy: 
                a.state = State.INFECTED_SY
                initialize_INFECTED(a)
                
            if i in infected_agents_ho: 
                a.state = State.INFECTED_HO
                initialize_INFECTED(a)
                           
            if i in recovered_agents: a.state = State.REMOVED
                
            if i in quarantined_agents: 
                a.state = State.QUARANTINED
                a.start_quarantine_time = self.schedule.time 
                
            if i in deceased_agents: a.state = State.DECEASED
                
                
    
                
                
    def update_parameters(self, pars):
                          
       # when called during step simulation it updates the parameters values
    
        ptrans_community_as, ptrans_other_as, ptrans_house_as, ptrans_work_as, ptrans_work_colleague_as, ptrans_school_as, ptrans_school_class_as, ptrans_school_friend_as, ptrans_community_sy, ptrans_other_sy, ptrans_house_sy, ptrans_work_sy, ptrans_work_colleague_sy, ptrans_school_sy, ptrans_school_class_sy, ptrans_school_friend_sy, ptrans_quarantine, ptrans_ho, pquar_as, pquar_sy, quarantine_time, time_to_hospitalization, time_to_death, time_to_quarantine_as, time_to_quarantine_sy = pars
        
        self.ptrans_community_as = ptrans_community_as
        self.ptrans_other_as = ptrans_other_as
        self.ptrans_house_as = ptrans_house_as        
        self.ptrans_work_as = ptrans_work_as
        self.ptrans_work_colleague_as = ptrans_work_colleague_as
        self.ptrans_school_as = ptrans_school_as 
        self.ptrans_school_class_as = ptrans_school_class_as
        self.ptrans_school_friend_as = ptrans_school_friend_as
        
        self.ptrans_community_sy = ptrans_community_sy
        self.ptrans_other_sy = ptrans_other_sy
        self.ptrans_house_sy = ptrans_house_sy        
        self.ptrans_work_sy = ptrans_work_sy
        self.ptrans_work_colleague_sy = ptrans_work_colleague_sy
        self.ptrans_school_sy = ptrans_school_sy
        self.ptrans_school_class_sy = ptrans_school_class_sy
        self.ptrans_school_friend_sy = ptrans_school_friend_sy
        
        self.ptrans_quarantine = ptrans_quarantine
        self.ptrans_ho = ptrans_ho
        
        self.pquar_as = pquar_as
        self.pquar_sy = pquar_sy
        
        self.quarantine_time = quarantine_time
        
        self.time_to_death = time_to_death
        self.time_to_hospitalization = time_to_hospitalization
        self.time_to_quarantine_as = time_to_quarantine_as
        self.time_to_quarantine_sy = time_to_quarantine_sy
        
        
        
        
    # DEFINE SIMULATION STEP
    def step(self):
        
        # update model params
        #self.update_parameters()
        
        # collect data (previous step)
        self.datacollector.collect(self)
        
        # update agents - execute scheduler step (of stages)
        self.schedule.step()
      
    
    '''
    # METHOD FOR MULTIRESOLUTION
    def infect_from_other_communities(self, mean_infection_other_communities):
        # pick some agents at random and make them infected (simulating they were exposed to virus in another community model)
        
        transportation_rate = 0.25 # cambia nel tempo.. es zona arancio
        
        susceptible_nodes = [agent 
                     for agent in self.grid.get_all_cell_contents() 
                     if agent.state is State.SUSCEPTIBLE]
        
        max_inf = int(mean_infection_other_communities*self.num_nodes*transportation_rate)
    
        infected_out = random.sample(susceptible_nodes, k = random.randint(0,max_inf))
        
        for agent in infected_out:
            initialize_EXPOSED(agent, time = self.schedule.time )
    '''
        


###############################################################################################################

# DEFINE SIMULATION         
        
        
def run_model(num_nodes, num_nodes_E, num_nodes_Ias, num_nodes_Isy, num_nodes_Iho, num_nodes_Q, num_nodes_R, num_nodes_D, 
              steps, change_times = None, change_pars = None, output_type = None, drop_S_R = False):
    
    model = NetworkInfectionModel(num_nodes, num_nodes_E, num_nodes_Ias, num_nodes_Isy, num_nodes_Iho, num_nodes_Q, num_nodes_R,num_nodes_D)
    
    idx = 0
    
    if change_times:
        change_time = change_times[idx]
    else:
        change_time = steps+1
    
    if output_type == 'Static': # plot compartment curves and simulation summary at the end
        
        R_t_s = []
        G_t_s = []
        
        for i in range(steps+1):
            
            # eventually update pars
            if i == change_time:
                model.update_parameters(change_pars[idx])
                
                idx+= 1
                try: change_time = change_times[idx]
                except: change_time = steps+1
            
            model.step()
            
            
        for i in range(steps+1):
            R_t_s.append(R_t(model.row, model.col, model.data, model.num_nodes, i + 1))
            G_t_s.append(generation_time_t(model.row, model.col, model.data, model.num_nodes, i + 1))

         
        print('SIMULATION SUMMARY:')
        print('Population size : ', model.num_nodes)
        #print('Clustering coefficient: ', round(nx.average_clustering(model.G), 3)) # not valid since random egdes added or actvated
        #print('Average degree :', compute_averagedegree(model.G))
        print('Reproduction number: min = {}, mean = {}, max = {}'.format(round(np.min(R_t_s), 3),
                                                                          round(np.mean(R_t_s), 3), round(np.max(R_t_s), 3)))
        print('Number of infections: ')
        tot_inf = np.sum(list(model.inf_counter.values()))
        for i in model.inf_counter.items():
            print('  {}: {} ({} %)'.format(i[0], i[1], round(100 * i[1]/tot_inf, 2)))
        print('Final size: ', round(final_size(model.row, model.col, model.data, model.num_nodes), 2), '%')
        print('Global Generation time: ', round(generation_time_tot(model.row, model.col, model.data, model.num_nodes), 3), ' steps')
        
        
        f, ax =plt.subplots(1,1,figsize=(8,4))
        plot_states(model,ax, drop_S_R)
        
        f, ax =plt.subplots(1,1,figsize=(8,4))
        ax.plot(range(steps+1), R_t_s)
        ax.set_title('Reproduction number')
        ax.set_ylabel('R_t')
        ax.set_xlabel('Step')
        
        f, ax =plt.subplots(1,1,figsize=(8,4))
        ax.plot(range(steps+1), G_t_s)
        ax.set_title('Generation Time')
        ax.set_ylabel('G_t')
        ax.set_xlabel('Step')
        
    else: 

        for i in range(steps+1):
            
            # eventually update pars
            if i == change_time:
                model.update_parameters(change_pars[idx])
                
                idx+= 1
                try: change_time = change_times[idx]
                except: change_time = steps+1
            
            model.step()
            
        X = get_column_data(model)
        
        solutiondf = pd.DataFrame()
    
        try : solutiondf['SUSCEPTIBLE'] = X['SUSCEPTIBLE'] 
        except: solutiondf['SUSCEPTIBLE'] = np.zeros(X.shape[0])
                 
        try : solutiondf['EXPOSED'] = X['EXPOSED'] 
        except: solutiondf['EXPOSED'] = np.zeros(X.shape[0])
            
        try : solutiondf['INFECTED_AS'] = X['INFECTED_AS']  
        except: solutiondf['INFECTED_AS'] = np.zeros(X.shape[0])
            
        try : solutiondf['INFECTED_SY'] = X['INFECTED_SY'] 
        except: solutiondf['INFECTED_SY'] = np.zeros(X.shape[0])
            
        try : solutiondf['INFECTED_HO'] = X['INFECTED_HO'] 
        except: solutiondf['INFECTED_HO'] = np.zeros(X.shape[0])
            
        try : solutiondf['QUARANTINED'] = X['QUARANTINED'] 
        except: solutiondf['QUARANTINED'] = np.zeros(X.shape[0])
            
        try : solutiondf['REMOVED'] = X['REMOVED'] 
        except: solutiondf['REMOVED'] = np.zeros(X.shape[0])
            
        try : solutiondf['DECEASED'] = X['DECEASED'] 
        except: solutiondf['DECEASED'] = np.zeros(X.shape[0])
            
        if output_type == 'Solutions':   # return complete solutions
            return solutiondf
        
        elif output_type == 'Last':   # return only last simulation step solution
            return solutiondf.iloc[-1, :].values
        
        else: print('ERROR: Specified output type not valid!')
        

        
        