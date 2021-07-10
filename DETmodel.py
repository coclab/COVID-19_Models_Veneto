

# imports
import os
import sys

import numpy as np
import pandas as pd

from scipy.integrate import odeint

import plotly.graph_objects as go

from typing import List, Tuple, Callable




def ode_model(z : List, t : float, pars : List, prob : List) -> List[float]:
    """
    Define the ODE system for SEIQRD epidemic model.
    :param z: a list with the initial conditions of the variables
    :param t: a number indicating the time units on which calculate the value of the ODE' solutions
    :param pars: a list of parameters 
    :param prob: a list of probabilities 
    """
    S, E, Ias, Isy, Iho, Q, R, D = z
    beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu = pars
    p_as, p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy = prob
    
    # system of equation SEIQRD epidemic model
    N = sum(z)
    I = Ias + Isy + Iho
    
    dSdt = -S/N*(beta_as*Ias + beta_sy*Isy + beta_ho*Iho + beta_quar*Q)
    dEdt = S/N*(beta_as*Ias + beta_sy*Isy + beta_ho*Iho + beta_quar*Q) - epsilon*E
    dIasdt = p_as*epsilon*E - (1 - pquar_as)*gamma_as*Ias - pquar_as*phi_as*Ias
    dIsydt = p_sy*epsilon*E - (1 - pquar_sy - pdie_sy - p_ho)*gamma_sy*Isy - pdie_sy*mu*Isy - pquar_sy*phi_sy*Isy - p_ho*psi*Isy
    dIhodt = p_ho*psi*Isy - (1 - pdie_ho)*gamma_ho*Iho - pdie_ho*mu*Iho
    dQdt = pquar_as*phi_as*Ias + pquar_sy*phi_sy*Isy - gamma_quar*Q 
    dRdt = (1 - pquar_as)*gamma_as*Ias + (1 - pquar_sy - pdie_sy - p_ho)*gamma_sy*Isy + (1-pdie_ho)*gamma_ho*Iho +  gamma_quar*Q 
    dDdt =  pdie_sy*mu*Isy + pdie_ho*mu*Iho
    
    return [dSdt, dEdt, dIasdt, dIsydt, dIhodt, dQdt, dRdt, dDdt]


def ode_solver(t, initial_conditions, params, prob):
    """
    Solve the ODE system for SEIQRD epidemic model.
    :param initial_conditions: a list with the initial conditions of the variables
    :param t: a number indicating the time units on which calculate the value of the ODE' solutions
    :param params: a list of parameters indicating, in order: beta, epsilon, gamma, ratequar, mu
    :param prob: a list of probabilities indicating, in order, prob of being asymptomatic/symptomatic/hospitalized/die,
       prob of being quarantined
    """
    
    initE, initIas, initIsy, initIho, initQ, initR, initN, initD = initial_conditions
    beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu = params
    p_as, p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy = prob
    
    initS = initN - (initE + initIas + initIsy + initIho + initQ + initR + initD)
    
    # solve system 
    res = odeint(ode_model, [initS, initE, initIas, initIsy, initIho, initQ, initR, initD], 
                 t, args=([beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu], 
                          [p_as, p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy]))
    return res



def main(initE, initIas, initIsy, initIho, initQ, initR, initD, initN, 
         beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu,
         days, 
         p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy, 
         output = 'Last'):
    
    """
    Return data or plot solutions of the ODE system for SEIQRD epidemic model.
    :param initE, initIas, initIsy, initIho, initQ, initR, initD, initN are the initial conditions of the variables
    :param beta, epsilon, gamma, ratequar, mu SEIQRD rate parameters 
    :param days: integer number indicating the time span on which calculate the ODE solutions
    :param psy, pho, pdie, pquar are the probabilities indicating, in order, prob of being asymptomatic/symptomatic/hospitalized/die,
         prob of being quarantined
    :param output: type of output to be returned, 'Last' for a list containg the last instant time value solutions, 'Solutions' 
        for a  dataframe with complete days solutions, 'Figure' for graphical and interactive plot
    """
    
    
    initial_conditions = [initE, initIas, initIsy, initIho, initQ, initR, initN, initD]
    
    params = [beta_as, beta_sy, beta_ho, beta_quar, epsilon, gamma_as, gamma_sy, gamma_ho, gamma_quar, phi_as, phi_sy, psi, mu]
    
    tspan = np.arange(0, days, 1)
    
    p_as = 1 - p_sy
    prob = p_as, p_sy, p_ho, pdie_sy, pdie_ho, pquar_as, pquar_sy
    
    # solve system
    sol = ode_solver(tspan, initial_conditions, params, prob)
    S, E, Ias, Isy, Iho, Q, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4], sol[:, 5], sol[:, 6], sol[:, 7]
    
    if output == 'Last': # return last day solutions
        return [sol[-1, 0], sol[-1, 1], sol[-1, 2], sol[-1, 3], sol[-1, 4], sol[-1, 5], sol[-1, 6], sol[-1, 7]]
    
    if output == 'Solutions': # return all days solution
        
        solutiondf = pd.DataFrame()
        solutiondf['SUSCEPTIBLE'] = S
        solutiondf['EXPOSED'] = E
        solutiondf['INFECTED_AS'] = Ias
        solutiondf['INFECTED_SY'] = Isy
        solutiondf['INFECTED_HO'] = Iho
        solutiondf['QUARANTINED'] = Q
        solutiondf['REMOVED'] = R
        solutiondf['DECEASED'] = D
        return solutiondf
    
    if output == 'Figure': # plot solutions
        # Create traces
        fig = go.Figure()
        #fig.add_trace(go.Scatter(x=tspan, y=S/initN, mode='lines+markers', name='Susceptible'))
        fig.add_trace(go.Scatter(x=tspan, y=E/initN, mode='lines+markers', name='Exposed'))
        fig.add_trace(go.Scatter(x=tspan, y=Ias/initN, mode='lines+markers', name='Infected asymptomatic'))
        fig.add_trace(go.Scatter(x=tspan, y=Isy/initN, mode='lines+markers', name='Infected symptomatic'))
        fig.add_trace(go.Scatter(x=tspan, y=Iho/initN, mode='lines+markers', name='Infected hospitalized'))
        fig.add_trace(go.Scatter(x=tspan, y=Q/initN, mode='lines+markers', name='Quarantined'))
        #fig.add_trace(go.Scatter(x=tspan, y=R/initN, mode='lines+markers',name='Recovered'))
        fig.add_trace(go.Scatter(x=tspan, y=D/initN, mode='lines+markers',name='Death'))

        if days <= 30:
            step = 1
        elif days <= 90:
            step = 7
        else:
            step = 30

        # Edit the layout
        fig.update_layout(title='Simulation of SEIQRD Model',
                           xaxis_title='Day',
                           yaxis_title='Counts',
                           title_x=0.5,
                          width=900, height=600
                         )
        fig.update_xaxes(tickangle=-90, tickformat = None, tickmode='array', tickvals=np.arange(0, days + 1, step))
        fig.show()
        
        
