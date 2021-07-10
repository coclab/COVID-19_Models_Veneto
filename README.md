# COVID-19 Models Veneto

## About

This repository contains the code produced for the final master thesis study on epidemiological models.

Author: *Claudia Cozzolino*

Title: *A data-driven epidemic model to analyse the course of COVID-19 in the Veneto region*

Supervisors: *Nicolò Navarin*, *Alessandro Sperduti*

Other contributers: *Vincenzo Baldo*, *Federico Zabeo*

Institution: *University of Padova*

Master degree: *Data Science*

Academic year: *2020-2021*

Abstract: 
>*The current COVID-19 pandemic is an unprecedented global health crisis, with severe economic impacts and social damages. Mathematical models are playing an important role in this
ongoing emergency, providing scientific support to inform public policies worldwide. In this
thesis work, an epidemic model for the spread of the novel Coronavirus disease in the Veneto
region has been proposed. Starting from the available local Health System data to examine
past year contagion numbers and other features potentiality, a SEIQRD (Susceptible Exposed
Infected Quarantined Removed Deceased) compartmental schema has been designed generalizing the classic SIR model. Then, the infection dynamics have been practically implemented
in two versions: as a Deterministic Equation-based formulation and as an Agent-based model.
While the former has been maintained simple and computationally inexpensive in order to
serve as a baseline and to quickly provide parameter estimates, for the latter a detailed metapopulation of agents with personalized attributes and network of contacts has been developed
to recreate as realistic as possible simulations. Once these models have been trained and validated, they could became valuable tools for various types of analysis and predictions. In particular, the agent-based version, thanks to its flexibility as well as to its higher resolution, could be
exploited for exclusive a posteriori evaluations of the effectiveness of the adopted containment
measures in reducing the pandemic in Veneto.*

<img width="1320" alt="image" src="https://user-images.githubusercontent.com/65338398/125161075-821e0100-e180-11eb-8016-da8e908307d6.png">



## Repository structure

### Code

* `ABmodel.py`
* `ABmodel_utils.py`
* `DETmodel.py`
* `COVID-19 models.ipynb.py`
* `fit_expDET.py`
* `fitABM_par.py`
* `expABM_par.py`

### Data
*  `Real_curves_dayly.csv` contains the dayly numbers of positive, quarantined, deceased (cumulative) and hospitalized subjects in Veneto from 2020 February 17 to 2021 February 24;
*  `param_estimates_ABM_8P` contains the estimated Deterministic model parameters obtained by fitting the epidemic curves on the Veneto observations;
*  `param_estimates_DET_8P` contains the estimated Agent-Based model parameters obtained by fitting the epidemic curves on the Veneto observations.


