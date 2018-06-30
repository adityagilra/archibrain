# Hybrid AuGMEnT

This is the code for the pre-print:  
Multi-timescale memory dynamics in a reinforcement learning network with attention-gated memory  
Marco Martinolli, Wulfram Gerstner, Aditya Gilra  
[arXiv:1712.10062 \[q-bio.NC\]](https://arxiv.org/abs/1712.10062)  
  
We extend the learning capability of the AuGMEnT network (Rombouts, Bohte, Roelfsema, 2015) by introducing multi-timescale leaky dynamics in the working memory.

## Instructions for the simulations

To run the augment simulations on task XXX type:
```
python3 AuGMEnT_XXX.py
```
where AuGMEnT_XXX.py is the main file.

The code will automatically construct the dataset importing the building functions from TASKS/task_XXX.py.
In addition, the network will be instantiated as an object of class AuGMEnT defined in file AuGMEnT_model.py.

In the main file you can define the network properties and set the model parameters.
Most important parameters: beta (learning rate), alpha (decay rate), leak (leaky coefficient).

N.B. The leak parameter can be either be a scalara or a list. In case it is a scalar, all M units in the memory decay with that parameter (e.g. leak=1.0 for standard AuGMEnT, leak=0.7 for leaky control). Otherwise, the memory is divided in equal parts (as many as the length of the list) with each subpart with different decay (e.g. leak=[0.7,1.0] for hybrid AuGMEnT).

Other standard simulation parameters are: N_sim (number of simulations), N_trial (number of maximum trials), stop (boolean wheter to stop or not after convergence), verb (boolean to enable verbose output or not),...

'further_analysis' contains additional analysis on performance of hybrid augment on 12AX-like tasks. See the inner readme.md for further details.

'DATA' folder collects all the data of convergence times, error trends, converged matrices and pre-defined training dataset.

'IMAGES' folder collects the most relevant images of performance analysis of AuGMEnT network and its variants on 12ax, saccade-antisaccade and sequence prediction tasks.
'Visualization Codes' contains the codes for most of the plots.

## Reference

* Rombouts, Jaldert O., Sander M. Bohte, and Pieter R. Roelfsema (2015). “How Attention Can Create Synaptic Tags for the Learning of Working Memories in Sequential Tasks”. In: PLOS Computational Biology 11.3, pp. 1–34. DOI: 10.1371/journal.pcbi.1004060. URL: https://doi.org/10.1371/journal.pcbi.1004060.
