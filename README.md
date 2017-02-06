# archibrain
Synthesize brain and neural network models of cognition

Inspired by brain architecture, Google DeepMind has recently coupled an external memory to a neural computer, enabling symbol and data manipulation tasks that are difficult with standard neural network approaches [Graves, et al. Nature Oct 2016]. At the same time, models based closely on brain architecture have existed for a while in the neuroscience community, notably from the labs of Eliasmith (SPAWN), O'Reilly (PBWM-LEABRA-Emergent), Hawkins (HTM), Alexander+Brown (HER), Roelfsema (AuGMEnT), and others. How these models compare to each other on standard tasks is unclear. Further, biological plausibility of these models is quite variable.

From the neuroscience perspective, we want to figure out how the brain performs cognitive tasks, by synthesizing current models and tasks, constrained by known architecture and learning rules,  From the machine learning perspective, we will explore whether brain-inspired architecture(s) can improve artificial intelligence (cf. copying bird flight didn't help build airplanes).

We will utilize a modular architecture to:
1) Specify the model such that we can 'plug and play' different modules -- controller, differentiable memories (multiple can be used at the same time). We should be able to interface both the abstract 'neurons' (LSTM, GRU, McCullough-Pitts, ReLU, ...) but also more biological spiking neurons.
2) Specify Reinforcement Learning or other tasks -- 1-2AX, Raven progressive matrices, BABI tasks...

We will also explore different memory interfacing schemes like content or list-based as in the DNC, or Eliasmith's HHR, address-value augmentation, etc.

A larger goal will be to see if the synethesized 'network' can build models of the world which generalize across tasks.

Mini-goal 1 (Aditya):
Read and understand various architectures. Explore current machine learning toolkits and figure out which one will be most suitable for our task.

Mini-goal 2 (Marco):
The Hierarchical Error Representation (HER) model by Alexander and Brown (2015, 2016), incorporating hierarchical predictive coding and gated working memory structures, is the most comparable to DeepMind's Differentiable Neural Computer (DNC). Thus, in this project, you will: (1) implement the DNC and the HER models using a standard machine learning toolkit (preferably python-based like Theano/Keras); (2) benchmark the two architectures on similar tasks; (3) incorporate the best features of the two models (and any others) into a hopefully better (faster learning / better performing / more biologically plausible) model.

Mini-goal 3 (Vineet):
To be determined.
