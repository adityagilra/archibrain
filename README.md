# archibrain
We will develop biologically plausible neural network models based on brain architecture that solve cognitive tasks performed in the laboratory.  
  
Inspired by brain architecture, the machine learning community has recently developed various memory-augmented neural networks, that enable symbol and data manipulation tasks that are difficult with standard neural network approaches, see especially from Google Deepmind ([NTM](https://arxiv.org/abs/1410.5401), [DNC](http://www.nature.com/nature/journal/v538/n7626/abs/nature20101.html), [one-shot learner](https://arxiv.org/abs/1605.06065)).  
  
At the same time, models based closely on brain architecture, that perform experimentally-studied tasks, have existed for a while in the neuroscience community, notably from the labs of Eliasmith ([SPAWN](http://www.sciencemag.org/content/338/6111/1202)), O'Reilly ([PBWM](http://dx.doi.org/10.1162/089976606775093909)-[LEABRA-Emergent](http://www.colorado.edu/faculty/oreilly/research)), Alexander+Brown ([HER](http://dx.doi.org/10.1162/NECO_a_00779)), Roelfsema ([AuGMEnT](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004060)), Hawkins ([HTM](https://en.wikipedia.org/wiki/Hierarchical_temporal_memory)), and others ([Heeger et al 2017](http://www.pnas.org/content/114/8/1773.abstract.html?etoc), ...). How these models compare to each other on standard tasks is unclear. Further, biological plausibility of these models is quite variable.
  
From the neuroscience perspective, we want to figure out how the brain performs cognitive tasks, by synthesizing current models and tasks, constrained by known architecture and learning rules. From the machine learning perspective, we will explore whether brain-inspired architecture(s) can improve artificial intelligence (cf. copying bird flight didn't help build airplanes, but copying neurons helped machine learning).  
  
As part of this project, we introduced an extension of [AuGMEnT](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004060), called hybrid AuGMEnT, that incorporates multiple timescales of memory dynamics, enabling it to solve tasks like 12AX which the original AuGMEnT could not. A paper is being written up by Marco Martinolli, Wulfram Gerstner and Aditya Gilra, for which the code is available at [https://github.com/martin592/hybrid_AuGMEnT](https://github.com/martin592/hybrid_AuGMEnT).
  
We utilize a modular architecture to:
1) Specify the model such that we can 'plug and play' different modules -- controller, differentiable memories (multiple can be used at the same time). We should be able to interface both the abstract 'neurons' (LSTM, GRU, McCullough-Pitts, ReLU, ...) but also more biological spiking neurons.
2) Specify Reinforcement Learning or other tasks -- 1-2AX, Raven progressive matrices, BABI tasks...

We will also explore different memory interfacing schemes like content or list-based as in the DNC, or Plate's/Eliasmith's Holographic Reduced Representations/Semantic Pointer Architecture, address-value augmentation, etc.  
  
A larger goal will be to see if the synthesized 'network' can build models of the 'world' which generalize across tasks.
  
Currently, we have three contributors: [Marco Martinolli](https://github.com/martin592), [Vineet Jain](https://github.com/vineetjain96) and [Aditya Gilra](https://github.com/adityagilra). We are looking for more contributors!  
  
Aditya explores ideas and architectures and ways to synthesize them.  
  
Marco implemented the Hierarchical Error Representation (HER) model by Alexander and Brown (2015, 2016), incorporating hierarchical predictive coding and gated working memory structures, and the AuGMEnT model by Rombouts, Bohte and Roelfsema (2015), as well as the relevant tasks Saccade-AntiSaccade, 12AX, and sequence prediction tasks. See the extention of AuGMEnT,  [hybrid AuGMEnT]((https://github.com/martin592/hybrid_AuGMEnT)), developed by him as part of this project.  
  
Vineet developed a common API for models and tasks as well as implemented some tasks. He also tested various parts of the memory architectures of DNC and NTM whose code has been incorporated from their official repositories. See his [one shot learning implementation](https://github.com/vineetjain96/one-shot-mann), an offshoot of this project.  
  
See also:  
[Overview of architectures (work in progress)](https://github.com/adityagilra/archibrain/wiki/Brain-models---ML-architectures)  
[Brief survey of toolkits] (https://github.com/adityagilra/archibrain/wiki/Machine-Learning-library-comparisons-for-testing-brain-architectures). We have chosen [Keras](https://keras.io/) as a primary toolkit. We will follow an [agile software development](https://en.wikipedia.org/wiki/Agile_software_development) process, and frequently re-factor code (might even change the framework).
